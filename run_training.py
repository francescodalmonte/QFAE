import os
import time
import configparser
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import CustomBMADTrainDataset_oneclass, CustomBMADTestDataset, custom_collate_fn
from dataset import ImagenetteDataset, InfiniteDataLoader
from utils import save_config_info, plot_examples, plot_reconstruction_examples
from utils import plot_anomaly_histogram, plot_reconstruction_examples_fullreso
from utils import plot_attn_maps, plot_loss_curves
from model import MQViTAE, create_masks
from training import train_step, eval_step
from losses import criterion_selector, multicriterion_selector



def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setupArgs():
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()

    parser.add_argument("--config",
                        type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.INI"),
                        help="Path to the config file.")
    config_path = parser.parse_args().config

    if os.path.exists(config_path):
        config.read(config_path)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return config


def train_model():

    # setup args
    params = setupArgs()["TRAINING"]

    # general
    seed = int(params["SEED"])
    experiment_name = params["EXPERIMENT_NAME"]
    train_data_path = params["TRAIN_DATA_PATH"]
    val_data_path = params["VAL_DATA_PATH"]
    model_ckpt_path = params["MODEL_CKPT_PATH"].strip()
    if len(model_ckpt_path) == 0:
        model_ckpt_path = None
    results_path = params["RESULTS_PATH"]
    eval_stepsize = int(params["EVAL_STEPSIZE"])
    save_model_ckpt = bool(int(params["SAVE_MODEL_CKPT"]))
    max_samples_train = int(params["MAX_SAMPLES_TRAIN"])
    max_samples_val = int(params["MAX_SAMPLES_VAL"])
    device = torch.device(params["DEVICE"])
    num_w = int(params["NUM_WORKERS"])

    # encoder
    model_name = [x.strip() for x in params["MODEL_NAME"].split(",")]
    cache_dir = [x.strip() for x in params["CACHE_DIR"].split(",")]

    n_patches = [int(x.strip()) for x in params["N_PATCHES"].split(",")]
    patch_size = [int(x.strip()) for x in params["PATCH_SIZE"].split(",")]
    n_reg_tokens = [int(x.strip()) for x in params["N_REG_TOKENS"].split(",")]
    if len(params["FINAL_PROJ_IN_FEATURES"].strip())*len(params["FINAL_PROJ_OUT_FEATURES"].strip()) == 0:
        final_proj_params = [None]*len(model_name)
    else:
        final_proj_params = [{"in_features": int(in_f.strip()), "out_features": int(out_f.strip()), "bias": True}
                             for in_f, out_f in zip(params["FINAL_PROJ_IN_FEATURES"].split(","), params["FINAL_PROJ_OUT_FEATURES"].split(","))]
    if len(params["ADDITIONAL_BLOCK_DIM"].strip())*len(params["ADDITIONAL_BLOCK_NUM_HEADS"].strip())*len(params["ADDITIONAL_BLOCK_MLP_RATIO"].strip()) == 0:
        additional_block_params = None
    else:
        additional_block_params = {"dim": int(params["ADDITIONAL_BLOCK_DIM"]),
                                   "num_heads": int(params["ADDITIONAL_BLOCK_NUM_HEADS"]),
                                   "mlp_ratio": float(params["ADDITIONAL_BLOCK_MLP_RATIO"])}
    if params["USE_HIDDEN_STATE"].strip() == "-1":
        use_hidden_state = None
    else:
        use_hidden_state = [int(x.strip()) for x in params["USE_HIDDEN_STATE"].split(",")]

    # junction
    junction_n_blocks = int(params["JUNCTION_N_BLOCKS"])
    junction_dim = int(params["JUNCTION_DIM"])
    junction_output_dim = int(params["JUNCTION_OUTPUT_DIM"])
    junction_n_queries = int(params["JUNCTION_N_QUERIES"])
    junction_heads = int(params["JUNCTION_HEADS"])
    junction_mlp_ratio = float(params["JUNCTION_MLP_RATIO"])

    # decoder
    decoder_dim = int(params["DECODER_DIM"])
    decoder_n_patches = int(params["DECODER_N_PATCHES"])
    decoder_patch_size = int(params["DECODER_PATCH_SIZE"])
    decoder_n_reg_tokens = int(params["DECODER_N_REG_TOKENS"])
    decoder_depth = int(params["DECODER_DEPTH"])
    decoder_num_heads = int(params["DECODER_NUM_HEADS"])
    decoder_mlp_ratio = int(params["DECODER_MLP_RATIO"])
    decoder_out_channels = int(params["DECODER_OUT_CHANNELS"])

    # optimization and evaluation
    criterion_train_name = [x.strip() for x in params["CRITERION_TRAIN"].split(",")]
    criterion_train_weights = [float(x.strip()) for x in params["CRITERION_TRAIN_WEIGHTS"].split(",")]
    criterion_val_name = [x.strip() for x in params["CRITERION_VAL"].split(",")]
    criterion_val_weights = [float(x.strip()) for x in params["CRITERION_VAL_WEIGHTS"].split(",")]
    perceptual_metric_train = params["PERCEPTUAL_METRIC_TRAIN"].strip()
    perceptual_metric_val = params["PERCEPTUAL_METRIC_VAL"].strip()
    perceptual_hstate_train = [int(x.strip()) for x in params["PERCEPTUAL_HSTATE_TRAIN"].split(",")]
    perceptual_hstate_val = [int(x.strip()) for x in params["PERCEPTUAL_HSTATE_VAL"].split(",")]
    reduction = params["REDUCTION"].strip()
    resize = int(params["RESIZE"])
    batch_size_train = int(params["BATCH_SIZE_TRAIN"])
    batch_size_test = int(params["BATCH_SIZE_TEST"])
    lr = float(params["LR"])
    scheduler_steps = [int(x.strip()) for x in params["SCHEDULER_STEPS"].split(",")]
    scheduler_gamma = float(params["SCHEDULER_GAMMA"])
    n_epochs = int(params["N_EPOCHS"])

    # Parse perceptual patch sizes if specified
    perceptual_patch_sizes_train = None
    if "PERCEPTUAL_PATCH_SIZE_TRAIN" in params and params["PERCEPTUAL_PATCH_SIZE_TRAIN"].strip():
        perceptual_patch_sizes_train = [int(x.strip()) for x in params["PERCEPTUAL_PATCH_SIZE_TRAIN"].split(",")]

    perceptual_patch_sizes_val = None
    if "PERCEPTUAL_PATCH_SIZE_VAL" in params and params["PERCEPTUAL_PATCH_SIZE_VAL"].strip():
        perceptual_patch_sizes_val = [int(x.strip()) for x in params["PERCEPTUAL_PATCH_SIZE_VAL"].split(",")]

    # Parse multi hstate reduction if specified
    multi_hstate_reduction = "max"  # default value
    if "MULTI_HSTATE_REDUCTION" in params and params["MULTI_HSTATE_REDUCTION"].strip():
        multi_hstate_reduction = params["MULTI_HSTATE_REDUCTION"].strip()

    # fix seeds
    fix_seeds(seed)

    # print device
    print(f"Running on device: {device}")
    # save config file
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    save_config_info(os.path.join(results_path, f"{experiment_name}_config.txt"), params)

    # 0. DATA

    # define data transforms
    train_transforms = v2.Compose([
        v2.Resize((resize, resize)),
        v2.RandomResizedCrop(resize, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
        v2.RandomRotation(degrees=10),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        #v2.Grayscale(num_output_channels=3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.449], std=[0.226])
        ])
    test_transforms = v2.Compose([
        v2.Resize((resize, resize)),
        #v2.Grayscale(num_output_channels=3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.449], std=[0.226])
        ])

    # load datasets
    print("Loading datasets...")
    train_ds = CustomBMADTrainDataset_oneclass(train_data_path,
                                               train_transforms,
                                               max_samples=max_samples_train,
                                               seed=seed)
    val_ds = CustomBMADTestDataset(val_data_path,
                                   test_transforms,
                                   max_samples=max_samples_val,
                                   seed=seed)
                                    
    train_dl = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True,
                          num_workers=num_w, collate_fn=custom_collate_fn, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size_test, shuffle=False,
                        num_workers=num_w, collate_fn=custom_collate_fn, pin_memory=True,
                        drop_last=True)

    
    print(f"TRAIN dataset/loader length: {len(train_ds)}/{len(train_dl)}")
    print(f"VAL dataset/loader length: {len(val_ds)}/{len(val_dl)}")

    # save imgs examples
    #plot_examples(train_dl, N=4, save_to=os.path.join(results_path, f"{experiment_name}_train_examples.png"))
    #plot_examples(val_dl, N=4, save_to=os.path.join(results_path, f"{experiment_name}_test_examples.png")) 

    # 1. MODEL

    # instantiate model
    print(f"Loading vision model... ({model_name})")
    model = MQViTAE(model_name=model_name,
                    cache_dir=cache_dir,
                    img_size=resize,
                    n_patches=n_patches,
                    patch_size=patch_size,
                    n_reg_tokens=n_reg_tokens,
                    junction_n_blocks=junction_n_blocks,
                    junction_dim=junction_dim,
                    junction_output_dim=junction_output_dim,
                    junction_n_queries=junction_n_queries,
                    junction_heads=junction_heads,
                    junction_mlp_ratio=junction_mlp_ratio,
                    decoder_dim=decoder_dim,
                    decoder_n_patches=decoder_n_patches,
                    decoder_patch_size=decoder_patch_size,
                    decoder_depth=decoder_depth,
                    decoder_num_heads=decoder_num_heads,
                    decoder_mlp_ratio=decoder_mlp_ratio,
                    decoder_out_channels=decoder_out_channels,
                    decoder_n_reg_tokens=decoder_n_reg_tokens,
                    additional_block_params=additional_block_params,
                    final_projection_params=final_proj_params,
                    use_hidden_state=use_hidden_state)
    model.to(device)

    # load model checkpoint if provided
    if model_ckpt_path is not None:
        print(f"Loading model checkpoint from: {model_ckpt_path}")
        model.load_state_dict(torch.load(model_ckpt_path, map_location=device))

    # frozen part
    for e in model.encoders:
        e.model.eval()
        e.model.requires_grad_(False)

    # 2. OPTIMIZATION

    # cross-attention masks for training
    #crossattn_masks = create_masks(junction_n_queries**0.5,
    #                               np.repeat(np.array(n_patches), len(use_hidden_state)),
    #                               np.repeat(np.array(n_reg_tokens), len(use_hidden_state)),
    #                               device="cpu")
    crossattn_masks = None

    # criterion
    kwargs_train = {
        "cache_dir": cache_dir[0],
        "img_size": resize,
        "device": device,
        "perceptual_metric": perceptual_metric_train,
        "perceptual_hstate": perceptual_hstate_train
    }
    # Add perceptual patch sizes if specified
    if perceptual_patch_sizes_train is not None:
        kwargs_train["perceptual_patch_sizes"] = perceptual_patch_sizes_train

    criterion_train = multicriterion_selector(criterion_train_name,
                                              weights=criterion_train_weights,
                                              **kwargs_train)
    kwargs_eval = {
        "cache_dir": cache_dir[0],
        "img_size": resize,
        "device": device,
        "perceptual_metric": perceptual_metric_val,
        "perceptual_hstate": perceptual_hstate_val
    }
    # Add perceptual patch sizes if specified
    if perceptual_patch_sizes_val is not None:
        kwargs_eval["perceptual_patch_sizes"] = perceptual_patch_sizes_val

    criterion_val = multicriterion_selector(criterion_val_name,
                                            weights=criterion_val_weights,
                                            **kwargs_eval)

    # optimizer
    params_encoder = []
    for e in model.encoders:
        params_encoder += list(e.additional_block.parameters())
        params_encoder += list(e.final_projection.parameters())
    params_junction = list(model.junction.parameters())
    params_decoder = list(model.decoder.parameters())

    print(f"N. params encoder: {sum(p.numel() for p in params_encoder)}")
    print(f"N. params junction: {sum(p.numel() for p in params_junction)}")
    print(f"N. params decoder: {sum(p.numel() for p in params_decoder)}")

    params = params_encoder + params_junction + params_decoder
    optimizer = torch.optim.Adam(params, lr=lr)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, scheduler_steps, scheduler_gamma
    #    )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, 
        total_steps=n_epochs*len(train_dl), pct_start=0.15
        )


    # 3. TRAINING

    for epoch in range(n_epochs):
        epoch_time = time.time()
        print(f"Epoch: {epoch}")

        # TRAIN STEP
        for e in model.encoders:
            e.additional_block.train()
            e.final_projection.train()
        model.junction.train()
        model.decoder.train()

        out = train_step(model,
                         train_dl,
                         optimizer,
                         criterion_train,
                         device,
                         verbose=True,
                         lr_scheduler=lr_scheduler, # pass lr_scheduler here if using OneCycleLR    
                         auxilary_dl=None,
                         crossattn_masks=crossattn_masks)

        current_queries = model.junction.queries 
        print(f"Queries max norm: {torch.norm(current_queries, dim=1).max().item():.5f}", end=" | ")
        
        row = [epoch] + [out[k] for k in out.keys()]

        if epoch % eval_stepsize == 0 or epoch == n_epochs-1:
            # VAL STEP
            for e in model.encoders:
                e.additional_block.eval()
                e.final_projection.eval()
            model.junction.eval()
            model.decoder.eval()

            out_val = eval_step(model,
                                #train_dl, # for overfit experiments
                                val_dl,
                                criterion_val,
                                device,
                                store_data=False,
                                image_reduction=reduction,
                                image_multi_hstate_reduction=multi_hstate_reduction,
                                pixel_reduction=reduction,
                                pixel_multi_hstate_reduction=multi_hstate_reduction)
            
            
            # # Prepare hidden state information for plotting
            # hstate_info = {
            #     'perceptual_hstates': perceptual_hstate_val,
            #     'criterion_names': criterion_val_name
            # }
            
            # # save anomaly histogram
            # save_hist = os.path.join(results_path, "hist_anomaly")
            # if not os.path.exists(save_hist):
            #     os.mkdir(save_hist)
            # plot_anomaly_histogram(out_val["ascores"],
            #                     out_val["labels"],
            #                     save_to=os.path.join(save_hist, f"{experiment_name}_ahist_epoch{epoch}.png"))
            # # save examples of reconstruction
            # save_reco = os.path.join(results_path, "reco_imgs")
            # if not os.path.exists(save_reco):
            #     os.mkdir(save_reco)
            # plot_reconstruction_examples(out_val["x"],
            #                             out_val["x_hat"],
            #                             out_val["labels"],
            #                             out_val["ascores"],
            #                             out_val["anomaly_maps_tot"],
            #                             N=12,
            #                             save_to=os.path.join(save_reco, f"{experiment_name}_reco_epoch{epoch}.png"),
            #                             hstate_info=hstate_info,
            #                             reduction=reduction,
            #                             multi_hstate_reduction=multi_hstate_reduction)
            # # save full resolution examples
            # save_fullreso = os.path.join(results_path, "reco_imgs_fullreso")
            # if not os.path.exists(save_fullreso):
            #     os.mkdir(save_fullreso)
            #     os.mkdir(os.path.join(save_fullreso, "good"))
            #     os.mkdir(os.path.join(save_fullreso, "anomalous"))
            # plot_reconstruction_examples_fullreso(out_val["x"],
            #                                       out_val["x_hat"],
            #                                       out_val["labels"],
            #                                       out_val["anomaly_maps_tot"],
            #                                       N=4,
            #                                       save_to=save_fullreso,
            #                                       hstate_info=hstate_info)
            # # save attention maps
            # plot_attn_maps(out_val["attention_maps"],
            #                save_to=os.path.join(results_path, f"attn_maps.png"),
            #                N=4)

            # save model checkpoint
            if save_model_ckpt:
                save_ckpt = os.path.join(results_path, "checkpoints")
                if not os.path.exists(save_ckpt):
                    os.mkdir(save_ckpt)
                torch.save(model.state_dict(), os.path.join(save_ckpt, f"last_ckpt.pth"))
                print(f"(ckpt saved)", end=" | ")

            row = row + [out_val[f"val_rec_loss_{c}"] for c in np.unique(out_val["labels"])] + [out_val["auroc"]]
        else:
            # fill with nans in case of no evaluation
            row = row + [np.nan for c in np.unique(out_val["labels"])] + [np.nan]
        row = row + [lr_scheduler.get_last_lr()[0]] 

        # write to .csv file
        csv_path = os.path.join(results_path, f"{experiment_name}_training.csv")
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            if epoch == 0:
                colnames = ["epoch"] + list(out.keys()) + [f"val_rec_loss_{c}" for c in np.unique(out_val["labels"])] + ["AUROC"] + ["LR"]
                writer.writerow(colnames)
            writer.writerow(row)

        # plot losses 
        if epoch>0: # skip first epoch
            save_loss_curves = os.path.join(results_path, "plots.png")
            plot_loss_curves(csv_path, save_to=save_loss_curves)
        
        # update lr
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}", end=" | ")
        #lr_scheduler.step() # comment if lr_scheduler is passed to train_step

        print(f"Elapsed: {time.time()-epoch_time:.2f}s.")


if __name__ == "__main__":
    start = time.time()

    train_model()

    print(f"Execution time: {time.time()-start:.2f} seconds.")
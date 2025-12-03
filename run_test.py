import os
import time
import configparser
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
import csv
from pytorch_msssim import ssim

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import CustomBMADTestDataset, custom_collate_fn
from utils import (save_config_info, plot_examples, plot_reconstruction_examples, 
                  plot_anomaly_histogram, plot_reconstruction_examples_fullreso,
                  plot_pixel_level_results, compute_detailed_pixel_metrics, 
                  save_pixel_level_predictions)
from model import MQViTAE
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
    
    parser.add_argument("--compute_pixel_metrics",
                        action="store_true",
                        help="Whether to compute pixel-level segmentation metrics")
    
    args = parser.parse_args()
    config_path = args.config

    if os.path.exists(config_path):
        config.read(config_path)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return config, args


def test_model():

    # setup args
    config, args = setupArgs()
    params = config["TESTING"]

    # general
    seed = int(params["SEED"])
    experiment_name = params["EXPERIMENT_NAME"]
    checkpoint_path_model = params["CHECKPOINT_PATH"]
    test_data_path = params["TEST_DATA_PATH"]
    results_path = params["RESULTS_PATH"]
    max_samples_test = int(params["MAX_SAMPLES_TEST"])
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
    criterion_val_name = [x.strip() for x in params["CRITERION_VAL"].split(",")]
    criterion_val_weights = [float(x.strip()) for x in params["CRITERION_VAL_WEIGHTS"].split(",")]
    perceptual_metric_val = params["PERCEPTUAL_METRIC_VAL"].strip()
    perceptual_hstate_val =  [int(x.strip()) for x in params["PERCEPTUAL_HSTATE_VAL"].split(",")]
    
    # Separate reduction parameters for Image and Pixel analysis
    # Image-level reduction parameters
    image_reduction = params.get("IMAGE_REDUCTION", params.get("REDUCTION", "mean")).strip()
    image_multi_hstate_reduction = params.get("IMAGE_MULTI_HSTATE_REDUCTION", params.get("MULTI_HSTATE_REDUCTION", "max")).strip()
    
    # Pixel-level reduction parameters
    pixel_reduction = params.get("PIXEL_REDUCTION", params.get("REDUCTION", "mean")).strip()
    pixel_multi_hstate_reduction = params.get("PIXEL_MULTI_HSTATE_REDUCTION", params.get("MULTI_HSTATE_REDUCTION", "max")).strip()
    
    resize = int(params["RESIZE"])
    batch_size_test = int(params["BATCH_SIZE_TEST"])

    # Parse perceptual patch sizes if specified
    perceptual_patch_sizes_val = None
    if "PERCEPTUAL_PATCH_SIZE_VAL" in params and params["PERCEPTUAL_PATCH_SIZE_VAL"].strip():
        perceptual_patch_sizes_val = [int(x.strip()) for x in params["PERCEPTUAL_PATCH_SIZE_VAL"].split(",")]

    # fix seeds
    fix_seeds(seed)

    # print device and reduction settings
    print(f"Running on device: {device}")
    print(f"Image-level reduction: {image_reduction}, multi-hstate: {image_multi_hstate_reduction}")
    print(f"Pixel-level reduction: {pixel_reduction}, multi-hstate: {pixel_multi_hstate_reduction}")
    print("NOTE: For pixel-level with C=1, only 3 unique combinations exist:")
    print("  1. multi_hstate_reduction='mean' (smooth averaging)")
    print("  2. multi_hstate_reduction='max' (strong signal selection)")
    print("  3. reduction='downsample_8x8' (patch-level processing)")
    
    # save config file
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    save_config_info(os.path.join(results_path, f"{experiment_name}_config.txt"), params)

    # 0. DATA

    # define data transforms
    test_transforms = v2.Compose([
        v2.Resize((resize, resize)),
        #v2.Grayscale(num_output_channels=3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.449], std=[0.226])
        ])

    # load datasets
    print("Loading datasets...")
    test_ds = CustomBMADTestDataset(test_data_path,
                                    test_transforms,
                                    max_samples=max_samples_test,
                                    seed=seed)
    test_dl = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False,
                         num_workers=num_w, collate_fn=custom_collate_fn, pin_memory=False,
                         drop_last=True)
    
    print(f"TEST dataset/loader length: {len(test_ds)}/{len(test_dl)}")

    # 1. MODEL

    # instantiate memViTAE model
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

    # load checkpoint 
    print(f"Loading checkpoint: {checkpoint_path_model}")
    model.load_state_dict(torch.load(checkpoint_path_model, weights_only=False, map_location=device))

    # freeze model
    model.eval()

    # 2. TESTING

    # criterion
    kwargs_eval = {
        "cache_dir": "/shared-local/fdalmonte/cached/",
        "img_size": resize,
        "device": device,
        "perceptual_metric": perceptual_metric_val,
        "perceptual_hstate": perceptual_hstate_val
    }

    # Add perceptual patch sizes if specified
    if perceptual_patch_sizes_val is not None:
        kwargs_eval["perceptual_patch_sizes"] = perceptual_patch_sizes_val
        print(f"Using custom perceptual patch sizes: {perceptual_patch_sizes_val}")

    criterion_val = multicriterion_selector(criterion_val_name,
                                            weights=criterion_val_weights,
                                            **kwargs_eval)

    with torch.no_grad():
        print("Running test...")
        # VAL STEP with separate reduction parameters
        out_val = eval_step(model,
                            test_dl,
                            criterion_val,
                            device, 
                            verbose=True,
                            store_data=True,
                            image_reduction=image_reduction,
                            image_multi_hstate_reduction=image_multi_hstate_reduction,
                            pixel_reduction=pixel_reduction,
                            pixel_multi_hstate_reduction=pixel_multi_hstate_reduction,
                            compute_pixel_metrics=args.compute_pixel_metrics)
        
        # Prepare hidden state information for plotting
        hstate_info = {
            'perceptual_hstates': perceptual_hstate_val,
            'criterion_names': criterion_val_name
        }

        # save anomaly histogram
        save_hist = os.path.join(results_path, "hist_anomaly")
        if not os.path.exists(save_hist):
            os.makedirs(save_hist, exist_ok=True)
        plot_anomaly_histogram(out_val["ascores"],
                               out_val["labels"],
                               save_to=os.path.join(save_hist, f"{experiment_name}_ahist.png"))
        
        # save examples of reconstruction
        save_reco = os.path.join(results_path, "reco_imgs")
        if not os.path.exists(save_reco):
            os.makedirs(save_reco, exist_ok=True)
            os.makedirs(os.path.join(save_reco, "good"), exist_ok=True)
            os.makedirs(os.path.join(save_reco, "anomalous"), exist_ok=True)
        
        plot_reconstruction_examples(out_val["x"],
                                     out_val["x_hat"],
                                     out_val["labels"],
                                     out_val["ascores"],
                                     out_val["anomaly_maps_tot"],
                                     N=16,
                                     save_to=os.path.join(save_reco, f"reco_multi.png"),
                                     hstate_info=hstate_info,
                                     reduction=image_reduction,
                                     multi_hstate_reduction=image_multi_hstate_reduction)

        # save full resolution examples
        save_fullreso = os.path.join(results_path, "reco_imgs_fullreso")
        if not os.path.exists(save_fullreso):
            os.makedirs(save_fullreso, exist_ok=True)
            os.makedirs(os.path.join(save_fullreso, "good"), exist_ok=True)
            os.makedirs(os.path.join(save_fullreso, "anomalous"), exist_ok=True)
        plot_reconstruction_examples_fullreso(out_val["x"],
                                                out_val["x_hat"],
                                                out_val["labels"],
                                                out_val["anomaly_maps_tot"],
                                                N=4,
                                                save_to=save_fullreso,
                                                hstate_info=hstate_info)

        # write to .csv file
        csv_path = os.path.join(results_path, f"{experiment_name}_training.csv")
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f"val_rec_loss_{c}" for c in np.unique(out_val["labels"])] + ["AUROC"])
            row = [[out_val[f"val_rec_loss_{c}"] for c in np.unique(out_val["labels"])] + [out_val["auroc"]]] 
            writer.writerow(row)

        # Save pixel-level visualizations if pixel metrics were computed
        if args.compute_pixel_metrics and "masks" in out_val and not np.isnan(out_val.get("pixel_auroc", np.nan)):
            print("Saving pixel-level visualizations...")
            
            # Create pixel-level results directory
            save_pixel = os.path.join(results_path, "pixel_level")
            if not os.path.exists(save_pixel):
                os.makedirs(save_pixel, exist_ok=True)
            
            # Plot pixel-level results (overlay of GT vs predictions)
            plot_pixel_level_results(
                out_val["anomaly_maps_tot"],
                out_val["masks"],
                out_val["labels"],
                reduction=pixel_reduction,
                multi_hstate_reduction=pixel_multi_hstate_reduction,
                N=8,
                save_to=os.path.join(save_pixel, f"{experiment_name}_pixel_overlay.png")
            )
            
            # Compute detailed metrics and plot summary
            detailed_metrics = compute_detailed_pixel_metrics(
                out_val["anomaly_maps_tot"],
                out_val["masks"],
                out_val["labels"],
                reduction=pixel_reduction,
                multi_hstate_reduction=pixel_multi_hstate_reduction
            )
            
            # Save individual prediction maps
            save_pixel_level_predictions(
                out_val["anomaly_maps_tot"],
                out_val["masks"],
                out_val["labels"],
                reduction=pixel_reduction,
                multi_hstate_reduction=pixel_multi_hstate_reduction,
                save_dir=os.path.join(save_pixel, "individual_predictions"),
                threshold=detailed_metrics.get("optimal_threshold", None)
            )
            
            # Save detailed metrics to text file
            metrics_txt_path = os.path.join(save_pixel, f"{experiment_name}_detailed_metrics.txt")
            with open(metrics_txt_path, 'w') as f:
                f.write("PIXEL-LEVEL DETAILED METRICS\n")
                f.write("="*40 + "\n")
                for key, value in detailed_metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\nReduction settings:\n")
                f.write(f"Image-level reduction: {image_reduction}, multi-hstate: {image_multi_hstate_reduction}\n")
                f.write(f"Pixel-level reduction: {pixel_reduction}, multi-hstate: {pixel_multi_hstate_reduction}\n")

        # write to .csv file with both detection and segmentation metrics
        csv_path = os.path.join(results_path, f"{experiment_name}_results.csv")
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            
            # Create headers
            headers = [f"val_rec_loss_{c}" for c in np.unique(out_val["labels"])] + ["Image_AUROC"]
            if args.compute_pixel_metrics and not np.isnan(out_val.get("pixel_auroc", np.nan)):
                headers.append("Pixel_AUROC")
                headers.append("Combined_Score")  # Sum of both metrics like in reference
            
            # Add reduction info to headers
            headers.extend([
                "Image_Reduction", "Image_Multi_Hstate_Reduction", 
                "Pixel_Reduction", "Pixel_Multi_Hstate_Reduction"
            ])
            
            writer.writerow(headers)
            
            # Create data row
            row = [out_val[f"val_rec_loss_{c}"] for c in np.unique(out_val["labels"])] + [out_val["auroc"]]
            if args.compute_pixel_metrics and not np.isnan(out_val.get("pixel_auroc", np.nan)):
                pixel_auroc = out_val["pixel_auroc"]
                combined_score = out_val["auroc"] + pixel_auroc  # Like in reference implementation
                row.extend([pixel_auroc, combined_score])
            
            # Add reduction settings to row
            row.extend([
                image_reduction, image_multi_hstate_reduction,
                pixel_reduction, pixel_multi_hstate_reduction
            ])
            
            writer.writerow(row)

        # Print final results summary
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        print(f"Image-level AUROC: {out_val['auroc']:.4f} (reduction: {image_reduction}, multi-hstate: {image_multi_hstate_reduction})")
        
        if args.compute_pixel_metrics and not np.isnan(out_val.get("pixel_auroc", np.nan)):
            print(f"Pixel-level AUROC: {out_val['pixel_auroc']:.4f} (reduction: {pixel_reduction}, multi-hstate: {pixel_multi_hstate_reduction})")
            print(f"Combined Score: {out_val['auroc'] + out_val['pixel_auroc']:.4f}")
        else:
            print("Pixel-level metrics: Not computed or no anomalous samples with masks found")
        
        print("="*50)


if __name__ == "__main__":
    start = time.time()

    test_model()

    print(f"Execution time: {time.time()-start:.2f} seconds.")
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn.functional as F


def _scale(x, mean=0.449, std=0.226):
    """Invert normalization (scale back to [0,1])."""
    return np.clip(x*std + mean, 0., 1.)


def save_config_info(path, config_dict):
    print("---------- TRAIN INFO ----------")
    with open(path, 'w') as file:
        for k in config_dict.keys():
            file.writelines([k, " >>> ", config_dict[k], '\n'])
            print(f"{k}  :  {config_dict[k]}")
    print("-------------------------------")


def plot_examples(dataloader, N=8, save_to=None):
    x, y, mask, _ = next(iter(dataloader))
    N=np.min([len(x), N])
    fig, ax = plt.subplots(ncols=N, nrows=3, figsize=(N*1.5,3), tight_layout=True)
    if N==1:
        ax = np.expand_dims(ax, 1)
    for i in range(N):
        img = _scale(np.transpose(x[i], (1,2,0)))
        ax[0,i].imshow(img); ax[0,i].axis("off")
        ax[0,i].text(s=f"{y[i].numpy()}", x=10, y=10, verticalalignment="top", fontsize=14, color="red")
        ax[1,i].imshow(mask[i][0], cmap="Greys_r"); ax[1,i].axis("off")
        ax[2,i].hist(x[i].flatten(), bins=20, color="tab:blue", alpha=0.5, density=True)
        ax[2,i].set_yscale("log")
    if save_to is not None:
        fig.savefig(save_to)
        plt.close()


def compute_intermediate_anomaly_maps(anomaly_maps, reduction="mean"):
    """Compute intermediate anomaly maps after spatial reduction but before multi-hstate reduction.
    
    Args:
        anomaly_maps: torch.Tensor (r, B, C, H, W) 
        reduction: str, reduction method
        
    Returns:
        intermediate_maps: numpy array (r, B, C) - maps after spatial reduction
    """
    if isinstance(anomaly_maps, np.ndarray):
        anomaly_maps = torch.from_numpy(anomaly_maps)
    
    if len(anomaly_maps.shape) == 6:
        anomaly_maps = torch.squeeze(anomaly_maps, dim=0)
    
    if reduction == "pixelwise_mean_max":
        # First compute mean across r dimension (dim=0)
        anomaly_maps_pixel = torch.mean(anomaly_maps, dim=0)  # (B, C, H, W)
        # Then compute max across spatial dimensions (H, W)
        intermediate = torch.amax(anomaly_maps_pixel, dim=(-2, -1))  # (B, C)
        # Return as (1, B, C) to maintain r dimension
        return intermediate.unsqueeze(0).cpu().numpy()
    
    elif reduction == "pixelwise_max_mean":
        # First compute max across r dimension (dim=0)
        anomaly_maps_pixel = torch.amax(anomaly_maps, dim=0)  # (B, C, H, W)
        # Then compute mean across spatial dimensions (H, W)
        intermediate = torch.mean(anomaly_maps_pixel, dim=(-2, -1))  # (B, C)
        # Return as (1, B, C) to maintain r dimension
        return intermediate.unsqueeze(0).cpu().numpy()
    
    elif reduction == "downsample_8x8":
        # Apply the downsample 8x8 logic and return intermediate scores
        r, B, C, H, W = anomaly_maps.shape
        
        # Apply 8x8 average pooling to each hidden state
        pooled_maps = []
        for i in range(r):
            # Apply average pooling with kernel size 8, stride 8
            pooled = F.avg_pool2d(anomaly_maps[i], kernel_size=8, stride=8)  # (B, C, H/8, W/8)
            pooled_maps.append(pooled)
        
        # Stack pooled maps
        pooled_stack = torch.stack(pooled_maps, dim=0)  # (r, B, C, H/8, W/8)
        
        # For intermediate visualization, compute mean across spatial dimensions of pooled maps
        # This shows the patch-level scores before max across hidden states
        intermediate = torch.mean(pooled_stack, dim=(-2, -1), keepdim=False)  # (r, B, C)
        return intermediate.cpu().numpy()
    
    elif reduction == "mean":
        intermediate = torch.mean(anomaly_maps, dim=(-3, -2, -1), keepdim=False)  # (r, B)
        return intermediate.cpu().numpy()
    elif reduction == "max":
        intermediate = torch.amax(anomaly_maps, dim=(-3, -2, -1), keepdim=False)  # (r, B)
        return intermediate.cpu().numpy()
    else:
        raise ValueError("reduction must be 'mean', 'max', 'pixelwise_mean_max', 'pixelwise_max_mean', or 'downsample_8x8'")


def plot_reconstruction_examples(inputs, recons, labels, losses, anomaly_maps, N=4, save_to=None, 
                                hstate_info=None, reduction="mean", multi_hstate_reduction="max"):
    """Plot reconstruction examples with enhanced anomaly map visualization.
    
    Args:
        inputs, recons, labels, losses: standard reconstruction data
        anomaly_maps: anomaly maps array
        N: number of samples to plot
        save_to: save path
        hstate_info: dict with 'perceptual_hstates' and 'criterion_names' keys
        reduction: reduction method used
        multi_hstate_reduction: multi-hstate reduction method
    """
    assert len(inputs)==len(recons)==len(labels)==len(losses)

    if len(anomaly_maps.shape) == 6:  # 1, r, b, n, h, w -> r, b, n, h, w
        anomaly_maps = np.squeeze(anomaly_maps, axis=0)

    idxs_good = np.where(labels==0)[0]
    idxs_anom = np.where(labels==1)[0]
    N = np.min([len(idxs_good), len(idxs_anom), N])
    idxs_good = idxs_good[:N]
    idxs_anom = idxs_anom[:N]

    s = anomaly_maps.shape # (r, b, n, h, w)

    # for readability
    inputs_good = inputs[idxs_good]
    recons_good = recons[idxs_good]
    labels_good = labels[idxs_good]
    losses_good = losses[idxs_good]
    anomaly_maps_good = anomaly_maps[:, idxs_good]

    inputs_anom = inputs[idxs_anom]
    recons_anom = recons[idxs_anom]
    labels_anom = labels[idxs_anom]
    losses_anom = losses[idxs_anom]
    anomaly_maps_anom = anomaly_maps[:, idxs_anom]

    # Skip intermediate calculations for downsample_8x8
    if reduction == "downsample_8x8":
        # Simplified visualization for downsample_8x8: just show original + reconstruction
        total_rows = 2 * 2  # orig + recon per class
        
        fig, ax = plt.subplots(ncols=N, nrows=total_rows, figsize=(N*1.5, total_rows*1.5), tight_layout=True)
        if N==1:
            ax = np.expand_dims(ax, 1)
        
        for i, (inputs, recons, labels) in enumerate(zip([inputs_good, inputs_anom],
                                                         [recons_good, recons_anom],
                                                         [labels_good, labels_anom])):
            
            class_label = "Normal" if i == 0 else "Anomalous"
            base_row = i * 2  # 2 rows per class
            
            for j in range(N):
                img = _scale(np.transpose(inputs[j], (1,2,0)))
                rec = _scale(np.transpose(recons[j], (1,2,0)))
                
                # Original image
                ax[base_row, j].imshow(img)
                ax[base_row, j].axis("off")
                if i == 0 and j == 0:
                    ax[base_row, j].set_title("Original", fontsize=10)
                
                # Reconstruction
                ax[base_row + 1, j].imshow(rec)
                ax[base_row + 1, j].axis("off")
                if i == 0 and j == 0:
                    ax[base_row + 1, j].set_title("Reconstruction", fontsize=10)
        
        fig.suptitle(f'Reconstruction Examples (Reduction: {reduction})', fontsize=12, y=0.95)
        
        if save_to is not None:
            fig.savefig(save_to, dpi=150, bbox_inches='tight')
            plt.close()
        return

    # Original logic for other reduction methods
    # Compute intermediate anomaly maps (after reduction, before multi-hstate reduction)
    intermediate_good = compute_intermediate_anomaly_maps(anomaly_maps_good, reduction)
    intermediate_anom = compute_intermediate_anomaly_maps(anomaly_maps_anom, reduction)

    vmin = np.quantile(np.concatenate([anomaly_maps_good, anomaly_maps_anom]), 0.001)
    vmax = np.quantile(np.concatenate([anomaly_maps_good, anomaly_maps_anom]), 0.999)

    # Calculate number of rows: 2 (orig + recon) + s[0] (hstates) + 1 (intermediate) per class
    total_rows = 2 * (2 + s[0] + 1)
    
    fig, ax = plt.subplots(ncols=N, nrows=total_rows, figsize=(N*1.5, total_rows*1.5), tight_layout=True)
    if N==1:
        ax = np.expand_dims(ax, 1)
    
    # Create labels for hidden states
    hstate_labels = []
    if hstate_info and 'perceptual_hstates' in hstate_info:
        perceptual_hstates = hstate_info['perceptual_hstates']
        criterion_names = hstate_info.get('criterion_names', ['Unknown'])
        
        # Create labels based on available information
        if len(perceptual_hstates) == s[0]:
            hstate_labels = [f"hstate_{h}" for h in perceptual_hstates]
        else:
            # If we have multiple criteria or complex structure, use generic labels with indices
            hstate_labels = [f"amap_{k}" for k in range(s[0])]
    else:
        hstate_labels = [f"amap_{k}" for k in range(s[0])]
    
    for i, (inputs, recons, losses, labels, anomaly_maps, intermediate) in enumerate(zip([inputs_good, inputs_anom],
                                                                   [recons_good, recons_anom],
                                                                   [losses_good, losses_anom],
                                                                   [labels_good, labels_anom],
                                                                   [anomaly_maps_good, anomaly_maps_anom],
                                                                   [intermediate_good, intermediate_anom])):
        
        class_label = "Normal" if i == 0 else "Anomalous"
        rows_per_class = 2 + s[0] + 1  # orig + recon + hstates + intermediate
        base_row = i * rows_per_class
        
        for j in range(N):
            img = _scale(np.transpose(inputs[j], (1,2,0)))
            rec = _scale(np.transpose(recons[j], (1,2,0)))
            amap = anomaly_maps[:, j]
            
            # Original image
            ax[base_row, j].imshow(img)
            ax[base_row, j].axis("off")
            
            # Reconstruction
            ax[base_row + 1, j].imshow(rec)
            ax[base_row + 1, j].axis("off")
            
            # Individual hidden state anomaly maps
            for k in range(s[0]):
                ax[base_row + 2 + k, j].imshow(amap[k, 0], vmin=vmin, vmax=vmax, cmap="jet")
                ax[base_row + 2 + k, j].axis("off")
            
            # Intermediate anomaly map (after reduction, before multi-hstate reduction)
            intermediate_row = base_row + 2 + s[0]
            
            if reduction in ["pixelwise_mean_max", "pixelwise_max_mean"]:
                # For pixelwise methods, intermediate is (1, B, C) - just show the score as text
                intermediate_score = intermediate[0, j, 0] if intermediate.shape[2] > 0 else 0
                ax[intermediate_row, j].text(0.5, 0.5, f"After {reduction}:\n{intermediate_score:.4f}", 
                                           horizontalalignment='center', verticalalignment='center',
                                           transform=ax[intermediate_row, j].transAxes, fontsize=12,
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                ax[intermediate_row, j].set_xlim(0, 1)
                ax[intermediate_row, j].set_ylim(0, 1)
            else:
                # For mean/max methods, show heatmap of scores per hstate
                hstate_scores = intermediate[:, j]  # (r,)
                
                # Create a horizontal bar visualization
                y_positions = np.arange(len(hstate_scores))
                colors = plt.cm.viridis(hstate_scores / (hstate_scores.max() + 1e-8))
                
                ax[intermediate_row, j].barh(y_positions, hstate_scores, color=colors)
                ax[intermediate_row, j].set_yticks(y_positions)
                ax[intermediate_row, j].set_yticklabels([f"H{k}" for k in range(len(hstate_scores))], fontsize=8)
                ax[intermediate_row, j].set_xlabel(f"After {reduction}", fontsize=8)
                
                # Add score values as text
                for k, score in enumerate(hstate_scores):
                    ax[intermediate_row, j].text(score/2, k, f"{score:.3f}", 
                                               ha='center', va='center', fontsize=8, color='white', weight='bold')
            
            ax[intermediate_row, j].set_title(f"Before multi-{multi_hstate_reduction}", fontsize=8, pad=5)
            
            # Add final anomaly score in the title of the first column
            if j == 0:
                # Calculate final score using the same logic as score_anomaly_maps
                if multi_hstate_reduction == "mean":
                    final_score = np.mean(intermediate[:, j])
                else:  # max
                    final_score = np.max(intermediate[:, j])
                
                fig.suptitle(f"{class_label} Samples - Final Scores (after {multi_hstate_reduction}): "
                           f"Sample 0: {final_score:.4f}", fontsize=12, y=0.98)
            
    if save_to is not None:
        fig.savefig(save_to, dpi=150, bbox_inches='tight')
        plt.close()


def plot_reconstruction_examples_fullreso(inputs, recons, labels, anomaly_maps, N=8, save_to=None,
                                         hstate_info=None):
    """Enhanced full resolution plotting with hidden state information."""
    assert len(inputs)==len(recons)==len(labels)

    if len(anomaly_maps.shape) == 6:  # 1, r, b, n, h, w -> r, b, n, h, w
        anomaly_maps = np.squeeze(anomaly_maps, axis=0)

    idxs_good = np.where(labels==0)[0]
    idxs_anom = np.where(labels==1)[0]
    N_good = np.min([len(idxs_good), N])
    N_anom = np.min([len(idxs_anom), N])
    idxs_good = idxs_good[:N_good]
    idxs_anom = idxs_anom[:N_anom]

    s = anomaly_maps.shape # (r, b, n, h, w)

    # for readability
    inputs_good = inputs[idxs_good]
    recons_good = recons[idxs_good]
    labels_good = labels[idxs_good]
    anomaly_maps_good = anomaly_maps[:, idxs_good]

    inputs_anom = inputs[idxs_anom]
    recons_anom = recons[idxs_anom]
    labels_anom = labels[idxs_anom]
    anomaly_maps_anom = anomaly_maps[:, idxs_anom]

    vmin = np.quantile(np.concatenate([anomaly_maps_good, anomaly_maps_anom]), 0.)
    vmax = np.quantile(np.concatenate([anomaly_maps_good, anomaly_maps_anom]), 1.)

    # Create labels for hidden states
    hstate_labels = []
    if hstate_info and 'perceptual_hstates' in hstate_info:
        perceptual_hstates = hstate_info['perceptual_hstates']
        if len(perceptual_hstates) == s[0]:
            hstate_labels = [f"hstate_{h}" for h in perceptual_hstates]
        else:
            hstate_labels = [f"amap_{k}" for k in range(s[0])]
    else:
        hstate_labels = [f"amap_{k}" for k in range(s[0])]

    for k, (inputs, recons, labels, anomaly_maps, N, dir) in enumerate(zip([inputs_good, inputs_anom],
                                                                   [recons_good, recons_anom],
                                                                   [labels_good, labels_anom],
                                                                   [anomaly_maps_good, anomaly_maps_anom],
                                                                   [N_good, N_anom],
                                                                   ["good", "anomalous"])):
        for i in range(N):
            img = _scale(np.transpose(inputs[i], (1,2,0)))
            rec = _scale(np.transpose(recons[i], (1,2,0)))
            amap = anomaly_maps[:, i]
            img = Image.fromarray((img*255).astype(np.uint8))
            rec = Image.fromarray((rec*255).astype(np.uint8))
            img.save(os.path.join(save_to, dir+f"/{i}_img.png"))
            rec.save(os.path.join(save_to, dir+f"/{i}_rec.png"))
            for j in range(s[0]):
                amap[j, 0] = (amap[j, 0] - vmin) / (vmax - vmin)
                amap_img = np.clip(amap[j, 0]*255, 0., 255.).astype(np.uint8)
                amap_img = Image.fromarray(amap_img)
                # Use descriptive filename with hidden state info
                hstate_label = hstate_labels[j] if j < len(hstate_labels) else f"amap_{j}"
                amap_img.save(os.path.join(save_to, dir+f"/{i}_{hstate_label}.png"))


def plot_reconstruction_examples_twin(inputs, recons1, recons2, labels,
                                      losses, N=8, save_to=None, anomaly_maps=None):
    assert len(inputs)==len(recons1)==len(recons2)==len(labels)==len(losses)
    N = np.min([len(inputs), N])

    # get anomaly maps if not provided
    if anomaly_maps is None:
        raise NotImplementedError("anomaly_maps must be provided")
    else:       
        s = anomaly_maps.shape # (r, b, n, h, w)
        minmax = np.quantile(anomaly_maps[:, :N], 0.001), np.quantile(anomaly_maps[:, :N], 0.999)

    fig, ax = plt.subplots(ncols=N, nrows=3+s[0], figsize=(N*1.5, 1.5*(3+s[0])), tight_layout=True)
    if N==1:
        ax = np.expand_dims(ax, 1)
    for i in range(N):
        img = _scale(np.transpose(inputs[i], (1,2,0)))
        rec1 = _scale(np.transpose(recons1[i], (1,2,0)))
        rec2 = _scale(np.transpose(recons2[i], (1,2,0)))
        amap = anomaly_maps[:, i]
        ax[0,i].imshow(img); ax[0,i].axis("off")
        ax[0,i].text(s=f"{labels[i]}", x=0, y=0, verticalalignment="top", fontsize=10, color="red")
        ax[1,i].imshow(rec1); ax[1,i].axis("off")
        ax[2,i].imshow(rec2); ax[2,i].axis("off")
        ax[2,i].text(s=f"{losses[i]:.6f}", x=0, y=0, verticalalignment="top", fontsize=10, color="red")
        for j in range(s[0]):
            ax[3+j,i].imshow(amap[j, 0], vmin=minmax[0], vmax=minmax[1], cmap="jet")
            ax[3+j,i].axis("off")
            ax[3+j,i].text(s=f"amap_{j}", x=0, y=0, verticalalignment="top", fontsize=10, color="red")
            
    if save_to is not None:
        fig.savefig(save_to)
        plt.close()


def plot_anomaly_histogram(losses, labels, save_to=None):
    assert len(losses)==len(labels)
    lims = (np.quantile(losses, 0.001), np.quantile(losses, 0.99))
    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
    ax.hist(losses[labels==0], bins=20, alpha=0.5, label="normal",
            color="tab:blue", range=lims, density=True)
    ax.hist(losses[labels==1], bins=20, alpha=0.5, label="anomaly",
            color="tab:orange", range=lims, density=True)
    ax.legend()
    if save_to is not None:
        fig.savefig(save_to)
        plt.close()


def plot_attn_maps(attnmaps, save_to=None, N=4):
    
    n_heads = attnmaps.shape[1]
    aspect_ratio = attnmaps.shape[-1] / attnmaps.shape[-2]

    fig, ax = plt.subplots(N+1, n_heads, figsize=(n_heads*1.5*aspect_ratio, (N+1)*1.5), tight_layout=True)
    for i in range(N):
        for j in range(0, n_heads):
            ax[i, j].imshow(np.sqrt(attnmaps[i, j-1]), interpolation="nearest", aspect="equal")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(f"img{i} - head{j}")
            ax[i, j].set_xlabel("encoder tokens"); ax[i, j].set_ylabel("queries")
    avg_attn = np.mean(attnmaps, axis=0)
    for j in range(n_heads):
        ax[N, j].imshow(np.sqrt(avg_attn[j]), interpolation="nearest", aspect="equal")
        ax[N, j].set_xticks([]); ax[N, j].set_yticks([])
        ax[N, j].set_title(f"avg - head{j}")
        ax[N, j].set_xlabel("encoder tokens"); ax[N, j].set_ylabel("queries")

    if save_to is not None:
        plt.savefig(save_to)


def plot_loss_curves(path_to_csv, save_to=None):
    # Load data
    cols = np.loadtxt(os.path.join(path_to_csv), delimiter=",", max_rows=1, dtype=str)
    data = np.loadtxt(os.path.join(path_to_csv), delimiter=",", skiprows=1, dtype=float)
    datadict = {c: data[:, i] for i, c in enumerate(cols)}

    # Check if we have student/teacher metrics or just a single AUROC
    has_separate_metrics = "student_auroc" in datadict and "teacher_auroc" in datadict
    auroc_key = "student_auroc" if has_separate_metrics else "AUROC"
    
    if auroc_key in datadict:
        max_auroc_idx = np.nanargmax(datadict[auroc_key])
    else:
        # If no AUROC column exists, just use the first epoch as max
        max_auroc_idx = 0

    # Plot res losses, AUC
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
    ax_t = ax.twinx() # first twin axis (AUROC)
    ax_t2 = ax.twinx() # second twin axis (LR)
    ax_t2.spines["right"].set_position(("axes", 1.16))
    
    # Determine which keys to use for losses (exclude epoch, metrics, and LR)
    metric_keys = ["epoch", "student_auroc", "teacher_auroc", "student_val_rec_loss_0", 
                  "student_val_rec_loss_1", "teacher_val_rec_loss_0", "teacher_val_rec_loss_1", 
                  "student_LR", "model_LR", "LR", "AUROC", "val_rec_loss_0", "val_rec_loss_1"]
    
    rec_losses_keys = [k for k in datadict.keys() if k not in metric_keys]
    rec_losses_colors = ["tab:grey", "tab:blue", "tab:cyan", "tab:purple", "tab:pink"][:len(rec_losses_keys)]
    
    # Plot training losses
    reclosses = [ax.plot(datadict["epoch"], datadict[k], label=k+" (train)", c=c)[0]
                 for k, c in zip(rec_losses_keys, rec_losses_colors)]
    
    # Validation losses - handle both formats (with/without separate student/teacher metrics)
    val_lines = []
    
    if has_separate_metrics:
        # Newer format with student/teacher metrics
        student_val0_mask = datadict["student_val_rec_loss_0"] == datadict["student_val_rec_loss_0"]
        teacher_val0_mask = datadict["teacher_val_rec_loss_0"] == datadict["teacher_val_rec_loss_0"]
        
        valloss_student0, = ax.plot(datadict["epoch"][student_val0_mask],
                            datadict["student_val_rec_loss_0"][student_val0_mask],
                            label="Student rec loss (val/cl0)", c="tab:red", linestyle=":")
        
        valloss_student1, = ax.plot(datadict["epoch"][student_val0_mask],
                            datadict["student_val_rec_loss_1"][student_val0_mask],
                            label="Student rec loss (val/cl1)", c="tab:red", linestyle="-.")
        
        valloss_teacher0, = ax.plot(datadict["epoch"][teacher_val0_mask],
                            datadict["teacher_val_rec_loss_0"][teacher_val0_mask],
                            label="Teacher rec loss (val/cl0)", c="tab:green", linestyle=":")
        
        valloss_teacher1, = ax.plot(datadict["epoch"][teacher_val0_mask],
                            datadict["teacher_val_rec_loss_1"][teacher_val0_mask],
                            label="Teacher rec loss (val/cl1)", c="tab:green", linestyle="-.")
        
        val_lines = [valloss_student0, valloss_student1, valloss_teacher0, valloss_teacher1]
    else:
        # Original format with just one set of metrics
        if "val_rec_loss_0" in datadict:
            val0_mask = datadict["val_rec_loss_0"] == datadict["val_rec_loss_0"]
            
            valloss0, = ax.plot(datadict["epoch"][val0_mask],
                              datadict["val_rec_loss_0"][val0_mask],
                              label="rec loss (val/cl0)", c="tab:red", linestyle=":")
            
            valloss1, = ax.plot(datadict["epoch"][val0_mask],
                              datadict["val_rec_loss_1"][val0_mask],
                              label="rec loss (val/cl1)", c="tab:red")
            
            val_lines = [valloss0, valloss1]

    # Plot AUROC
    auroc_lines = []
    if has_separate_metrics:
        # Plot both student and teacher AUROCs
        student_auroc_mask = datadict["student_auroc"] == datadict["student_auroc"]
        student_auroc, = ax_t.plot(datadict["epoch"][student_auroc_mask],
                        datadict["student_auroc"][student_auroc_mask],
                        label="Student AUROC", c="tab:orange", linestyle="--")
        
        teacher_auroc_mask = datadict["teacher_auroc"] == datadict["teacher_auroc"]
        teacher_auroc, = ax_t.plot(datadict["epoch"][teacher_auroc_mask],
                        datadict["teacher_auroc"][teacher_auroc_mask],
                        label="Teacher AUROC", c="tab:purple", linestyle="--")
        
        # Mark max AUROC point for student
        student_max_auroc_idx = np.nanargmax(datadict["student_auroc"])
        ax_t.plot(student_max_auroc_idx, datadict["student_auroc"][student_max_auroc_idx], "x", c="tab:orange")
        ax_t.text(student_max_auroc_idx, datadict["student_auroc"][student_max_auroc_idx],
                f"{datadict['student_auroc'][student_max_auroc_idx]:.2f}", va="bottom", ha="right")
        
        # Mark max AUROC point for teacher
        teacher_max_auroc_idx = np.nanargmax(datadict["teacher_auroc"])
        ax_t.plot(teacher_max_auroc_idx, datadict["teacher_auroc"][teacher_max_auroc_idx], "x", c="tab:purple")
        ax_t.text(teacher_max_auroc_idx, datadict["teacher_auroc"][teacher_max_auroc_idx],
                f"{datadict['teacher_auroc'][teacher_max_auroc_idx]:.2f}", va="bottom", ha="right")
        
        auroc_lines = [student_auroc, teacher_auroc]
    else:
        # Original format with just one AUROC
        if "AUROC" in datadict:
            auroc_mask = datadict["AUROC"] == datadict["AUROC"]
            auroc, = ax_t.plot(datadict["epoch"][auroc_mask],
                            datadict["AUROC"][auroc_mask],
                            label="AD auc", c="tab:orange", linestyle="--")
            
            ax_t.plot(max_auroc_idx, datadict["AUROC"][max_auroc_idx], "x", c="black")
            ax_t.text(max_auroc_idx, datadict["AUROC"][max_auroc_idx],
                    f"{datadict['AUROC'][max_auroc_idx]:.2f}", va="bottom", ha="right")
            
            auroc_lines = [auroc]

    # Plot learning rates
    lr_lines = []
    if "student_LR" in datadict and "model_LR" in datadict:
        # Plot separate learning rates for student and rest of model
        student_LR, = ax_t2.plot(datadict["epoch"], datadict["student_LR"],
                            label="Student LR", c="tab:green", linestyle="-", alpha=0.7)
        model_LR, = ax_t2.plot(datadict["epoch"], datadict["model_LR"],
                        label="Model LR", c="tab:purple", linestyle="-", alpha=0.7)
        lr_lines = [student_LR, model_LR]
    elif "LR" in datadict:
        # Original format with just one learning rate
        LR, = ax_t2.plot(datadict["epoch"], datadict["LR"],
                        label="LR", c="tab:green", linestyle="-", alpha=0.7)
        lr_lines = [LR]
    
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax_t.set_ylabel("AUROC")
    ax_t2.set_ylabel("LR")
    ax_t.set_ylim(0, 1)

    # Combine all lines for legend
    all_lines = reclosses + val_lines + auroc_lines + lr_lines
    ax.legend(handles=all_lines, loc="center right", fontsize=8)

    # Save to file
    if save_to is not None:
        fig.savefig(save_to)
        plt.close()


# =====================================================
# PIXEL-LEVEL VISUALIZATION FUNCTIONS
# =====================================================

def plot_pixel_level_results(anomaly_maps, gt_masks, labels, reduction="mean", 
                           multi_hstate_reduction="max", N=8, save_to=None):
    """Plot pixel-level anomaly detection results with ground truth masks.
    
    Args:
        anomaly_maps: numpy array (r, B, C, H, W) - anomaly maps from model
        gt_masks: numpy array (B, H, W) or (B, 1, H, W) - ground truth masks
        labels: numpy array (B,) - image-level labels
        reduction: str, reduction method used
        multi_hstate_reduction: str, multi-hstate reduction method
        N: int, number of samples to plot
        save_to: str, path to save the plot
    """
    if isinstance(anomaly_maps, torch.Tensor):
        anomaly_maps = anomaly_maps.numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
        
    # Ensure gt_masks has the right shape (B, H, W)
    if len(gt_masks.shape) == 4 and gt_masks.shape[1] == 1:
        gt_masks = gt_masks.squeeze(1)
    
    # Filter to only anomalous samples that have masks
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) == 0:
        print("No anomalous samples found for pixel-level visualization")
        return
        
    N = min(N, len(anomaly_indices))
    selected_indices = anomaly_indices[:N]
    
    # Compute pixel-level scores for selected samples
    selected_maps = anomaly_maps[:, selected_indices]  # (r, N, C, H, W)
    selected_masks = gt_masks[selected_indices]  # (N, H, W)
    
    # Reduce anomaly maps to pixel-level scores (N, H, W)
    if reduction == "downsample_8x8":
        # Apply the same downsample_8x8 logic for visualization
        r, N_samples, C, H, W = selected_maps.shape
        selected_maps_torch = torch.from_numpy(selected_maps)
        
        # Apply 8x8 average pooling to each hidden state
        pooled_maps = []
        for i in range(r):
            pooled = F.avg_pool2d(selected_maps_torch[i], kernel_size=8, stride=8)  # (N, C, H/8, W/8)
            pooled_maps.append(pooled)
        
        # Stack and take max across hidden states
        pooled_stack = torch.stack(pooled_maps, dim=0)  # (r, N, C, H/8, W/8)
        max_pooled = torch.amax(pooled_stack, dim=0)  # (N, C, H/8, W/8)
        
        # Resize back to original spatial dimensions
        resized = F.interpolate(max_pooled, size=(H, W), mode='bilinear', align_corners=False)  # (N, C, H, W)
        
        # Average across channels to get pixel scores
        pixel_scores = torch.mean(resized, dim=1).numpy()  # (N, H, W)
        
    elif reduction == "mean":
        if multi_hstate_reduction == "mean":
            pixel_scores = np.mean(np.mean(selected_maps, axis=0), axis=1)  # (N, H, W)
        else:  # max
            pixel_scores = np.mean(np.max(selected_maps, axis=0), axis=1)  # (N, H, W)
    elif reduction == "max":
        if multi_hstate_reduction == "mean":
            pixel_scores = np.max(np.mean(selected_maps, axis=0), axis=1)  # (N, H, W)
        else:  # max
            pixel_scores = np.max(np.max(selected_maps, axis=0), axis=1)  # (N, H, W)
    else:
        # Default to mean for unknown reduction methods
        if multi_hstate_reduction == "mean":
            pixel_scores = np.mean(np.mean(selected_maps, axis=0), axis=1)  # (N, H, W)
        else:  # max
            pixel_scores = np.mean(np.max(selected_maps, axis=0), axis=1)  # (N, H, W)
    
    # Create visualization
    fig, axes = plt.subplots(3, N, figsize=(N*3, 9), tight_layout=True)
    if N == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(N):
        # Ground truth mask
        axes[0, i].imshow(selected_masks[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'GT Mask {i}')
        axes[0, i].axis('off')
        
        # Predicted anomaly map
        axes[1, i].imshow(pixel_scores[i], cmap='jet', vmin=pixel_scores.min(), vmax=pixel_scores.max())
        axes[1, i].set_title(f'Pred Map {i}')
        axes[1, i].axis('off')
        
        # Overlay
        overlay = np.zeros((*selected_masks[i].shape, 3))
        # Ground truth in red channel
        overlay[:, :, 0] = selected_masks[i]
        # Prediction in green channel (normalized)
        pred_norm = (pixel_scores[i] - pixel_scores[i].min()) / (pixel_scores[i].max() - pixel_scores[i].min() + 1e-8)
        overlay[:, :, 1] = pred_norm
        
        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f'Overlay {i}\n(Red: GT, Green: Pred)')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Pixel-Level Results (Reduction: {reduction}, Multi-hstate: {multi_hstate_reduction})')
    
    if save_to is not None:
        fig.savefig(save_to, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_detailed_pixel_metrics(anomaly_maps, gt_masks, labels, reduction="mean", 
                                 multi_hstate_reduction="max"):
    """Compute detailed pixel-level metrics including AP and optimal threshold.
    
    Args:
        anomaly_maps: torch.Tensor or numpy array (r, B, C, H, W)
        gt_masks: torch.Tensor or numpy array (B, H, W) or (B, 1, H, W)
        labels: torch.Tensor or numpy array (B,) - image-level labels
        reduction: str, reduction method
        multi_hstate_reduction: str, multi-hstate reduction method
        
    Returns:
        dict with detailed metrics
    """
    from training import compute_pixel_level_metrics
    
    # Convert to tensors if needed
    if isinstance(anomaly_maps, np.ndarray):
        anomaly_maps = torch.from_numpy(anomaly_maps)
    if isinstance(gt_masks, np.ndarray):
        gt_masks = torch.from_numpy(gt_masks)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Filter to only anomalous samples
    anomaly_indices = torch.where(labels == 1)[0]
    if len(anomaly_indices) == 0:
        return {
            "pixel_auroc": np.nan,
            "pixel_ap": np.nan,
            "optimal_threshold": np.nan,
            "precision_at_threshold": np.nan,
            "recall_at_threshold": np.nan,
            "num_anomalous_samples": 0
        }
    
    # Compute pixel-level metrics
    pixel_auroc, pixel_scores_flat, gt_masks_flat = compute_pixel_level_metrics(
        anomaly_maps, 
        gt_masks, 
        reduction=reduction,
        multi_hstate_reduction=multi_hstate_reduction
    )
    
    # Compute additional metrics
    pixel_ap = average_precision_score(gt_masks_flat, pixel_scores_flat)
    
    # Find optimal threshold (maximizing F1 score)
    precision, recall, thresholds = precision_recall_curve(gt_masks_flat, pixel_scores_flat)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    
    return {
        "pixel_auroc": pixel_auroc,
        "pixel_ap": pixel_ap,
        "optimal_threshold": optimal_threshold,
        "precision_at_threshold": precision[optimal_idx],
        "recall_at_threshold": recall[optimal_idx],
        "f1_at_threshold": f1_scores[optimal_idx],
        "num_anomalous_samples": len(anomaly_indices),
        "num_positive_pixels": np.sum(gt_masks_flat),
        "num_total_pixels": len(gt_masks_flat)
    }


def save_pixel_level_predictions(anomaly_maps, gt_masks, labels, reduction="mean", 
                                multi_hstate_reduction="max", save_dir=None, 
                                threshold=None):
    """Save pixel-level predictions as images for qualitative analysis.
    
    Args:
        anomaly_maps: numpy array (r, B, C, H, W)
        gt_masks: numpy array (B, H, W) or (B, 1, H, W)
        labels: numpy array (B,)
        reduction: str, reduction method
        multi_hstate_reduction: str, multi-hstate reduction method
        save_dir: str, directory to save images
        threshold: float, threshold for binary predictions (if None, saves continuous maps)
    """
    if save_dir is None:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "gt_masks"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "pred_maps"), exist_ok=True)
    if threshold is not None:
        os.makedirs(os.path.join(save_dir, "pred_binary"), exist_ok=True)
    
    # Ensure gt_masks has the right shape
    if len(gt_masks.shape) == 4 and gt_masks.shape[1] == 1:
        gt_masks = gt_masks.squeeze(1)
    
    # Filter to anomalous samples
    anomaly_indices = np.where(labels == 1)[0]
    
    for idx in anomaly_indices:
        # Compute pixel-level score for this sample
        sample_maps = anomaly_maps[:, idx:idx+1]  # (r, 1, C, H, W)
        sample_mask = gt_masks[idx:idx+1]  # (1, H, W)
        
        # Reduce to pixel scores based on reduction method
        if reduction == "downsample_8x8":
            # Apply the same downsample_8x8 logic
            r, B, C, H, W = sample_maps.shape
            sample_maps_torch = torch.from_numpy(sample_maps)
            
            # Apply 8x8 average pooling to each hidden state
            pooled_maps = []
            for i in range(r):
                pooled = F.avg_pool2d(sample_maps_torch[i], kernel_size=8, stride=8)  # (1, C, H/8, W/8)
                pooled_maps.append(pooled)
            
            # Stack and take max across hidden states
            pooled_stack = torch.stack(pooled_maps, dim=0)  # (r, 1, C, H/8, W/8)
            max_pooled = torch.amax(pooled_stack, dim=0)  # (1, C, H/8, W/8)
            
            # Resize back to original spatial dimensions
            resized = F.interpolate(max_pooled, size=(H, W), mode='bilinear', align_corners=False)  # (1, C, H, W)
            
            # Average across channels to get pixel scores
            pixel_score = torch.mean(resized, dim=1)[0].numpy()  # (H, W)
            
        elif reduction == "mean":
            if multi_hstate_reduction == "mean":
                pixel_score = np.mean(np.mean(sample_maps, axis=0), axis=1)[0]  # (H, W)
            else:  # max
                pixel_score = np.mean(np.max(sample_maps, axis=0), axis=1)[0]  # (H, W)
        elif reduction == "max":
            if multi_hstate_reduction == "mean":
                pixel_score = np.max(np.mean(sample_maps, axis=0), axis=1)[0]  # (H, W)
            else:  # max
                pixel_score = np.max(np.max(sample_maps, axis=0), axis=1)[0]  # (H, W)
        else:
            # Default to mean for unknown reduction methods
            if multi_hstate_reduction == "mean":
                pixel_score = np.mean(np.mean(sample_maps, axis=0), axis=1)[0]  # (H, W)
            else:  # max
                pixel_score = np.mean(np.max(sample_maps, axis=0), axis=1)[0]  # (H, W)
        
        # Save ground truth mask
        gt_img = Image.fromarray((sample_mask[0] * 255).astype(np.uint8))
        gt_img.save(os.path.join(save_dir, "gt_masks", f"sample_{idx:04d}_gt.png"))
        
        # Save continuous prediction map
        pred_norm = (pixel_score - pixel_score.min()) / (pixel_score.max() - pixel_score.min() + 1e-8)
        pred_img = Image.fromarray((pred_norm * 255).astype(np.uint8))
        pred_img.save(os.path.join(save_dir, "pred_maps", f"sample_{idx:04d}_pred.png"))
        
        # Save binary prediction if threshold provided
        if threshold is not None:
            binary_pred = (pixel_score > threshold).astype(np.uint8)
            binary_img = Image.fromarray(binary_pred * 255)
            binary_img.save(os.path.join(save_dir, "pred_binary", f"sample_{idx:04d}_binary.png"))


def analyze_pixel_performance_by_class(anomaly_maps, gt_masks, labels, 
                                     reduction="mean", multi_hstate_reduction="max"):
    """Analyze pixel-level performance separately for different anomaly classes if applicable.
    
    This is useful for multi-class anomaly detection or when you have different types of anomalies.
    """
    unique_labels = np.unique(labels)
    results = {}
    
    for label in unique_labels:
        if label == 0:  # Skip normal samples
            continue
            
        label_indices = np.where(labels == label)[0]
        if len(label_indices) == 0:
            continue
            
        label_maps = anomaly_maps[:, label_indices]
        label_masks = gt_masks[label_indices]
        label_labels = labels[label_indices]
        
        metrics = compute_detailed_pixel_metrics(
            label_maps, label_masks, label_labels,
            reduction=reduction, multi_hstate_reduction=multi_hstate_reduction
        )
        
        results[f"class_{label}"] = metrics
    
    return results
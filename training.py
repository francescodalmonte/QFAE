import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from losses import criterion_selector, resize_like

def compute_rec_loss(criterion, x, x_hat, pixel_wise=False): 
    """Compute reconstruction loss.
    x / x_hat : input and reconstructed image
    pixel_wise : pixel- or sample-wise loss

    Returns:
    rec_losses : total reconstruction loss
    """
    out = criterion(x, x_hat)
    ckeys = out.keys()
    img_rec_losses = {k : out[k] for k in ckeys}

    rec_losses = {}
    for k in ckeys:
        loss_tensor = img_rec_losses[k]
        
        # Handle different tensor shapes
        if loss_tensor.dim() == 4:  # (B, C, H, W)
            loss_tensor = loss_tensor.unsqueeze(0)  # (1, B, C, H, W)
        elif loss_tensor.dim() == 5:  # (R, B, C, H, W)
            pass  # Already correct shape
        else:
            raise ValueError(f"criterion() output must have 4 or 5 dimensions, got {loss_tensor.dim()}")
        
        if pixel_wise:
            rec_losses[k] = torch.mean(loss_tensor, dim=(2), keepdim=True) # (R, B, 1, H, W)
        else:
            rec_losses[k] = torch.mean(loss_tensor, dim=(2,3,4), keepdim=False) # (R, B)
    
    return rec_losses
    

def train_step(model,
               train_dl,
               optimizer,
               criterion,
               device,
               verbose = True,
               lr_scheduler = None,
               auxilary_dl = None,
               crossattn_masks = None):
    """Train model for one epoch."""

    minibatch_losses = {"rec_tot": []}
    minibatch_sizes = []

    if crossattn_masks is not None:
        crossattn_masks.to(device)
    for i, (x, _, _, _) in enumerate(train_dl):
        x = x.to(device)
        
        # forward pass
        assert model.junction.queries.requires_grad, "junction queries should require grad"

        if crossattn_masks is not None:
            # sample random cross-attention mask
            ridxs = torch.randint(0, len(crossattn_masks), (1,))
            # create unique mask
            mask = crossattn_masks[ridxs].repeat(x.size(0), 1, 1, 1)
        else:
            mask = None

        out = model(x, crossattn_mask=mask)
        x_hat, _, _ = out

        # tot loss
        rec_losses_dict = compute_rec_loss(criterion, x, x_hat, pixel_wise=False)
        if i==0:
            minibatch_losses.update({k: [] for k in rec_losses_dict.keys()})

        # forward pass on auxilary data (regularization term)
        if auxilary_dl is not None:
            x_aux = next(auxilary_dl).to(device)
            out_aux = model(x_aux)
            x_hat_aux, _, _= out_aux

            rec_losses_dict["auxilary"] = torch.mean(torch.abs(x_hat_aux), dim=(1,2,3), keepdim=False)

        # single value for backprop (avg batch dim, sum h dim)
        rec_loss = torch.sum( torch.cat( [v.flatten() for v in rec_losses_dict.values()] ).mean() )

        # backward pass
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # store losses
        minibatch_losses["rec_tot"].append(rec_loss.item())
        for k in rec_losses_dict.keys():
            minibatch_losses[k].append(torch.mean(rec_losses_dict[k]).item())
        minibatch_sizes.append(x.size(0))

        # print batch info
        curlosses = [l[-1] for l in minibatch_losses.values()]
        print(f"batch {i+1}/{len(train_dl)} - current losses: {curlosses}", end="\r")

    # epoch losses
    epoch_losses = {k: np.sum(np.multiply(minibatch_losses[k], minibatch_sizes)) / np.sum(minibatch_sizes)
                    for k in minibatch_losses.keys()}

    if verbose:
        for k, v in epoch_losses.items():
            print(f"{k}: {v:.6f}", end=" | ")

    return epoch_losses


def score_anomaly_maps(anomaly_maps, reduction="mean", multi_hstate_reduction="max"):
    """Compute anomaly scores from anomaly maps FOR IMAGE-LEVEL SCORING.
    anomaly maps : torch.Tensor (r, B, C, H, W)
    reduction : str, "mean"/"max"/"pixelwise_mean_max"/"pixelwise_max_mean"/"downsample_8x8"
    multi_hstate_reduction : str, "mean"/"max"
    
    Returns: torch.Tensor (B,) - scalar scores for each image
    """
    if len(anomaly_maps.shape) == 6:
        anomaly_maps = torch.squeeze(anomaly_maps, dim=0)  # (r, B, C, H, W)
    
    if reduction == "pixelwise_mean_max":
        # First compute mean across r dimension (dim=0)
        anomaly_maps_pixel = torch.mean(anomaly_maps, dim=0)  # (B, C, H, W)
        # Then compute max across spatial dimensions (H, W)
        anomaly_maps = torch.amax(anomaly_maps_pixel, dim=(-2, -1))  # (B, C)
        # Finally compute mean across channels
        anomaly_maps = torch.mean(anomaly_maps, dim=-1).cpu()  # (B,)
        # Skip multi_hstate_reduction since we've already reduced the r dimension
        return anomaly_maps
    
    elif reduction == "pixelwise_max_mean":
        # First compute max across r dimension (dim=0)
        anomaly_maps_pixel = torch.amax(anomaly_maps, dim=0)  # (B, C, H, W)
        # Then compute mean across spatial dimensions (H, W)
        anomaly_maps = torch.mean(anomaly_maps_pixel, dim=(-2, -1))  # (B, C)
        # Finally compute mean across channels
        anomaly_maps = torch.mean(anomaly_maps, dim=-1).cpu()  # (B,)
        # Skip multi_hstate_reduction since we've already reduced the r dimension
        return anomaly_maps
    
    elif reduction == "downsample_8x8":
        # Downsample by 8x8 avg pooling, take max between hstates, resize back
        r, B, C, H, W = anomaly_maps.shape
        
        # Apply 8x8 average pooling to each hidden state
        pooled_maps = []
        for i in range(r):
            # Apply average pooling with kernel size 8, stride 8
            pooled = F.avg_pool2d(anomaly_maps[i], kernel_size=8, stride=8)  # (B, C, H/8, W/8)
            pooled_maps.append(pooled)
        
        # Stack and take max across hidden states
        pooled_stack = torch.stack(pooled_maps, dim=0)  # (r, B, C, H/8, W/8)
        max_pooled = torch.amax(pooled_stack, dim=0)  # (B, C, H/8, W/8)
        
        # Resize back to original spatial dimensions
        resized = F.interpolate(max_pooled, size=(H, W), mode='bilinear', align_corners=False)  # (B, C, H, W)
        
        # Compute final score (mean across spatial and channel dimensions)
        anomaly_maps = torch.mean(resized, dim=(-3, -2, -1), keepdim=False).cpu()  # (B,)
        
        # Skip multi_hstate_reduction since we've already handled it
        return anomaly_maps
    
    elif reduction == "mean":
        anomaly_maps = torch.mean(anomaly_maps, dim=(-3, -2, -1), keepdim=False).cpu()
    elif reduction == "max":
        anomaly_maps = torch.amax(anomaly_maps, dim=(-3, -2, -1), keepdim=False).cpu()
    else:
        raise ValueError("reduction must be 'mean', 'max', 'pixelwise_mean_max', 'pixelwise_max_mean', or 'downsample_8x8'")
    
    if multi_hstate_reduction == "mean":
        anomaly_maps = torch.mean(anomaly_maps, dim=0, keepdim=False).cpu()
    elif multi_hstate_reduction == "max":
        anomaly_maps = torch.amax(anomaly_maps, dim=0, keepdim=False).cpu()
    else:
        raise ValueError("multi_hstate_reduction must be 'mean' or 'max'")
    
    return anomaly_maps


def compute_pixel_level_metrics(anomaly_maps, gt_masks, reduction="mean", multi_hstate_reduction="max"):
    """Compute pixel-level segmentation metrics.
    
    Args:
        anomaly_maps: torch.Tensor (r, B, C, H, W) - anomaly maps from model
        gt_masks: torch.Tensor (B, 1, H, W) - ground truth binary masks
        reduction: str, reduction method for spatial dimensions
        multi_hstate_reduction: str, reduction method for multiple hidden states
    
    Returns:
        pixel_auroc: float, pixel-level AUROC score
        pixel_scores_flat: np.array, flattened pixel scores for further analysis
        gt_masks_flat: np.array, flattened ground truth masks
    """
    if isinstance(anomaly_maps, np.ndarray):
        anomaly_maps = torch.from_numpy(anomaly_maps)
    if isinstance(gt_masks, np.ndarray):
        gt_masks = torch.from_numpy(gt_masks)
    
    # Ensure anomaly_maps has the right shape
    if len(anomaly_maps.shape) == 6:
        anomaly_maps = torch.squeeze(anomaly_maps, dim=0)  # (r, B, C, H, W)
    
    # Ensure gt_masks has the right shape (B, H, W)
    if len(gt_masks.shape) == 4 and gt_masks.shape[1] == 1:
        gt_masks = gt_masks.squeeze(1)  # (B, H, W)
    
    # Handle different reduction methods for pixel-level scores
    if reduction == "pixelwise_mean_max":
        # First compute mean across r dimension (dim=0), preserving spatial dimensions
        anomaly_maps_pixel = torch.mean(anomaly_maps, dim=0)  # (B, C, H, W)
        # Average across channels to get pixel scores (preserve spatial for pixel-level)
        pixel_scores = torch.mean(anomaly_maps_pixel, dim=1)  # (B, H, W)
        
    elif reduction == "pixelwise_max_mean":
        # First compute max across r dimension (dim=0), preserving spatial dimensions
        anomaly_maps_pixel = torch.amax(anomaly_maps, dim=0)  # (B, C, H, W)
        # Average across channels to get pixel scores (preserve spatial for pixel-level)
        pixel_scores = torch.mean(anomaly_maps_pixel, dim=1)  # (B, H, W)
        
    elif reduction == "downsample_8x8":
        # Apply the same downsample_8x8 logic but return spatial maps instead of scalars
        r, B, C, H, W = anomaly_maps.shape
        
        # Apply 8x8 average pooling to each hidden state
        pooled_maps = []
        for i in range(r):
            # Apply average pooling with kernel size 8, stride 8
            pooled = F.avg_pool2d(anomaly_maps[i], kernel_size=8, stride=8)  # (B, C, H/8, W/8)
            pooled_maps.append(pooled)
        
        # Stack and take max across hidden states
        pooled_stack = torch.stack(pooled_maps, dim=0)  # (r, B, C, H/8, W/8)
        max_pooled = torch.amax(pooled_stack, dim=0)  # (B, C, H/8, W/8)
        
        # Resize back to original spatial dimensions
        resized = F.interpolate(max_pooled, size=(H, W), mode='bilinear', align_corners=False)  # (B, C, H, W)
        
        # Average across channels to get pixel scores
        pixel_scores = torch.mean(resized, dim=1)  # (B, H, W)
        
    elif reduction == "mean":
        # Average across channels and hidden states
        if multi_hstate_reduction == "mean":
            pixel_scores = torch.mean(torch.mean(anomaly_maps, dim=0), dim=1)  # (B, H, W)
        else:  # max
            pixel_scores = torch.mean(torch.amax(anomaly_maps, dim=0), dim=1)  # (B, H, W)
    elif reduction == "max":
        # Max across channels and hidden states  
        if multi_hstate_reduction == "mean":
            pixel_scores = torch.amax(torch.mean(anomaly_maps, dim=0), dim=1)  # (B, H, W)
        else:  # max
            pixel_scores = torch.amax(torch.amax(anomaly_maps, dim=0), dim=1)  # (B, H, W)
    else:
        # For unknown reduction methods, fall back to mean
        if multi_hstate_reduction == "mean":
            pixel_scores = torch.mean(torch.mean(anomaly_maps, dim=0), dim=1)  # (B, H, W)
        else:  # max
            pixel_scores = torch.mean(torch.amax(anomaly_maps, dim=0), dim=1)  # (B, H, W)
    
    # Resize anomaly maps to match ground truth mask size if needed
    if pixel_scores.shape[-2:] != gt_masks.shape[-2:]:
        pixel_scores = F.interpolate(
            pixel_scores.unsqueeze(1), 
            size=gt_masks.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
    
    # Convert to numpy and flatten
    pixel_scores_flat = pixel_scores.cpu().numpy().flatten()
    gt_masks_flat = gt_masks.cpu().numpy().flatten()
    
    # Binarize ground truth masks
    gt_masks_flat = (gt_masks_flat > 0.01).astype(np.int_)
    
    # Compute pixel-level AUROC
    pixel_auroc = roc_auc_score(gt_masks_flat, pixel_scores_flat)
    pixel_auprc = average_precision_score(gt_masks_flat, pixel_scores_flat)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("pixel_auroc: ", pixel_auroc)
    print("pixel_auprc: ", pixel_auprc)
    
    return pixel_auroc, pixel_scores_flat, gt_masks_flat


def eval_step(model,
              val_dl,
              criterion,
              device,
              verbose=True,
              store_data=False,
              image_reduction="mean",
              image_multi_hstate_reduction="max",
              pixel_reduction="mean",
              pixel_multi_hstate_reduction="max",
              compute_pixel_metrics=False):
    """Evaluate model on validation set with separate reduction parameters for image and pixel metrics."""
    
    xs, labels, masks = [], [], []
    x_hats, features = [], []
    rec_amaps, ascores = [], []
    attention_maps = []

    with torch.no_grad():
        for i, (x, ls, ms, _) in enumerate(val_dl):
            print(f"eval batch {i+1}/{len(val_dl)}", end="\r", flush=True)
            x = x.to(device)
            if ms is not None:
                ms = ms.to(device)

            # forward pass
            out = model(x, crossattn_mask=None)
            x_hat, f, attnmaps = out

            # tot loss
            rec_losses_dict = compute_rec_loss(criterion, x, x_hat, pixel_wise=True)
            rec_losses = list(rec_losses_dict.values())

            # Handle different sized anomaly maps by resizing to largest
            if len(rec_losses) > 1:
                sizes = [r.size(-1) for r in rec_losses]
                max_size_idx = sizes.index(max(sizes))
                target_shape = rec_losses[max_size_idx]
                
                resized_losses = []
                for j, r in enumerate(rec_losses):
                    if r.shape != target_shape.shape:
                        resized = resize_like(r, target_shape)
                        resized_losses.append(resized)
                    else:
                        resized_losses.append(r)
                rec_losses = resized_losses

            rec_losses = torch.cat(rec_losses, dim=0)  # Concatenate along first dimension

            # store losses
            rec_amaps.append(rec_losses.cpu())
            
            # Use image-specific reduction parameters for image-level scoring
            batch_scores = score_anomaly_maps(rec_losses, 
                                            reduction=image_reduction, 
                                            multi_hstate_reduction=image_multi_hstate_reduction)
            ascores.append(batch_scores)
            labels.append(ls)
            
            if store_data:
                xs.append(x.cpu())
                features.append(f.cpu())
                x_hats.append(x_hat.cpu())
                attention_maps.append(attnmaps.cpu())
                
            # Store masks for pixel-level evaluation
            if compute_pixel_metrics and ms is not None:
                masks.append(ms.cpu())
        
        rec_amaps = torch.cat(rec_amaps, dim=1).numpy()  # Concatenate along batch dimension
        ascores = torch.cat(ascores).numpy()
        labels = torch.cat(labels).numpy()
        
        if store_data:
            xs = torch.cat(xs, dim=0).numpy()
            features = torch.cat(features, dim=0).numpy()
            x_hats = torch.cat(x_hats, dim=0).numpy()
            attention_maps = torch.cat(attention_maps, dim=0).numpy()
        
        if compute_pixel_metrics and masks:
            masks = torch.cat(masks, dim=0)

    ascores_classavg = []
    for c in np.unique(labels):
        a_classavg = np.mean([a for a, l in zip(ascores, labels) if l==c])
        if verbose:
            print(f"val rec loss {c}: {a_classavg:.5f}", end = " - ")
        ascores_classavg.append((c, a_classavg))
    
    # compute AUROC for anomaly detection (image-level)
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, ascores)
        if verbose:
            print(f"Image AUROC: {auroc:.5f}", end=" | ")
    else:
        auroc = np.nan
    
    # compute pixel-level metrics if requested using pixel-specific reduction parameters
    pixel_auroc = np.nan
    if compute_pixel_metrics and masks is not None and len(masks) > 0:
        pixel_auroc, _, _ = compute_pixel_level_metrics(
            rec_amaps, 
            masks, 
            reduction=pixel_reduction,
            multi_hstate_reduction=pixel_multi_hstate_reduction
        )
        if verbose and not np.isnan(pixel_auroc):
            print(f"Pixel AUROC: {pixel_auroc:.5f}", end=" | ")
        
    out_dict = { f"val_rec_loss_{c}": v for c, v in ascores_classavg } | {
        "auroc": auroc,
        "pixel_auroc": pixel_auroc
    }
    out_dict["labels"] = labels
    if store_data:
        out_dict["ascores"] = ascores
        out_dict["x"] = xs
        out_dict["features"] = features
        out_dict["x_hat"] = x_hats
        out_dict["anomaly_maps_tot"] = rec_amaps
        out_dict["attention_maps"] = attention_maps
        if compute_pixel_metrics and masks is not None:
            out_dict["masks"] = masks.numpy()
    
    return out_dict


def train_step_twin(model,
                    train_dl,
                    optimizer,
                    criterion,
                    device,
                    verbose = True,
                    lr_scheduler = None,
                    auxilary_dl = None):
    """Train model for one epoch."""

    minibatch_losses = {"rec_tot": [], "rec_img1": [], "rec_img2": [], "rec_twin": []}
    minibatch_sizes = []
    for i, (x, _, _, _) in enumerate(train_dl):
        x = x.to(device)
        
        # forward pass
        assert model.junction.queries.requires_grad, "junction queries should require grad"
        out = model(x)
        x1_hat, x2_hat, features = out

        # tot loss
        rec_loss_1 = compute_rec_loss(criterion, x, x1_hat, pixel_wise=False)
        rec_loss_twin = compute_rec_loss(criterion, x1_hat.detach(), x2_hat, pixel_wise=False)

        # forward pass on auxilary data (regularization term)
        if auxilary_dl is not None:
            x_aux = next(auxilary_dl).to(device)
            out_aux = model(x_aux)
            x1_hat_aux, x2_hat_aux, _ = out_aux

            reg_1 = compute_rec_loss(criterion, x_aux, x1_hat_aux, pixel_wise=False)
            reg2 = torch.mean(torch.abs(x2_hat_aux), dim=(1,2,3), keepdim=False)

            rec_loss_1 += reg_1
            rec_loss_twin += reg2

        rec_loss = rec_loss_1 + rec_loss_twin

        rec_loss = torch.mean(rec_loss, dim=0) # single value for backprop (avg batch dim, sum h dim)

        # backward pass
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # store losses
        minibatch_losses["rec_tot"].append(rec_loss.item())
        minibatch_losses["rec_img1"].append(torch.mean(rec_loss_1).item())
        minibatch_losses["rec_twin"].append(torch.mean(rec_loss_twin).item())
        minibatch_sizes.append(x.size(0))

        # print batch info
        print(f"batch {i+1}/{len(train_dl)} - current losses: {torch.mean(rec_loss_1).item()} - {torch.mean(rec_loss_twin).item()}", end="\r")
                                                                                                 
    # epoch losses
    epoch_rec_loss_tot = np.sum(np.multiply(minibatch_losses["rec_tot"], minibatch_sizes)) / np.sum(minibatch_sizes)
    epoch_rec_loss_img1 = np.sum(np.multiply(minibatch_losses["rec_img1"], minibatch_sizes)) / np.sum(minibatch_sizes)
    epoch_rec_loss_twin = np.sum(np.multiply(minibatch_losses["rec_twin"], minibatch_sizes)) / np.sum(minibatch_sizes)

    if verbose:
        print(f"rec loss: {epoch_rec_loss_tot:.8f}", end=" | ")
        print(f"rec img1 loss: {epoch_rec_loss_img1:.8f}", end=" | ")
        print(f"rec loss twin: {epoch_rec_loss_twin:.8f}", end=" | ")

    return {
        "rec": epoch_rec_loss_tot,
        "rec_img1": epoch_rec_loss_img1,
        "rec_twin": epoch_rec_loss_twin
    }


def eval_step_twin(model,
                   val_dl,
                   criterion,
                   device,
                   verbose=True,
                   store_data=False,
                   reduction="mean",
                   multi_hstate_reduction="max"):
    """Evaluate model on validation set."""
    
    xs, labels = [], []
    x1_hats, x2_hats, features = [], [], []
    rec_amaps, ascores = [], []

    with torch.no_grad():
        for i, (x, ls, _, _) in enumerate(val_dl):
            print(f"eval batch {i+1}/{len(val_dl)}", end="\r")
            x = x.to(device)

            # forward pass
            out = model(x)
            x1_hat, x2_hat, f = out

            # tot loss
            rec_losses = compute_rec_loss(criterion, x1_hat, x2_hat, pixel_wise=True)
            rec_losses = torch.unsqueeze(rec_losses, 0) # (1, b, c, h, w)

            # store losses
            rec_amaps.append(rec_losses.cpu())
            ascores.append(score_anomaly_maps(rec_losses, reduction=reduction, multi_hstate_reduction=multi_hstate_reduction))
            labels.append(ls)
            if store_data:
                xs.append(x.cpu())
                features.append(f.cpu())
                x1_hats.append(x1_hat.cpu())
                x2_hats.append(x2_hat.cpu())

        rec_amaps = torch.cat(rec_amaps, dim=-4).numpy()
        ascores = torch.cat(ascores).numpy()
        labels = torch.cat(labels).numpy()
        if store_data:
            xs = torch.cat(xs, dim=-4).numpy()
            features = torch.cat(features).numpy()
            x1_hats = torch.cat(x1_hats, dim=-4).numpy()
            x2_hats = torch.cat(x2_hats, dim=-4).numpy()

    ascores_classavg = []
    for c in np.unique(labels):
        a_classavg = np.mean([a for a, l in zip(ascores, labels) if l==c])
        if verbose:
            print(f"val rec loss {c}: {a_classavg:.5f}", end = " - ")
        ascores_classavg.append((c, a_classavg))
    
    # compute AUROC for anomaly detection (using mean anomaly map as score)
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, ascores)
        if verbose:
            print(f"AUROC: {auroc:.5f}", end=" | ")
    else:
        auroc = np.nan
        
    out_dict = { f"val_rec_loss_{c}": v for c, v in ascores_classavg } | {"auroc": auroc}
    out_dict["labels"] = labels
    if store_data:
        out_dict["ascores"] = ascores
        out_dict["x"] = xs
        out_dict["features"] = features
        out_dict["x1_hat"] = x1_hats
        out_dict["x2_hat"] = x2_hats
        out_dict["anomaly_maps_tot"] = rec_amaps
    
    return out_dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim, ms_ssim

from model import ViTEncoder_timm, ResNet_timm

def resize_like(map, target):
    """Shortcut for resizing.
    map: (R, B, C, H, W) 
    target: (R, B, C, H, W) or (B, C, H, W)
    """
    if target.dim() == 4:  # (B, C, H, W)
        target_h, target_w = target.size(-2), target.size(-1)
    elif target.dim() == 5:  # (R, B, C, H, W)
        target_h, target_w = target.size(-2), target.size(-1)
    else:
        raise ValueError(f"Target tensor must be 4D or 5D, got {target.dim()}D")
    
    size = (target_h, target_w)
    
    if map.dim() == 4:  # (B, C, H, W) -> add R dimension
        map = map.unsqueeze(0)  # (1, B, C, H, W)
    
    # Now map should be (R, B, C, H, W)
    map_r, map_b, map_c, map_h, map_w = map.size()
    
    # Reshape to [R*B, C, H, W] for interpolation
    reshaped = map.reshape(-1, map_c, map_h, map_w)
    
    # Apply interpolation
    interpolated = F.interpolate(reshaped, size=size, mode="bilinear", align_corners=False)
    
    # Reshape back to original structure with new spatial dimensions
    result = interpolated.reshape(map_r, map_b, map_c, target_h, target_w)
    return result

def custom_MSE_loss(x, x_hat):
    loss = torch.nn.MSELoss(reduction='none')(x, x_hat)  # (B, C, H, W)
    return loss.unsqueeze(0)  # (1, B, C, H, W)

def custom_MAE_loss(x, x_hat):
    loss = torch.nn.L1Loss(reduction='none')(x, x_hat)  # (B, C, H, W)
    return loss.unsqueeze(0)  # (1, B, C, H, W)

def custom_SSIM_loss(x, x_hat):
    one = torch.ones_like(x[:,:1,:,:], device=x.device)
    s = ssim(x_hat, x, data_range=1., win_size=7, size_average=False)
    loss = one - s[:, None, None, None]  # (B, 1, 1, 1) -> broadcast to (B, 1, H, W)
    loss = loss.expand(-1, -1, x.size(-2), x.size(-1))  # (B, 1, H, W)
    return loss.unsqueeze(0)  # (1, B, 1, H, W)

def custom_MS_SSIM_loss(x, x_hat):
    one = torch.ones_like(x[:,:1,:,:], device=x.device)
    s = ms_ssim(x_hat, x, data_range=1., size_average=False)
    loss = one - s[:, None, None, None]  # (B, 1, 1, 1) -> broadcast to (B, 1, H, W)
    loss = loss.expand(-1, -1, x.size(-2), x.size(-1))  # (B, 1, H, W)
    return loss.unsqueeze(0)  # (1, B, 1, H, W)

def custom_COSSIM_loss(x, x_hat):
    one = torch.ones_like(x[:,:1,:,:], device=x.device)
    c = torch.nn.CosineSimilarity(dim=1)(x, x_hat)
    loss = one - c[:, None, :, :]  # (B, 1, H, W)
    return loss.unsqueeze(0)  # (1, B, 1, H, W)


def custom_perceptual_ViT_loss(x, x_hat, encoder,
                           metric="L2", use_hstate=[-1]):
    hstates = []
    hstates_hat = []
    _, encoder_hstates, _ = encoder(x)
    _, encoder_hstates_hat, _ = encoder(x_hat)

    for h_idx in use_hstate:
        hstates.append(encoder_hstates[h_idx])
        hstates_hat.append(encoder_hstates_hat[h_idx])

    distance_maps = []
    # difference 
    if metric=="L2":
        for i, (h, h_hat) in enumerate(zip(hstates, hstates_hat)):
            d = torch.pow(h-h_hat, 2).mean(1, keepdim=True)
            distance_maps.append(d)

    elif metric=="L2tot":
        for i, (h, h_hat) in enumerate(zip(hstates, hstates_hat)):
            d = torch.pow(h-h_hat, 2).mean(dim=(1,2,3), keepdim=True)
            # Expand to spatial dimensions
            d = d.expand(-1, -1, h.size(-2), h.size(-1))
            distance_maps.append(d)

    elif metric=="L1":
        for i, (h, h_hat) in enumerate(zip(hstates, hstates_hat)):
            d = torch.abs(h-h_hat).mean(1, keepdim=True)
            distance_maps.append(d)

    elif metric=="cosine":
        for i, (h, h_hat) in enumerate(zip(hstates, hstates_hat)):
            d = 1 - torch.nn.CosineSimilarity(dim=1)(h, h_hat)
            d = d[:, None, :, :]
            distance_maps.append(d)

    elif metric=="cosinetot":
        for i, (h, h_hat) in enumerate(zip(hstates, hstates_hat)):
            # flatten
            h_flat, h_hat_flat = h.view(h.size(0), -1), h_hat.view(h_hat.size(0), -1)
            # cosine
            d = 1 - torch.nn.CosineSimilarity(dim=1)(h_flat, h_hat_flat)
            d = d[:, None, None, None]
            # Expand to spatial dimensions
            d = d.expand(-1, -1, h.size(-2), h.size(-1))
            distance_maps.append(d)
    else: 
        raise ValueError(f"Metric {metric} not recognized.")
    
    result = torch.stack(distance_maps, dim=0)  # (num_hstates, B, C, H, W)
    return result
    

def custom_perceptual_ResNet_loss(x, x_hat, encoder,
                                  metric="L2", use_hstate=[-1]):

    features, features_hat = [], []

    encoder_features = encoder(x)
    encoder_features_hat = encoder(x_hat)
    
    for h_idx in use_hstate:
        features.append(encoder_features[h_idx])
        features_hat.append(encoder_features_hat[h_idx])
    
    # Find the largest feature map size for resizing reference
    sizes = [f.size()[-1] for f in features]
    argmax_size = np.argmax(sizes)
    target_size = features[argmax_size].size()[-2:]
    
    distance_maps = []
    
    # difference
    if metric=="L1":
        for i, (f, f_hat) in enumerate(zip(features, features_hat)):
            d = torch.abs(f-f_hat).mean(1, keepdim=True)
            distance_maps.append(d)

    elif metric=="L2":
        for i, (f, f_hat) in enumerate(zip(features, features_hat)):
            d = torch.pow(f-f_hat, 2).mean(1, keepdim=True)
            distance_maps.append(d)

    elif metric=="cosine":
        for i, (f, f_hat) in enumerate(zip(features, features_hat)):
            d = 1 - torch.nn.CosineSimilarity(dim=1)(f, f_hat)
            d = d[:, None, :, :]
            distance_maps.append(d)

    elif metric=="L2tot":
        for i, (f, f_hat) in enumerate(zip(features, features_hat)):
            d = torch.pow(f-f_hat, 2).mean(dim=(1,2,3), keepdim=True)
            # Expand to spatial dimensions
            d = d.expand(-1, -1, f.size(-2), f.size(-1))
            distance_maps.append(d)

    elif metric=="L1tot":
        for i, (f, f_hat) in enumerate(zip(features, features_hat)):
            d = torch.abs(f-f_hat).mean(dim=(1,2,3), keepdim=True)
            # Expand to spatial dimensions
            d = d.expand(-1, -1, f.size(-2), f.size(-1))
            distance_maps.append(d)

    elif metric=="cosinetot":
        for i, (f, f_hat) in enumerate(zip(features, features_hat)):
            # flatten
            f_flat, f_hat_flat = f.view(f.size(0), -1), f_hat.view(f_hat.size(0), -1)
            # cosine
            d = 1 - torch.nn.CosineSimilarity(dim=1)(f_flat, f_hat_flat)
            d = d[:, None, None, None]
            # Expand to spatial dimensions
            d = d.expand(-1, -1, f.size(-2), f.size(-1))
            distance_maps.append(d)

    elif metric=="content+style":
        for i, (f, f_hat) in enumerate(zip(features, features_hat)):
            # content
            d_content = torch.pow(f-f_hat, 2).mean((1,2,3), keepdim=True)
            d_content = d_content.expand(-1, -1, f.size(-2), f.size(-1))

            # style
            h, w = f.size(2), f.size(3)
            f_gram = torch.einsum("bchw,bdhw->bcd", f, f)/(h*w)
            f_hat_gram = torch.einsum("bchw,bdhw->bcd", f_hat, f_hat)/(h*w)
            d_style = torch.pow(f_gram-f_hat_gram, 2).mean((1,2))
            d_style = d_style[:, None, None, None]
            d_style = d_style.expand(-1, -1, f.size(-2), f.size(-1))

            # combine
            d = d_style + d_content
            distance_maps.append(d)
         
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    
    # Resize all distance maps to the same size
    resized_maps = []
    for i, d in enumerate(distance_maps):
        if d.size()[-2:] != target_size:
            d_resized = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
            resized_maps.append(d_resized)
        else:
            resized_maps.append(d)
    
    # Stack and average
    stacked = torch.stack(resized_maps, dim=0)  # (num_features, B, C, H, W)
    result = stacked.mean(0, keepdim=True)  # (1, B, C, H, W)
    return result


def criterion_selector(criterion_name: str, **kwargs):
    """Selects the criterion to be used among those available.
    criterion_name: criterion to be used.
    **kwargs: additional arguments (relevant for PERCEPTUAL loss).
    """
    if criterion_name=="EMBED_MSE":
        model = ViTEncoder_timm("vit_base_patch16_224.mae",
                                cache_dir=kwargs["cache_dir"],
                                n_patches=14, patch_size=16, img_size=kwargs["img_size"])
        model.to(kwargs["device"])
        model.eval()
        model.requires_grad_(False)

        loss = lambda x, x_hat: custom_MSE_loss(model.patch_embed(x), model.patch_embed(x_hat)) 

    elif criterion_name=="PERCEPTUAL_VIT_mae":
        model = ViTEncoder_timm("vit_large_patch16_224.mae",
                                cache_dir=kwargs["cache_dir"],
                                n_patches=14, patch_size=16, img_size=kwargs["img_size"])
        model.to(kwargs["device"])
        model.eval()
        model.requires_grad_(False)

        metric = kwargs["perceptual_metric"]
        hstate = kwargs["perceptual_hstate"]
        loss = lambda x, x_hat: custom_perceptual_ViT_loss(x, x_hat, model, metric, hstate)
        return loss

    elif criterion_name=="PERCEPTUAL_VIT_mae_multi":
        # Get patch sizes from config (default to original values if not specified)
        patch_sizes = kwargs.get("perceptual_patch_sizes", [16, 32, 56])
        img_size = kwargs["img_size"]
        
        models = []
        for patch_size in patch_sizes:
            # Calculate n_patches based on patch_size and img_size
            n_patches = img_size // patch_size
            if img_size % patch_size != 0:
                raise ValueError(f"Image size {img_size} is not divisible by patch size {patch_size}")
            
            model = ViTEncoder_timm('vit_large_patch16_224.mae',
                                   cache_dir=kwargs["cache_dir"],
                                   n_patches=n_patches, 
                                   patch_size=patch_size, 
                                   img_size=img_size)
            model.to(kwargs["device"])
            model.eval()
            model.requires_grad_(False)
            models.append(model)

        metric = kwargs["perceptual_metric"]
        hstate = kwargs["perceptual_hstate"]

        def multi_loss(x, x_hat):
            loss_maps = []
            for model in models:
                loss_map = custom_perceptual_ViT_loss(x, x_hat, model, metric, hstate)
                # Resize to input dimensions for multiplication
                loss_map_resized = resize_like(loss_map, x.unsqueeze(0))  # Add R dimension to x
                loss_maps.append(loss_map_resized)
            
            # Multiply all loss maps together
            result = loss_maps[0]
            for loss_map in loss_maps[1:]:
                result = result * loss_map
            
            # # Take the nth root to get geometric mean
            # n = len(loss_maps)
            # result = result ** (1.0 / n)
            
            return result
        
        return multi_loss

    elif criterion_name=="PERCEPTUAL_RESNET18":
        model = ResNet_timm("resnet18", kwargs["cache_dir"])
        model.to(kwargs["device"])
        model.eval()
        model.requires_grad_(False)

        metric = kwargs["perceptual_metric"]
        hstate = kwargs["perceptual_hstate"]
        loss = lambda x, x_hat: custom_perceptual_ResNet_loss(x, x_hat, model, metric, hstate)
        return loss

    elif criterion_name=="PERCEPTUAL_RESNET34":
        model = ResNet_timm("resnet34", kwargs["cache_dir"])
        model.to(kwargs["device"])
        model.eval()
        model.requires_grad_(False)

        metric = kwargs["perceptual_metric"]
        hstate = kwargs["perceptual_hstate"]
        loss = lambda x, x_hat: custom_perceptual_ResNet_loss(x, x_hat, model, metric, hstate)
        return loss
    
    elif criterion_name=="PERCEPTUAL_RESNET101":
        model = ResNet_timm("resnet101", kwargs["cache_dir"])
        model.to(kwargs["device"])
        model.eval()
        model.requires_grad_(False)

        metric = kwargs["perceptual_metric"]
        hstate = kwargs["perceptual_hstate"]
        loss = lambda x, x_hat: custom_perceptual_ResNet_loss(x, x_hat, model, metric, hstate)
        return loss

    else:
        losses = {
            "MSE": custom_MSE_loss,
            "MAE": custom_MAE_loss,
            "SSIM": custom_SSIM_loss,
            "MS_SSIM": custom_MS_SSIM_loss,
            "COSSIM": custom_COSSIM_loss
        }

        return losses[criterion_name]

def multicriterion_selector(names: list[str,],
                            weights: list[float,] = None,
                            **kwargs
                            ):
    """To use multiple criteria at once (returns a dictionary {"ith_name": ith_loss})."""
    losses = [criterion_selector(n, **kwargs) for n in names]
    if weights is not None:
        assert len(weights)==len(names), "Invalid number of params"
    else:
        weights = [1.]*len(names)

    tot_loss = lambda x, x_hat: {n : w*l(x, x_hat) for n, l, w in zip(names, losses, weights)}
    return tot_loss
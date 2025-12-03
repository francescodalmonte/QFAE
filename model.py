import os
import sys
from typing import List

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import EncoderBlock
from timm.models.vision_transformer import Block
import timm
from transformers import CLIPVisionModel, Dinov2Model, AutoImageProcessor

from zeta.nn import MultiQueryAttention, SimpleFeedForward
from attention import CrossAttention


def patchify(imgs, patch_size, n_ch=3):
    """
    (code adapted from: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py#L95)

    imgs: (N, n_ch, H, W)
    x: (N, L, patch_size**2 *n_ch)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0, "img must be square and divisible by patch size"

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], n_ch, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * n_ch))
    return x


def unpatchify(x, patch_size, n_ch=3):
    """
    (code adapted from: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py#L109)

    x: (N, L, patch_size**2 *n_ch)
    imgs: (N, n_ch, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, n_ch))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], n_ch, h * p, h * p))
    return imgs


def create_masks(
        n_patches_q: int,
        n_patches_v: List[int, ],
        n_reg_tokens: List[int, ],
        device: str = "cpu"
    ):
    """Returns a tensor of shape (len(n_patches_v), 1, 1, sum(n_patches_v**2), n_patches_q**2) 
    to be used as mask in the cross-attention mechanism."""

    # masks only make sense if there are multiple encoders
    assert len(n_patches_v) > 1, "Cannot do random masking with only one encoder"

    mask = []
    
    # iterate over different encoders (i.e. different patch sizes/n. reg. tokens)
    for i, (nv, nr) in enumerate(zip(n_patches_v, n_reg_tokens)):
        mask.append(torch.ones(int(n_patches_q**2), int(nv**2+nr+1), device=device)*i)
    mask = torch.cat(mask, dim=1)

    # create a set of masks, each one leaving two encoders visible
    masks_set = [mask != i for i in range(len(n_patches_v))]
    # add one "transparent" mask (i.e. all encoders visible)
    masks_set = torch.stack([torch.ones_like(mask, dtype=bool)] + masks_set, dim=0)

    return masks_set[:, None, :, :].to(device)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    (code adapted from: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py)
    
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    (code adapted from: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py)

    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Get 2D positional embedding using sin-cos functions.
    (code adapted from: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py)

    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).float().unsqueeze(0)


class ResNet_timm(torch.nn.Module):
    """ResNet using timm models."""
    def __init__(
            self,
            model_name: str,
            cache_dir: str
        ):
        super(ResNet_timm, self).__init__()
        self.model_name = model_name
        torch.hub.set_dir(cache_dir)
        self.model = timm.create_model(
            model_name,
            pretrained=True
        )

    def forward(self, x):
        return self.model.forward_intermediates(x)[-1]


class ViTEncoder_timm(torch.nn.Module):
    """ViTEncoder using timm models (enable flexible patch_size).""" 
    def __init__(
            self,
            model_name: str,
            cache_dir: str,
            n_patches: int, # needed for reshaping hidden states
            patch_size: int,
            img_size: int = 224,
            additional_block_params: dict = None,
            final_projection_params: dict = None,
            use_hidden_state: list[int,] = None
        ):
        super(ViTEncoder_timm, self).__init__()
        self.model_name = model_name
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.hidden_state = use_hidden_state
        
        # pretrained ViT from timm
        torch.hub.set_dir(cache_dir)
        print(f"Loading model {model_name} from timm")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            patch_size=self.patch_size,
            img_size=self.img_size
        )

        # (optional) additional transformer block
        if additional_block_params is not None:
            params = additional_block_params
            self.additional_block = Block(**params)
        else:
            print("No additional block")
            self.additional_block = nn.Identity()

        print("final_projection_params: ", end="")
        # (optional) final projection
        if final_projection_params is not None:
            print(final_projection_params)
            self.final_projection = nn.Linear(**final_projection_params)
        else:
            print("No final projection")
            self.final_projection = nn.Identity()


    def forward(self, x, norm=False):
        # hstates = list of tuples: ([b, nregisters+1, d], [b, l, d])
        hstates = self.model.forward_intermediates(
            x, output_fmt="NLC", intermediates_only=True,
            return_prefix_tokens=True, norm=norm
        )

        # concat register/cls token with patch tokens
        hstates = [(torch.cat([h_cls, h_patch], dim=1)) for h_patch, h_cls in hstates]
        n_registers = hstates[0].shape[1] - self.n_patches**2

        if self.hidden_state is not None:
            # if a list of hidden states is provided, they are concatenated and fed to the additional block
            x = [hstates[h] for h in self.hidden_state]
            x = torch.cat(x, dim=-2)
        else: # otherwise, the last hidden state is used
            x = hstates[-1]
        x = self.additional_block(x)
        x = self.final_projection(x)

        # reshape hidden states (B,L,D -> B,D,H,W) and return them as second output (intermediate features)
        reshaped_hstates = [torch.einsum('bld->bdl', h[:, n_registers:]) for h in hstates]
        reshaped_hstates = [h.reshape(shape=(h.shape[0], h.shape[1], self.n_patches, self.n_patches)) for h in reshaped_hstates]
        # return register tokens
        register_tokens = [h[:, :n_registers] for h in hstates]
        
        return x, reshaped_hstates, register_tokens


class Decoder(torch.nn.Module):
    """Decoder module for ViT-based autoencoder.
    (code adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py)"""
    def __init__(self,
                 img_size: int,
                 n_patches: int,
                 patch_size: int,
                 embed_dim: int,
                 decoder_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float,
                 out_channels: int,
                 pos_embed: bool = True,
                 cls_token: bool = False,
                 reg_tokens: int = 0,
        ):
        super(Decoder, self).__init__()
        
        self.out_channels = out_channels
        self.img_size = img_size
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_dim), requires_grad=True) if cls_token else None 
        self.reg_tokens = nn.Parameter(torch.zeros(1, reg_tokens, decoder_dim), requires_grad=True) if reg_tokens > 0 else None
        # number of additional tokens (cls_token + reg_tokens)
        self.n_add_tokens = int(cls_token) + reg_tokens

        # projection (from embed to decoder dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        # positional embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_patches**2, decoder_dim), requires_grad=False) if pos_embed else None
        # decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # projection (from decoder dim to image)
        self.decoder_prediction = nn.Linear(decoder_dim, (patch_size**2)*out_channels, bias=True)

        self.init_weights()


    def init_weights(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-3)
        if self.reg_tokens is not None:
            nn.init.normal_(self.reg_tokens, std=1e-3)
        if self.decoder_pos_embed is not None:
            # initialize with 2D sin-cos positional embedding
            pembed = get_2d_sincos_pos_embed(self.decoder_dim, self.n_patches, cls_token=False)
            self.decoder_pos_embed.data.copy_(pembed)


    def forward(self, x):
        x = self.decoder_embed(x)

        if self.decoder_pos_embed is not None:
            x = x + self.decoder_pos_embed

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([reg_tokens, x], dim=1)

        hstates = []
        register_tokens = []
        for block in self.decoder_blocks:
            x = block(x)

            register_tokens.append(x[:, :self.n_add_tokens] if self.n_add_tokens > 0 else None)
            h_ = torch.einsum('bld->bdl', x[:, self.n_add_tokens:]) # (b, l, d) -> (b, d, l)
            hstates.append(h_.reshape(shape=(x.shape[0], x.shape[2], self.n_patches, self.n_patches))) # added: return reshaped hidden states
        
        x = self.decoder_norm(x)
        x = self.decoder_prediction(x)[:, self.n_add_tokens:]
        x = unpatchify(x, self.patch_size, self.out_channels)
        if self.out_channels == 1:
            x = torch.cat([x, x, x], dim=1)     

        return x, hstates, register_tokens


class mqformer(nn.Module):
    """Junction building block"""
    def __init__(self,
                 dim: int,
                 output_dim: int,
                 heads: int,
                 mlp_ratio: float = 4.,
                 dropout: float = 0.01,
                 first_block : bool = True,
                 cross_attn: bool = True
        ):
        super(mqformer, self).__init__()

        self.dim = dim
        self.heads = heads
        self.first_block = first_block


        self.attn = MultiQueryAttention(
            dim=dim, heads=heads
        )
        
        if cross_attn:
            self.cross_attn = CrossAttention(
                dim=dim, heads=heads, dropout=0.1,
            )
        else:
            self.cross_attn = lambda x, _: (x, None)

        self.projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, queries, patch_tokens, mask=None):
        # queries shape: (1|b, n_queries, dim)
        # patch_tokens shape: (b, l, dim)

        if self.first_block:
            b = patch_tokens.shape[0]
            queries = queries.tile(dims=(b,1,1)) # shape: (b, n_queries, dim)
        
        # self-attention
        x, _, _ = self.attn(queries) # shape: (b, n_queries, dim)

        # cross-attention (query is tiled along batch dimension)
        x, attnmaps = self.cross_attn(x, patch_tokens, mask=mask) # shape: (b, n_queries, dim)

        # feedforward
        x = self.projection(x) # shape: (b, n_queries, dim)

        return x, attnmaps


class Junction(nn.Module):
    """Module performing multi-query attention, cross-attention and feedforward operations.
    This version uses multiple self/cross attention blocks."""
    def __init__(self,
                 dim: int,
                 output_dim: int,
                 n_queries: int,
                 heads: int,
                 mlp_ratio: float = 4.,
                 dropout: float = 0.01,
                 n_blocks: int = 1
        ):
        super(Junction, self).__init__()

        self.dim = dim
        self.heads = heads
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim), requires_grad=True)
        nn.init.normal_(self.queries, std=1e-3)

        self.blocks = nn.ModuleList([
            mqformer(dim=dim if i==0 else output_dim, output_dim=output_dim if i==n_blocks-1 else dim,
                     heads=heads, mlp_ratio=mlp_ratio, dropout=dropout, first_block=(i==0),
                     cross_attn=(i%2==0)) # cross_attn every other block
            for i in range(n_blocks)
        ])

    def forward(self, patch_tokens, mask=None):
        x = self.queries
        for block in self.blocks:
            x, attnmaps = block(x, patch_tokens, mask=mask)
        return x, attnmaps


class ViTAE(nn.Module):
    def __init__(self, 
                 model_name: str,
                 cache_dir: str,
                 img_size: int,
                 n_patches: int,
                 patch_size: int,
                 embed_dim: int,
                 decoder_dim: int,
                 decoder_depth: int,
                 decoder_num_heads: int,
                 decoder_mlp_ratio: float,
                 decoder_out_channels: int,
                 additional_block_params: dict = None,
                 final_projection_params: dict = None,
                 use_hidden_state: list[int,] = None
        ):
        super(ViTAE, self).__init__()
        
        self.img_size = img_size
        self.n_patches = n_patches
        self.patch_size = patch_size

        # vision encoder
        self.encoder = ViTEncoder_timm(model_name,
                                       cache_dir,
                                       n_patches,
                                       patch_size,
                                       img_size,
                                       additional_block_params,
                                       final_projection_params,
                                       use_hidden_state
                                       )
        # decoder
        self.decoder = Decoder(img_size,
                               n_patches,
                               patch_size, 
                               embed_dim,
                               decoder_dim,
                               decoder_depth,
                               decoder_num_heads,
                               decoder_mlp_ratio,
                               decoder_out_channels
                               )

    def forward(self, x, device):
        # encode image
        features, hstates_e, _ = self.encoder(x, device)
        # decode to image space
        x_hat, hstates_d, _ = self.decoder(features)

        return x_hat, features, hstates_e, hstates_d


class MQViTAE(nn.Module):
    def __init__(self, 
                 model_name: list[str, ],
                 cache_dir: list[str, ],
                 img_size: int,
                 n_patches: list[int, ],
                 patch_size: list[int, ],
                 n_reg_tokens: list[int, ],
                 junction_n_blocks: int,
                 junction_dim: int,
                 junction_output_dim: int,
                 junction_n_queries: int,
                 junction_heads: int,
                 junction_mlp_ratio: float,
                 decoder_dim: int,
                 decoder_n_patches: int,
                 decoder_patch_size: int,
                 decoder_depth: int,
                 decoder_num_heads: int,
                 decoder_mlp_ratio: float,
                 decoder_out_channels: int,
                 decoder_n_reg_tokens: int,
                 additional_block_params: None, # TODO: implement
                 final_projection_params: list[dict, ] = None,
                 use_hidden_state: list[int, ] = None, # TODO: implement
        ):
        super(MQViTAE, self).__init__()
        
        assert len(model_name)==len(cache_dir)==len(n_patches)==len(patch_size), "Inconsistent number of models params"

        self.img_size = img_size
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.use_hidden_state = use_hidden_state
        self.decoder_n_patches = decoder_n_patches
        self.decoder_patch_size = decoder_patch_size
                                        
        # vision encoder(s)
        self.encoders = nn.ModuleList([])
        for i in range(len(model_name)):
            self.encoders.append(ViTEncoder_timm(model_name[i],
                                                 cache_dir[i],
                                                 n_patches[i],
                                                 patch_size[i],
                                                 img_size,
                                                 additional_block_params,
                                                 final_projection_params[i],
                                                 use_hidden_state
                                                 ))

        # junction
        self.junction = Junction(junction_dim,
                                 junction_output_dim,
                                 junction_n_queries,
                                 junction_heads,
                                 junction_mlp_ratio, 
                                 n_blocks=junction_n_blocks,
                                 )

        # decoder
        self.decoder = Decoder(img_size,
                               decoder_n_patches,
                               decoder_patch_size, 
                               junction_output_dim,
                               decoder_dim,
                               decoder_depth,
                               decoder_num_heads,
                               decoder_mlp_ratio,
                               decoder_out_channels,
                               pos_embed = True,
                               cls_token = True,
                               reg_tokens = decoder_n_reg_tokens
                               )
        
    def forward(self, x, crossattn_mask=None):
        # encode image (parallel)
        futures = [torch.jit.fork(e, x) for e in self.encoders]

        # collect results
        features_list = [torch.jit.wait(f)[0] for f in futures]

        # concatenate features
        features = torch.cat(features_list, dim=1)

        # junction
        x, attnmaps = self.junction(features, mask=crossattn_mask)
        # decode to image space
        x_hat, _, _ = self.decoder(x)

        return x_hat, features, attnmaps


class MQViTAE_twin(nn.Module):
    def __init__(self, 
                 model_name: str,
                 cache_dir: str,
                 img_size: int,
                 n_patches: int,
                 patch_size: int,
                 junction_dim: int,
                 junction_output_dim: int,
                 junction_n_queries: int,
                 junction_heads: int,
                 junction_mlp_ratio: float,
                 decoder_dim: int,
                 decoder_depth: int,
                 decoder_num_heads: int,
                 decoder_mlp_ratio: float,
                 decoder_out_channels: int,
                 additional_block_params: dict = None,
                 final_projection_params: dict = None,
                 use_hidden_state: list[int,] = None
        ):
        super(MQViTAE_twin, self).__init__()
        
        self.img_size = img_size
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.use_hidden_state = use_hidden_state

        # vision encoder
        self.encoder = ViTEncoder_timm(model_name,
                                       cache_dir,
                                       n_patches,
                                       patch_size,
                                       img_size,
                                       additional_block_params,
                                       final_projection_params,
                                       use_hidden_state
                                       )
        # junction
        self.junction = Junction(junction_dim,
                                 junction_output_dim,
                                 junction_n_queries,
                                 junction_heads,
                                 junction_mlp_ratio
                                 )

        # decoder
        self.decoder = Decoder(img_size,
                               n_patches,
                               patch_size, 
                               junction_output_dim,
                               decoder_dim,
                               decoder_depth,
                               decoder_num_heads,
                               decoder_mlp_ratio,
                               decoder_out_channels,
                               pos_embed = True,
                               cls_token = True,
                               reg_tokens = 4
                               )
        
    def forward(self, x):
        # encode image
        features, _, regs_e = self.encoder(x)

        # remove register tokens
        nreg = regs_e[0].shape[1]
        if self.use_hidden_state is not None and len(self.use_hidden_state) > 1:
            enc_dim = features.shape[1]/len(self.use_hidden_state) - nreg
            enc_dim = int(enc_dim)
            features = torch.cat([features[:, i*(enc_dim+nreg):((i*nreg)+((i+1)*enc_dim))]
                                  for i in range(len(self.use_hidden_state))], dim=1)
        else:
            features = features[:, nreg:]

        # junction w/o info bottleneck
        x1 = features
        # junction w/ info bottleneck
        x2 = self.junction(features)

        # decode to image space (1)
        x1_hat, _, _ = self.decoder(x1)
        # decode to image space (2)
        x2_hat, _, _ = self.decoder(x2)

        return x1_hat, x2_hat, features

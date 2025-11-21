# network/bimba_vim.py

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from utils.mamba_simple_xai import Mamba

try:
    from utils.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:  # safe fallback when Triton is unavailable
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import ml_collections


def get_ml_config(params) -> ml_collections.ConfigDict:
    """
    Returns a minimal config object for the model.

    Only keeps fields that are used by the current code:
      - hidden_size
    """
    config = ml_collections.ConfigDict()
    config.hidden_size = params["hidden_size"]
    config.dropout_rate = params.get("dropout", 0.0)
    return config

class PointFeatureEncoder(nn.Module):
    """
    Encodes residue-level features into a 128-D embedding.

    Expected feature layout per residue:
      - scalar  : x[:, :4]
      - vector  : x[:, 4:7]
      - polarity: x[:, 7:11]
      - residue one-hot: x[:, 11:32]
      - secondary-structure one-hot: x[:, 32:]
    """

    def __init__(self, residue_dim: int = 21, ss_dim: int = 8, out_dim: int = 16):
        super().__init__()
        self.scalar_fc = nn.Linear(4, out_dim)
        self.vector_fc = nn.Linear(3, out_dim)
        self.polarity_fc = nn.Linear(4, out_dim)
        self.residue_fc = nn.Linear(residue_dim, out_dim)
        self.ss_fc = nn.Linear(ss_dim, out_dim)
        self.act = nn.GELU()

        concat_dim = 5 * out_dim
        self.last_fc = nn.Linear(concat_dim, 128)
        self.act_combine = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        scalar = x[:, :4]
        vector = x[:, 4:7]
        polarity = x[:, 7:11]
        residue_oh = x[:, 11:32]
        ss_oh = x[:, 32:]

        scalar_feat = self.act(self.scalar_fc(scalar))
        vector_feat = self.act(self.vector_fc(vector))
        polarity_feat = self.act(self.polarity_fc(polarity))
        residue_feat = self.act(self.residue_fc(residue_oh))
        ss_feat = self.act(self.ss_fc(ss_oh))

        concat_all = torch.cat(
            [scalar_feat, vector_feat, polarity_feat, residue_feat, ss_feat],
            dim=-1,
        )
        ext_feats = self.act_combine(self.last_fc(concat_all))
        return ext_feats  # [B, 128]


class PatchEmbed(nn.Module):
    """
    2D image -> patch embeddings via Conv2d.

    Input:  [B, C, H, W]
    Output: [B, N, D] where N = num_patches, D = embed_dim
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        stride: int = 16,
        in_chans: int = 3,
        embed_dim: int = 128,
        norm_layer=None,
        flatten: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}x{W}) "
            f"doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        )
        x = self.proj(x)  # [B, D, H', W']
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, D, HW] -> [B, HW, D]
        x = self.norm(x)
        return x

class Block(nn.Module):
    """
    One Mamba block with optional fused residual + norm.

    Structure:
        residual <- residual + DropPath(hidden_states)
        hidden_states <- Norm(residual)
        hidden_states <- Mamba(hidden_states)
    """

    def __init__(
        self,
        dim: int,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None and layer_norm_fn is not None and rms_norm_fn is not None, (
                "Fused add + norm requires Triton RMSNorm to be installed."
            )
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model: int,
    d_state: int,
    norm_epsilon: float,
    drop_path: float,
    rms_norm: bool,
    residual_in_fp32: bool,
    fused_add_norm: bool,
    layer_idx: int,
    device=None,
    dtype=None,
) -> Block:
    """
    Factory for a single Mamba block.
    """
    ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        use_fast_path=True,
        layer_idx=layer_idx,
        **ssm_cfg,
        **factory_kwargs,
    )

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon,
        **factory_kwargs,
    )

    block = Block(
        dim=d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    """
    Initialization similar to GPT-style + mild rescaling.
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m: nn.Module):
    """
    Simple trunc_normal/init for linear + layernorm layers.
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# -------------------------------------------------------------------------
# Main model: Vision Mamba + residue features
# -------------------------------------------------------------------------
class VisionMambaWithFeatures(nn.Module):
    """
    Vision Mamba encoder for 2D protein surface patches + residue-level features.

    Args (semi-clean constructor, Option B):
        config:         ml_collections config with .hidden_size and .transformer.dropout_rate
        img_size:       input image size H=W
        patch_size:     patch size (Conv2d kernel)
        stride:         stride for Conv2d
        depth:          number of Mamba blocks
        d_state:        Mamba state dimension
        channels:       input channels per image
        num_classes:    number of classes (we output one logit, so this is mainly for clarity)
        drop_rate:      dropout rate (positional dropout)
        drop_path_rate: stochastic depth
        rms_norm:       use RMSNorm instead of LayerNorm
        use_abs_pos:    enable learnable absolute position embeddings
        bidirectional:  use bidirectional Mamba (fwd + reversed)
        use_cls:        use CLS token
        use_middle_cls: insert CLS token in the middle of the sequence
    """

    def __init__(
        self,
        config,
        img_size: int = 18,
        patch_size: int = 3,
        stride: int = 3,
        depth: int = 8,
        d_state: int = 8,
        channels: int = 5,
        num_classes: int = 2,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        rms_norm: bool = True,
        use_abs_pos: bool = True,
        bidirectional: bool = True,
        use_cls: bool = True,
        use_middle_cls: bool = True,
        norm_epsilon: float = 1e-5,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.config = config
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_abs_pos = use_abs_pos
        self.use_cls = use_cls
        self.use_middle_cls = use_middle_cls
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        embed_dim = config.hidden_size
        factory_kwargs = {"device": device, "dtype": dtype}

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.num_tokens = 1 if use_cls else 0
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Absolute positional embedding
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)
        else:
            self.pos_embed = None
            self.pos_drop = nn.Identity()

        # Mamba layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=embed_dim,
                    d_state=d_state,
                    norm_epsilon=norm_epsilon,
                    drop_path=dpr[i],
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # Final normalization for the sequence
        if rms_norm and RMSNorm is not None:
            self.norm_f = RMSNorm(embed_dim, eps=norm_epsilon, **factory_kwargs)
        else:
            self.norm_f = nn.LayerNorm(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # Residue-level feature encoder
        self.extra_feats_encoder = PointFeatureEncoder()

        # Combine CLS + residue features
        combined_dim = embed_dim + 128
        self.layernorm = nn.LayerNorm(combined_dim)
        self.cls_fc = nn.Linear(combined_dim, 1)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # Initialization
        self.patch_embed.apply(segm_init_weights)
        self.cls_fc.apply(segm_init_weights)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if use_cls:
            trunc_normal_(self.cls_token, std=0.02)

        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                initializer_range=0.02,
                rescale_prenorm_residual=True,
                n_residuals_per_layer=1,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


    def forward_features(self, x: Tensor, inference_params=None) -> Tensor:
        """
        Returns CLS embedding [B, D].
        """
        x = self.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape

        # CLS token
        if self.use_cls:
            cls_token = self.cls_token.expand(B, 1, D)
            if self.use_middle_cls:
                half = N // 2
                x = torch.cat((x[:, :half], cls_token, x[:, half:]), dim=1)
                cls_index = half
            else:
                x = torch.cat((cls_token, x), dim=1)
                cls_index = 0
        else:
            cls_index = 0

        # Positional embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        residual = None
        hidden_states = x

        if self.bidirectional:
            # use forward and reversed sequence pairs
            num_layers = len(self.layers)
            assert num_layers % 2 == 0, "Bidirectional mode expects an even number of layers."

            for i in range(num_layers // 2):
                # forward
                hidden_states_f, residual_f = self.layers[2 * i](
                    hidden_states, residual, inference_params=inference_params
                )
                # backward (on reversed sequence)
                hidden_states_b, residual_b = self.layers[2 * i + 1](
                    hidden_states.flip(dims=[1]),
                    None if residual is None else residual.flip(dims=[1]),
                    inference_params=inference_params,
                )

                hidden_states = hidden_states_f + hidden_states_b.flip(dims=[1])
                residual = residual_f + residual_b.flip(dims=[1])
        else:
            for layer in self.layers:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        # Final norm + residual
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        cls_embed = hidden_states[:, cls_index, :]  # [B, D]
        return cls_embed


    def forward(
        self,
        x: Tensor,
        extra_feat: Tensor,
        return_features: bool = False,
        inference_params=None,
    ):
        """
        Args:
            x: [B, C, H, W] surface patch images.
            extra_feat: [B, F] residue-level features.

        Returns:
            logits: [B]
            optionally, (logits, cls_embed, extra_embed, combined)
        """
        x = self.forward_features(x, inference_params)
        extra_feat = self.extra_feats_encoder(extra_feat)
        x = torch.cat([x, extra_feat], dim=1)  # Concatenate extra features
        x = self.layernorm(x)
        x = self.cls_fc(x)
        return x


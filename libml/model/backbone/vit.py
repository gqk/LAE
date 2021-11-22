""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman

Credit: https://raw.githubusercontent.com/rwightman/pytorch-image-models/v0.5.4/timm/models/vision_transformer.py
"""

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.models.helpers import (
    adapt_input_conv,
    build_model_with_cfg,
    named_apply,
)
from timm.models.layers import (
    DropPath,
    PatchEmbed,
    lecun_normal_,
    trunc_normal_,
)
from timm.models.layers.helpers import to_2tuple

from ..pet_mixin import AdapterMixin, PrefixMixin, PromptMixin

_logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        **kwargs,
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    "vit_tiny_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_tiny_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_base_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_base_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch8_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_large_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
    ),
    "vit_large_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_large_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_huge_patch14_224": _cfg(url=""),
    "vit_giant_patch14_224": _cfg(url=""),
    "vit_gigantic_patch14_224": _cfg(url=""),
    # patch models, imagenet21k (weights from official Google JAX impl)
    "vit_tiny_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_small_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_small_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_base_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_base_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    ),
    "vit_base_patch8_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    ),
    "vit_large_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
    ),
    "vit_large_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz",
    ),
    "vit_huge_patch14_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz",
        hf_hub="timm/vit_huge_patch14_224_in21k",
    ),
    # SAM trained models (https://arxiv.org/abs/2106.01548)
    "vit_base_patch32_sam_224": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz"
    ),
    "vit_base_patch16_sam_224": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz"
    ),
    # deit models (FB weights)
    "deit_tiny_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_small_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "deit_tiny_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_small_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_distilled_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    # ViT ImageNet-21K-P pretraining by MILL
    "vit_base_patch16_224_miil_in21k": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
    ),
    "vit_base_patch16_224_miil": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm"
        "/vit_base_patch16_224_1k_miil_84_4.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
    ),
    # patch models, paper (weights from official Google JAX impl)
    "vit_b32": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_32.npz",
    ),
    "vit_b16": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz",
    ),
    "vit_b8": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_8.npz",
    ),
    "vit_l32": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_32.npz",
    ),
    "vit_l16": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz",
    ),
    # patch models, paper imagenet21k (weights from official Google JAX impl)
    "vit_b32_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz",
    ),
    "vit_b16_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
    ),
    "vit_b8_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_8.npz",
    ),
    "vit_l32_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz",
    ),
    "vit_l16_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz",
    ),
}


class Mlp(nn.Module, AdapterMixin):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.adapt_module("fc1", x)  # x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.adapt_module("fc2", x)  # x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module, PromptMixin, PrefixMixin, AdapterMixin):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.add_prompt(x)

        B, N, C = x.shape
        qkv = self.adapt_module("qkv", x)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.chunk(3, dim=-1)
        k, v = self.add_prefix(k, v)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(B, -1, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.compensate_prefix(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.adapt_module("proj", x)  # x = self.proj(x)
        x = self.proj_drop(x)

        x = self.reduce_prompt(x)

        return x


class Block(nn.Module, AdapterMixin):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(
            self.adapt_module("attn", self.norm1(x))  # self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.adapt_module("mlp", self.norm2(x))  # self.mlp(self.norm2(x))
        )
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()

        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        if mode.startswith("jax"):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)  # shape: B N C
        return x


def _init_vit_weights(
    module: nn.Module, name: str = "", jax_impl: bool = False
):
    """ViT weight initialization
    * When called without n, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if jax_impl:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if "mlp" in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(
    model: VisionTransformer, checkpoint_path: str, prefix: str = ""
):
    """Load weights from .npz checkpoints for official Google Brain Flax implementation"""
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(
                stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])
            )
        )
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}conv{r + 1}/kernel"])
                        )
                        getattr(block, f"norm{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/scale"])
                        )
                        getattr(block, f"norm{r + 1}").bias.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/bias"])
                        )
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f"{bp}conv_proj/kernel"])
                        )
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f"{bp}gn_proj/scale"])
                        )
                        block.downsample.norm.bias.copy_(
                            _n2p(w[f"{bp}gn_proj/bias"])
                        )
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1],
            _n2p(w[f"{prefix}embedding/kernel"]),
        )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    pos_embed_w = _n2p(
        w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False
    )
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.proj.weight.copy_(
            _n2p(w[f"{mha_prefix}out/kernel"]).flatten(1)
        )
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
            )
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info(
        "Resized position embedding: %s to %s", posemb.shape, posemb_new.shape
    )
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(
        "Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new
    )
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, gs_new[0] * gs_new[1], -1
    )
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        # ignore head and pre_logits
        if k.startswith("head.") or k.startswith("pre_logits"):
            continue
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(
    variant, pretrained=True, default_cfg=None, **kwargs
):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    model = build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in default_cfg["url"],
        **kwargs,
    )
    return model


def vit_tiny_patch16_224(pretrained=True, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs
    )
    model = _create_vision_transformer(
        "vit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_tiny_patch16_384(pretrained=True, **kwargs):
    """ViT-Tiny (Vit-Ti/16) @ 384x384."""
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs
    )
    model = _create_vision_transformer(
        "vit_tiny_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch32_224(pretrained=True, **kwargs):
    """ViT-Small (ViT-S/32)"""
    model_kwargs = dict(
        patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "vit_small_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch32_384(pretrained=True, **kwargs):
    """ViT-Small (ViT-S/32) at 384x384."""
    model_kwargs = dict(
        patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "vit_small_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_224(pretrained=True, **kwargs):
    """ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_384(pretrained=True, **kwargs):
    """ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "vit_small_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_224(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_384(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_384(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch8_224(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch8_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch32_224(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights."""
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_large_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch32_384(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_large_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch16_224(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch16_384(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_large_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_sam_224(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548"""
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch16_sam_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_sam_224(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548"""
    model_kwargs = dict(
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch32_sam_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_huge_patch14_224(pretrained=True, **kwargs):
    """ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929)."""
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_huge_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_giant_patch14_224(pretrained=True, **kwargs):
    """ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560"""
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1408,
        mlp_ratio=48 / 11,
        depth=40,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_giant_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_gigantic_patch14_224(pretrained=True, **kwargs):
    """ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560"""
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1664,
        mlp_ratio=64 / 13,
        depth=48,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_gigantic_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_tiny_patch16_224_in21k(pretrained=True, **kwargs):
    """ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs
    )
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch32_224_in21k(pretrained=True, **kwargs):
    """ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "vit_small_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_224_in21k(pretrained=True, **kwargs):
    """ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "vit_small_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_224_in21k(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224_in21k(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch8_224_in21k(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch8_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch32_224_in21k(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_large_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch16_224_in21k(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_large_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_huge_patch14_224_in21k(pretrained=True, **kwargs):
    """ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_huge_patch14_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def deit_tiny_patch16_224(pretrained=True, **kwargs):
    """DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs
    )
    model = _create_vision_transformer(
        "deit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def deit_small_patch16_224(pretrained=True, **kwargs):
    """DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def deit_base_patch16_224(pretrained=True, **kwargs):
    """DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def deit_base_patch16_384(pretrained=True, **kwargs):
    """DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "deit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def deit_tiny_distilled_patch16_224(pretrained=True, **kwargs):
    """DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs
    )
    model = _create_vision_transformer(
        "deit_tiny_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def deit_small_distilled_patch16_224(pretrained=True, **kwargs):
    """DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer(
        "deit_small_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def deit_base_distilled_patch16_224(pretrained=True, **kwargs):
    """DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "deit_base_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def deit_base_distilled_patch16_384(pretrained=True, **kwargs):
    """DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "deit_base_distilled_patch16_384",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def vit_base_patch16_224_miil_in21k(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_miil_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224_miil(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_miil", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_b32(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_b32", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_b16(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_b16", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_b8(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_b8", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_l32(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights."""
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_l32", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_l16(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_l16", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_b32_in21k(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_b32_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_b16_in21k(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_b16_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_b8_in21k(pretrained=True, **kwargs):
    """ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer(
        "vit_b8_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_l32_in21k(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_l32_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_l16_in21k(pretrained=True, **kwargs):
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer(
        "vit_l16_in21k", pretrained=pretrained, **model_kwargs
    )
    return model

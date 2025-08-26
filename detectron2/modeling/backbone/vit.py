import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from functools import partial
from typing import Dict, List, Optional, Tuple, Type

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    FrozenBatchNorm2d,
    ShapeSpec,
    get_norm,
)
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from detectron2.structures import Instances

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
)

logger = logging.getLogger(__name__)


__all__ = [
    "ViT",
    "SimpleFeaturePyramid",
    "get_vit_lr_decay_rate",
    "build_vit_backbone",
    "build_vitdet_backbone",
]


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        use_residual_block: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        from timm.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class ViT(Backbone):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        window_block_indexes: List[int] = [],
        residual_block_indexes: List[int] = [],
        use_act_checkpoint: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        num_classes: Optional[int] = None,
        out_features: Optional[List[str]] = None,
        square_pad: int = 0,
        freeze_at: int = 0,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
            square_pad (int): If > 0, require input images to be padded to specific square size.
            freeze_at (int): The number of stages at the beginning to freeze.
        """
        super().__init__()
        self.depth = depth
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        self._out_feature_channels = {}
        self._out_feature_strides = {}

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        self.block_names = []
        for i in range(depth):
            name = "vit" + str(i + 1)

            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            if use_act_checkpoint:
                # TODO: use torch.utils.checkpoint
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)
            self.block_names.append(name)

            self._out_feature_channels[name] = embed_dim
            self._out_feature_strides[name] = patch_size

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(embed_dim, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        self._size_divisibility = patch_size
        self._square_pad = square_pad

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

        self.freeze(freeze_at)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {
            "size_divisibility": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        outputs = {}
        for name, blk in zip(self.block_names, self.blocks):
            x = blk(x)
            if name in self._out_features:
                outputs[name] = x.permute(0, 3, 1, 2)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    def freeze(self, freeze_at: int = 0) -> Type[nn.Module]:
        """
        Freeze the first several stages of the ViT. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "block". All layers are divided into 4 equal stages.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the patch embedding and position embedding.
                `2` means freezing the patch embedding, position embedding and
                one vit stage, etc.

        Returns:
            nn.Module: this ViT itself
        """
        if freeze_at >= 1:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            if self.pos_embed is not None:
                self.pos_embed.requires_grad = False
            # if self.num_classes is not None:
            #     self.class_embed.weight.requires_grad = False
        if freeze_at >= 2:
            freeze_at_block = self.depth // 4 * (freeze_at - 1) if freeze_at != 5 else self.depth
            for idx, blk in enumerate(self.blocks, start=1):
                if freeze_at_block >= idx:
                    blk.freeze()
        return self

    @torch.no_grad()
    def get_img_classes_oh(self, targets: List[Instances]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.num_classes is not None, "self.num_class is None!"
        img_classes = [torch.unique(t.gt_classes, sorted=True).to(torch.int64) for t in targets]
        img_classes_oh = torch.cat(
            [
                gt.new_zeros(
                    (1, self.num_classes), dtype=torch.float
                ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
                for gt in img_classes
            ],
            dim=0,
        )
        return img_classes, img_classes_oh


class SimpleFeaturePyramid(Backbone):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net: Backbone,
        in_feature: str,
        out_channels: int,
        scale_factors: List[float],
        top_block: Optional[Type[nn.Module]] = None,
        norm: str = "LN",
        square_pad: int = 0,
        freeze_at: int = 0,
    ) -> None:
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
            freeze_at (int): If > 0, freeze the FPN stages to 
        """
        super(SimpleFeaturePyramid, self).__init__()
        assert isinstance(net, Backbone)

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

        self.freeze(freeze_at)

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.net(x)
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def freeze(self, freeze_at: int = 0) -> Type[nn.Module]:
        """
        Freeze the first several stages of the FPN layers in ViTDet. Commonly used
        in fine-tuning.

        Each layer that produce the different feature map spatial size is defined
        as one stage.

        Args:
            freeze_at (int): number of stages to freeze.

        Returns:
            nn.Module: this ViTDet itself
        """
        num_stages = len(self.stages) + (self.top_block.num_levels if self.top_block is not None else 0)
        assert freeze_at <= num_stages, "There are all {} stages in FPN, but freeze_at={}.".format(
            num_stages, freeze_at
        )
        if freeze_at >= 1:
            for idx, stage in enumerate(self.stages, start=1):
                if min(freeze_at, len(self.stages)) >= idx:
                    for p in stage.parameters():
                        p.requires_grad = False
                    FrozenBatchNorm2d.convert_frozen_batchnorm(stage)
            if freeze_at > len(self.stages):
                self.top_block.freeze(freeze_at - len(self.stages))
        return self


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


@BACKBONE_REGISTRY.register()
def build_vit_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a ViT instance from config.

    Args:
        cfg (CfgNode): a detectron2 CfgNode.
        input_shape (ShapeSpec): The shape of the input tensor.

    Returns:
        ViT (ViT): backbone module, must be a subclass of :class:`Backbone`.
    """
    # fmt: off
    num_classes            = cfg.MODEL.BACKBONE.NUM_CLASSES
    freeze_at              = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features           = cfg.MODEL.VIT.OUT_FEATURES
    img_size               = cfg.MODEL.VIT.IMG_SIZE
    patch_size             = cfg.MODEL.VIT.PATCH_SIZE
    embed_dim              = cfg.MODEL.VIT.EMBED_DIM
    depth                  = cfg.MODEL.VIT.DEPTH
    num_heads              = cfg.MODEL.VIT.NUM_HEADS
    mlp_ratio              = cfg.MODEL.VIT.MLP_RATIO
    qkv_bias               = cfg.MODEL.VIT.QKV_BIAS
    drop_path_rate         = cfg.MODEL.VIT.DROP_PATH_RATE
    use_abs_pos            = cfg.MODEL.VIT.USE_ABS_POS
    use_rel_pos            = cfg.MODEL.VIT.USE_REL_POS
    rel_pos_zero_init      = cfg.MODEL.VIT.REL_POS_ZERO_INIT
    window_size            = cfg.MODEL.VIT.WINDOW_SIZE
    global_block_indexes   = cfg.MODEL.VIT.GLOBAL_BLOCK_INDEXES
    residual_block_indexes = cfg.MODEL.VIT.RESIDUAL_BLOCK_INDEXES
    use_act_checkpoint     = cfg.MODEL.VIT.USE_ACT_CHECKPOINT
    pretrain_img_size      = cfg.MODEL.VIT.PRETRAIN_IMG_SIZE
    pretrain_use_cls_token = cfg.MODEL.VIT.PRETRAIN_USE_CLS_TOKEN
    square_pad             = cfg.MODEL.VIT.SQUARE_PAD
    # fmt: on

    return ViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=input_shape.channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=use_abs_pos,
        use_rel_pos=use_rel_pos,
        rel_pos_zero_init=rel_pos_zero_init,
        window_size=window_size,
        window_block_indexes=[i for i in range(depth) if i not in global_block_indexes],
        residual_block_indexes=residual_block_indexes,
        use_act_checkpoint=use_act_checkpoint,
        pretrain_img_size=pretrain_img_size,
        pretrain_use_cls_token=pretrain_use_cls_token,
        num_classes=num_classes,
        out_features=out_features,
        square_pad=square_pad,
        freeze_at=freeze_at,
    )


@BACKBONE_REGISTRY.register()
def build_vitdet_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a ViTDet instance from config.

    Args:
        cfg (CfgNode): a detectron2 CfgNode.
        input_shape (ShapeSpec): The shape of the input tensor.

    Returns:
        ViTDet (SimpleFeaturePyramid): backbone module, must be a subclass of :class:`Backbone`.
    """
    net = build_vit_backbone(cfg, input_shape)

    # fmt: off
    freeze_fpn_at   = cfg.MODEL.BACKBONE.FREEZE_FPN_AT
    in_features     = cfg.MODEL.FPN.IN_FEATURES
    out_channels    = cfg.MODEL.FPN.OUT_CHANNELS
    scale_factors   = cfg.MODEL.FPN.SCALE_FACTORS
    norm            = cfg.MODEL.FPN.NORM
    square_pad      = cfg.MODEL.FPN.SQUARE_PAD
    # fmt: on

    return SimpleFeaturePyramid(
        net=net,
        in_feature=in_features[-1],
        out_channels=out_channels,
        scale_factors=scale_factors,
        top_block=None,
        norm=norm,
        square_pad=square_pad,
        freeze_at=freeze_fpn_at,
    )

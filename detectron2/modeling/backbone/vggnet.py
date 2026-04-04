# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "VGGNetBlockBase",
    "PlainBlock",
    "DeformPlainBlockWS",
    "VGGNet",
    "make_vggnet_stage",
    "build_vggnet_backbone",
]


class PlainBlock(CNNBlockBase):
    """
    The standard bottleneck plain block used by VGG-11-WS, 13-WS, 16-WS and 19-WS
    defined in :paper:`ResNet-WS`.  It contains num_convs conv layers with kernels
    3x3, and a pooling layer if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        dilation=1,
        has_pool=True,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        self.num_convs = num_convs
        self.has_pool = has_pool

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=True,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        if num_convs > 1:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=num_groups,
                dilation=dilation,
                norm=get_norm(norm, bottleneck_channels),
            )
        else:
            self.conv2 = None

        if num_convs > 2:
            self.conv3 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=num_groups,
                dilation=dilation,
                norm=get_norm(norm, bottleneck_channels),
            )
        else:
            self.conv3 = None

        if num_convs > 3:
            self.conv4 = Conv2d(
                bottleneck_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=num_groups,
                dilation=dilation,
                norm=get_norm(norm, bottleneck_channels),
            )
        else:
            self.conv4 = None

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

        if has_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)

        # Zero-initialize the last normalization in each vgg branch,
        # so that at the beginning, the vgg branch starts with zeros,
        # and each vgg block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each vgg block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.num_convs > 1:
            out = self.conv2(out)
            out = F.relu_(out)

        if self.num_convs > 2:
            out = self.conv3(out)
            out = F.relu_(out)

        if self.num_convs > 3:
            out = self.conv4(out)
            out = F.relu_(out)

        if self.has_pool:
            out = self.pool(out)
        return out


class DeformPlainBlockWS(CNNBlockBase):
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
        has_pool=False,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.num_convs = num_convs
        self.deform_modulated = deform_modulated
        self.has_pool = has_pool

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=True,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        if num_convs > 1:
            if deform_modulated:
                deform_conv_op = ModulatedDeformConv
                # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
                offset_channels = 27
            else:
                deform_conv_op = DeformConv
                offset_channels = 18

            self.conv2_offset = Conv2d(
                bottleneck_channels,
                offset_channels * deform_num_groups,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                dilation=dilation,
            )
            self.conv2 = deform_conv_op(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deform_num_groups,
                norm=get_norm(norm, bottleneck_channels),
            )
        else:
            self.conv2 = None

        if num_convs > 2:
            self.conv3 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=num_groups,
                dilation=dilation,
                norm=get_norm(norm, bottleneck_channels),
            )
        else:
            self.conv3 = None

        if num_convs > 3:
            self.conv4 = Conv2d(
                bottleneck_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1 * dilation,
                bias=True,
                groups=num_groups,
                dilation=dilation,
                norm=get_norm(norm, bottleneck_channels),
            )
        else:
            self.conv4 = None

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

        if num_convs > 1:
            nn.init.constant_(self.conv2_offset.weight, 0)
            nn.init.constant_(self.conv2_offset.bias, 0)

        if has_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.num_convs > 1:
            if self.deform_modulated:
                offset_mask = self.conv2_offset(out)
                offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
                offset = torch.cat((offset_x, offset_y), dim=1)
                mask = mask.sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = F.relu_(out)

        if self.num_convs > 2:
            out = self.conv3(out)
            out = F.relu_(out)

        if self.num_convs > 3:
            out = self.conv4(out)
            out = F.relu_(out)

        if self.has_pool:
            out = self.pool(out)
        return out


class VGGNet(Backbone):
    """
    Implement :paper:`ResNet-WS`.
    """

    def __init__(self, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "vgg1", "vgg2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.num_classes = num_classes

        current_stride = 1
        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max([{
                "vgg1": 1, "vgg2": 2, "vgg3": 3, "vgg4": 4, "vgg5": 5
            }.get(f, 0) for f in out_features])
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "vgg" + str(i + 1)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"VGGNetWS takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the VGGNetWS. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.

        Returns:
            nn.Module: this VGGNetWS itself
        """
        for idx, stage in enumerate(self.stages):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_convs, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks that forms one VGGNetWS stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_convs (int): number of convolutions in this stage.
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. The same argument is passed to every block
                in the stage.

        Returns:
            list[CNNBlockBase]: a list of block module.

        Examples:
        ::
            stage = VGGNetWS.make_stage(
                BottleneckBlock, num_convs=3, in_channels=16, out_channels=64,
                bottleneck_channels=64, num_groups=1, stride=2, dilations=1
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride`` should all be 1.
        """
        return [block_class(
            in_channels=in_channels, out_channels=out_channels, num_convs=num_convs, **kwargs
        )]

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of VGGNetWS stages from pre-defined depth (one of 11, 13, 16, 19).
        If it doesn't create the VGGNetWS variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.

        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.

        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_convs_per_stage = {
            11: [1, 1, 2, 2, 2],
            13: [2, 2, 2, 2, 2],
            16: [2, 2, 3, 3, 3],
            19: [2, 2, 4, 4, 4],
        }[depth]
        if block_class is None:
            block_class = PlainBlock
        in_channels = [3, 64, 128, 256, 512]
        out_channels = [64, 128, 256, 512, 512]
        ret = []
        for (n, s, i, o) in zip(num_convs_per_stage, [2, 2, 2, 1, 1], in_channels, out_channels):
            ret.append(
                VGGNet.make_stage(
                    block_class=block_class,
                    num_convs=n,
                    in_channels=i,
                    out_channels=o,
                    bottleneck_channels=o,
                    stride=s,
                    dilation=1 if s == 2 else 2,
                    has_pool=True if s == 2 else False,
                    **kwargs,
                )
            )
        return ret


VGGNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


def make_vggnet_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    """
    return VGGNet.make_stage(*args, **kwargs)


@BACKBONE_REGISTRY.register()
def build_vggnet_backbone(cfg, input_shape):
    """
    Create a VGGNet-WS instance from config.

    Returns:
        VGGNetWS: a :class:`VGGNet-WS` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.VGGNETS.NORM

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.VGGNETS.OUT_FEATURES
    depth               = cfg.MODEL.VGGNETS.DEPTH
    num_groups          = cfg.MODEL.VGGNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.VGGNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = input_shape.channels
    out_channels        = cfg.MODEL.VGGNETS.VGG1_OUT_CHANNELS
    vgg5_dilation       = cfg.MODEL.VGGNETS.VGG5_DILATION
    deform_on_per_stage = cfg.MODEL.VGGNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.VGGNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.VGGNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert out_channels == 64, "Must set MODEL.VGGNETSWS.VGG1_OUT_CHANNELS = 64 for VGGNetWS"
    assert vgg5_dilation in {1, 2}, "vgg5_dilation cannot be {}.".format(vgg5_dilation)

    num_convs_per_stage = {
        11: [1, 1, 2, 2, 2],
        13: [2, 2, 2, 2, 2],
        16: [2, 2, 3, 3, 3],
        19: [2, 2, 4, 4, 4],
    }[depth]

    stages = []

    for idx, stage_idx in enumerate(range(1, 6)):
        # vgg5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = vgg5_dilation if stage_idx == 4 or stage_idx == 5 else 1
        last_stride = 1 if (stage_idx == 4 and dilation == 2) or stage_idx == 5 else 2
        has_pool = False if stage_idx == 5 else True
        stage_kargs = {
            "num_convs": num_convs_per_stage[idx],
            "stride": last_stride,
            "has_pool": has_pool,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformPlainBlockWS
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = PlainBlock
        blocks = VGGNet.make_stage(**stage_kargs)
        in_channels = out_channels
        if stage_idx < 4:
            out_channels *= 2
            bottleneck_channels *= 2
        stages.append(blocks)
    return VGGNet(stages, out_features=out_features, freeze_at=freeze_at)

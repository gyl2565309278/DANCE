# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_wsod_cfg(cfg):
    """
    Add config for WSOD nets.
    """

    # -----------------------------------------------------------------------------
    # Config definition
    # -----------------------------------------------------------------------------
    _C = cfg
    _C.EPSILON = 1e-6


    # ---------------------------------------------------------------------------- #
    # VGG options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.VGGNETS = CN()

    _C.MODEL.VGGNETS.DEPTH = 16
    _C.MODEL.VGGNETS.OUT_FEATURES = ["vgg4"]  # vgg4 for C4 backbone, vgg2..5 for FPN backbone

    # Number of groups to use; 1 ==> VGGNet; > 1 ==> VGGNeXt
    _C.MODEL.VGGNETS.NUM_GROUPS = 1

    # Options: "", "FrozenBN", "GN", "SyncBN", "BN"
    _C.MODEL.VGGNETS.NORM = ""

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.VGGNETS.WIDTH_PER_GROUP = 64

    # Apply dilation in stage "vgg5"
    _C.MODEL.VGGNETS.VGG5_DILATION = 1

    # Output width of vgg1. Scaling this parameters will scale the width of all 3x3 convs in VGGNet
    # This needs to be set to 64
    _C.MODEL.VGGNETS.VGG1_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on vgg1, vgg2, vgg3, vgg4, vgg5
    _C.MODEL.VGGNETS.DEFORM_ON_PER_STAGE = [False, False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.VGGNETS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.VGGNETS.DEFORM_NUM_GROUPS = 1


    # ---------------------------------------------------------------------------- #
    # ResNet_ws options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RESNETSWS = CN()

    _C.MODEL.RESNETSWS.DEPTH = 50
    _C.MODEL.RESNETSWS.OUT_FEATURES = ["res5"]

    # Number of groups to use; 1 ==> ResNet-WS; > 1 ==> ResNeXt-WS
    _C.MODEL.RESNETSWS.NUM_GROUPS = 1

    # Options: "", "FrozenBN", "GN", "SyncBN", "BN"
    _C.MODEL.RESNETSWS.NORM = "FrozenBN"

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.RESNETSWS.WIDTH_PER_GROUP = 64

    # Apply dilation in stage "res5"
    _C.MODEL.RESNETSWS.RES5_DILATION = 2

    # Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet-WS
    # For R18 and R34, this needs to be set to 64
    _C.MODEL.RESNETSWS.RES2_OUT_CHANNELS = 256
    _C.MODEL.RESNETSWS.STEM_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    _C.MODEL.RESNETSWS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.RESNETSWS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.RESNETSWS.DEFORM_NUM_GROUPS = 1


    # ---------------------------------------------------------------------------- #
    # ROI Box Head options
    # ---------------------------------------------------------------------------- #
    # Choose whether to use proposal cluster loss
    _C.MODEL.ROI_BOX_HEAD.USE_PCL_LOSS = False


    # ---------------------------------------------------------------------------- #
    # WSOD options
    # ---------------------------------------------------------------------------- #
    _C.WSOD = CN()

    _C.WSOD.REFINE_K = 3
    _C.WSOD.REFINE_REG = [False, False, False]


    # ---------------------------------------------------------------------------- #
    # PCL options
    # ---------------------------------------------------------------------------- #
    _C.WSOD.PCL = CN()
    # If the remaining verticles is smaller than MIN_REMAIN_COUNT, stop filter pseudo gt
    _C.WSOD.PCL.MIN_REMAIN_COUNT = 5
    # The maximum number of proposal clusters per class
    _C.WSOD.PCL.MAX_NUM_PC = 5

    _C.WSOD.PCL.KMEANS = CN()
    _C.WSOD.PCL.KMEANS.NUM_CLUSTERS = 3
    _C.WSOD.PCL.KMEANS.SEED = 2

    _C.WSOD.PCL.GRAPH = CN()
    _C.WSOD.PCL.GRAPH.IOU_THRESHOLD = 0.4


    # ---------------------------------------------------------------------------- #
    # MIST options
    # ---------------------------------------------------------------------------- #
    _C.WSOD.MIST = CN()
    # Top p percent of the region proposals to pick
    _C.WSOD.MIST.TOP_SCORE_PERCENT = 0.15

    _C.WSOD.MIST.GRAPH = CN()
    # IoU threshold for NMS
    _C.WSOD.MIST.GRAPH.IOU_THRESHOLD = 0.2


    # ---------------------------------------------------------------------------- #
    # SLV options
    # ---------------------------------------------------------------------------- #
    _C.WSOD.SLV = CN()
    _C.WSOD.SLV.SCORE_THRESH = 0.001

    _C.WSOD.SLV.HEATMAP = CN()
    _C.WSOD.SLV.HEATMAP.BINARY_SCORE_THRESH = 0.5


    # ---------------------------------------------------------------------------- #
    # TEST options
    # ---------------------------------------------------------------------------- #
    _C.TEST.EVAL_TRAIN = False
    # Options: "GeneralizedRCNNWSWithTTAAverage", "GeneralizedRCNNWSWithTTAUnion"
    _C.TEST.AUG.NAME = "GeneralizedRCNNWSWithTTAAverage"

    _C.TEST.VIS = CN({"ENABLED": False})
    _C.TEST.VIS.SCORE_THRESH = 0.5

# Copyright (c) Facebook, Inc. and its affiliates.
from annotator.oneformer.detectron2.layers import ShapeSpec

from .anchor_generator import ANCHOR_GENERATOR_REGISTRY, build_anchor_generator
from .backbone import (
    BACKBONE_REGISTRY,
    FPN,
    Backbone,
    MViT,
    ResNet,
    ResNetBlockBase,
    SimpleFeaturePyramid,
    SwinTransformer,
    ViT,
    build_backbone,
    build_resnet_backbone,
    get_vit_lr_decay_rate,
    make_stage,
)
from .meta_arch import (
    FCOS,
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    GeneralizedRCNN,
    PanopticFPN,
    ProposalNetwork,
    RetinaNet,
    SemanticSegmentor,
    build_model,
    build_sem_seg_head,
)
from .mmdet_wrapper import MMDetBackbone, MMDetDetector
from .postprocessing import detector_postprocess
from .proposal_generator import (
    PROPOSAL_GENERATOR_REGISTRY,
    RPN_HEAD_REGISTRY,
    build_proposal_generator,
    build_rpn_head,
)
from .roi_heads import (
    ROI_BOX_HEAD_REGISTRY,
    ROI_HEADS_REGISTRY,
    ROI_KEYPOINT_HEAD_REGISTRY,
    ROI_MASK_HEAD_REGISTRY,
    BaseKeypointRCNNHead,
    BaseMaskRCNNHead,
    FastRCNNOutputLayers,
    ROIHeads,
    StandardROIHeads,
    build_box_head,
    build_keypoint_head,
    build_mask_head,
    build_roi_heads,
)
from .test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


from annotator.oneformer.detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata

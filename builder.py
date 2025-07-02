import torch
import torch.nn as nn
import torch.nn.functional as F

from init_func import init_weight
from load_utils import load_pretrain
from functools import partial
import numpy as np
from logger import get_logger
import warnings

# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
# from mmcv.utils import Registry

# MODELS = Registry('models', parent=MMCV_MODELS)
# ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

# BACKBONES = MODELS
# NECKS = MODELS
# HEADS = MODELS
# LOSSES = MODELS
# SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn("train_cfg and test_cfg is deprecated, please specify them in model", UserWarning)
    assert cfg.get("train_cfg") is None or train_cfg is None, (
        "train_cfg specified in both outer field and model field "
    )
    assert cfg.get("test_cfg") is None or test_cfg is None, "test_cfg specified in both outer field and model field "
    return SEGMENTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


logger = get_logger()


class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim, num_classes)
 
    def forward(self, x):
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    ):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg

        if cfg.backbone == "DFormer-Large":
            from encoders.DFormer import DFormer_Large as backbone

            self.channels = [96, 192, 288, 576]
        elif cfg.backbone == "DFormer-Base":
            from encoders.DFormer import DFormer_Base as backbone

            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Small":
            from encoders.DFormer import DFormer_Small as backbone

            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Tiny":
            from encoders.DFormer import DFormer_Tiny as backbone

            self.channels = [32, 64, 128, 256]

        elif cfg.backbone == "DFormerv2_L":
            from encoders.DFormerv2 import DFormerv2_L as backbone

            self.channels = [112, 224, 448, 640]
        elif cfg.backbone == "DFormerv2_B":
            from encoders.DFormerv2 import DFormerv2_B as backbone

            self.channels = [80, 160, 320, 512]
        elif cfg.backbone == "DFormerv2_S":
            from encoders.DFormerv2 import DFormerv2_S as backbone

            self.channels = [64, 128, 256, 512]
        else:
            raise NotImplementedError

        if syncbn:
            norm_cfg = dict(type="SyncBN", requires_grad=True)
        else:
            norm_cfg = dict(type="BN", requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        # from mmseg.models.decode_heads.ham_head import LightHamHead as DecoderHead
        self.decode_head = ClassificationHead(dim=256,num_classes=cfg.num_classes)

        self.criterion = criterion

        self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )

    def encode_decode(self, rgb):
        x = self.backbone(rgb)
        out = self.decode_head.forward(x[-1])
        return out

    @torch.no_grad()
    def evaluate(self,rgb):
        out = self.encode_decode(rgb)
        prob = nn.Softmax(dim=-1)(out)
        index = np.argmax(prob.cpu().detach().numpy(),axis=1)
        return index

    def forward(self, rgb, label=None):
        out = self.encode_decode(rgb)
        if label is not None:
            loss = self.criterion(out, label.long()).mean()
            return loss
        return out

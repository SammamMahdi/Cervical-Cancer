"""Published cervical-cytology SOTA architectures for the RQ6 comparison (Ablation §3).

Reference reimplementations that follow each method's published architecture description (NOT official
pretrained weights), wrapped to expose the SAME interface as models.backbones.Classifier
(features / pool / head / features_vec / forward / set_backbone_trainable / last_conv_module) so they
train, calibrate, profile and Grad-CAM through the identical leakage-safe protocol as every other model.

  * CCanNet     -- compact VGG-style CNN for cervical cancer cytology classification.
  * CerviXpert  -- computationally-efficient multi-structure CNN (depthwise-separable blocks).
  * MaxCerVixT  -- MaxViT-based vision transformer for cervical cytology (built on torchvision maxvit_t).

Each is registered so models.backbones.build("ccannet"|"cervixpert"|"maxcervixt", ...) returns it.
"""
import torch
import torch.nn as nn

import config as C


# --------------------------------------------------------------------------- #
#  Common wrapper: trunk (N,C,H,W) -> pool -> head, matching Classifier's API
# --------------------------------------------------------------------------- #
class SOTAClassifier(nn.Module):
    def __init__(self, name, trunk, feature_dim, num_classes=C.NUM_CLASSES, hidden=512,
                 dropout=0.3, transformer=False):
        super().__init__()
        self.backbone_name = name
        self.feature_dim = feature_dim
        self._transformer = transformer
        self.features = trunk
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_dim), nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def features_vec(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x):
        return self.head(self.features_vec(x))

    def set_backbone_trainable(self, flag):
        for p in self.features.parameters():
            p.requires_grad = bool(flag)
        return self

    @property
    def last_conv_module(self):
        if self._transformer:
            return self.features                    # trunk output is the Grad-CAM feature map
        last = None
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last


# --------------------------------------------------------------------------- #
#  CCanNet -- VGG-style conv stack
# --------------------------------------------------------------------------- #
def _conv_block(cin, cout, n=2):
    layers = []
    for k in range(n):
        layers += [nn.Conv2d(cin if k == 0 else cout, cout, 3, 1, 1, bias=False),
                   nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


def ccannet_trunk():
    return nn.Sequential(
        _conv_block(3, 32, 2), _conv_block(32, 64, 2),
        _conv_block(64, 128, 3), _conv_block(128, 256, 3),
    )                                                # -> (N,256,H/16,W/16)


# --------------------------------------------------------------------------- #
#  CerviXpert -- depthwise-separable, computationally efficient
# --------------------------------------------------------------------------- #
def _dwsep(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, 1, 0, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def cervixpert_trunk():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        _dwsep(32, 64, 2), _dwsep(64, 128, 1), _dwsep(128, 128, 2),
        _dwsep(128, 256, 1), _dwsep(256, 256, 2),
    )                                                # -> (N,256,...)


# --------------------------------------------------------------------------- #
#  MaxCerVixT -- MaxViT-based transformer trunk (reuse torchvision maxvit_t)
# --------------------------------------------------------------------------- #
def maxcervixt_trunk(pretrained=True):
    from torchvision import models as tvm
    from models.backbones import _MaxViTTrunk        # ModuleList-aware trunk (stem -> each block)
    w = tvm.MaxVit_T_Weights.DEFAULT if pretrained else None
    net = tvm.maxvit_t(weights=w)
    feat = net.classifier[3].in_features if hasattr(net.classifier[3], "in_features") else 512
    return _MaxViTTrunk(net.stem, net.blocks), feat


# --------------------------------------------------------------------------- #
#  Registry + factory
# --------------------------------------------------------------------------- #
SOTA_MODELS = ("ccannet", "cervixpert", "maxcervixt")


def is_sota(name):
    return name in SOTA_MODELS


def build_sota(name, num_classes=C.NUM_CLASSES, hidden=512, dropout=0.3, pretrained=True):
    if name == "ccannet":
        return SOTAClassifier(name, ccannet_trunk(), 256, num_classes, hidden, dropout)
    if name == "cervixpert":
        return SOTAClassifier(name, cervixpert_trunk(), 256, num_classes, hidden, dropout)
    if name == "maxcervixt":
        trunk, feat = maxcervixt_trunk(pretrained=pretrained)
        return SOTAClassifier(name, trunk, feat, num_classes, hidden, dropout, transformer=True)
    raise ValueError(f"unknown SOTA model {name!r}; choose from {SOTA_MODELS}")


if __name__ == "__main__":
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    for n in SOTA_MODELS:
        m = build_sota(n, pretrained=False).eval()
        with torch.no_grad():
            out = m(x); feat = m.features_vec(x)
        tot = sum(p.numel() for p in m.parameters())
        print(f"{n:11s} logits={tuple(out.shape)} feat={tuple(feat.shape)} "
              f"params={tot/1e6:.2f}M last_conv={type(m.last_conv_module).__name__}")

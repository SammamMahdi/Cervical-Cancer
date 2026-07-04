"""Configurable torchvision classifier backbones (A8 ablation + teacher).

A single `Classifier` wraps any of the supported torchvision nets behind a uniform interface so
the ablation (A8), the EfficientNet-B4 teacher (U2) and the EfficientNet-B0 student all share one
code path:

    trunk (conv features)  ->  global avg pool  ->  custom MLP head  ->  logits

The custom head (BatchNorm1d -> Linear -> ReLU -> Dropout -> Linear) replaces each net's native
1000-class classifier with our 5-class cytology head. Torchvision exposes two different layouts:

  * efficientnet_* / mobilenet_v3_large : ``.features`` (conv trunk) + ``.avgpool`` + ``.classifier``
  * resnet50                            : conv1..layer4 + ``.avgpool`` + ``.fc``

We normalise both to ``self.features`` (an nn.Module producing (N, feat, H, W)) so downstream code
(Grad-CAM via ``last_conv_module``, feature extraction via ``features_vec``, stage-wise freezing via
``set_backbone_trainable``) is backbone-agnostic.

Weights come exclusively from torchvision (download.pytorch.org); timm / HF Hub are blocked.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B4_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    MaxVit_T_Weights,
    ShuffleNet_V2_X1_0_Weights,
    DenseNet121_Weights,
)

import config as C


class _SwinNCHW(nn.Module):
    """Adapter: Swin's `.features` emit (N,H,W,C); apply final norm and permute to (N,C,H,W) so the
    uniform AdaptiveAvgPool2d + conv-style head/Grad-CAM path works."""
    def __init__(self, features, norm):
        super().__init__(); self.features = features; self.norm = norm
    def forward(self, x):
        x = self.features(x)        # (N,H,W,C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()   # (N,C,H,W)


class _MaxViTTrunk(nn.Module):
    """MaxViT `.blocks` is a ModuleList; run stem then each block -> (N,C,H,W)."""
    def __init__(self, stem, blocks):
        super().__init__(); self.stem = stem; self.blocks = blocks
    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return x

# ---------------------------------------------------------------------------
# registry: name -> (torchvision builder, default weights enum, feature_dim)
# feature_dim = channels of the trunk output (verified by forward pass).
# ---------------------------------------------------------------------------
_BACKBONES = {
    "efficientnet_b0": (tvm.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_b2": (tvm.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, 1408),
    "efficientnet_b4": (tvm.efficientnet_b4, EfficientNet_B4_Weights.DEFAULT, 1792),  # teacher
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT, 960),
    "resnet50": (tvm.resnet50, ResNet50_Weights.DEFAULT, 2048),
    "swin_t": (tvm.swin_t, Swin_T_Weights.DEFAULT, 768),       # transformer (model comparison §3)
    "maxvit_t": (tvm.maxvit_t, MaxVit_T_Weights.DEFAULT, 512),  # hybrid transformer
    "shufflenet_v2_x1_0": (tvm.shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.DEFAULT, 1024),  # edge (§3)
    "densenet121": (tvm.densenet121, DenseNet121_Weights.DEFAULT, 1024),                       # heavier CNN (§3)
}


def available_backbones():
    """Return the list of registered backbone names."""
    return list(_BACKBONES.keys())


class Classifier(nn.Module):
    """Uniform backbone + custom classification head.

    Args:
        backbone:    registry key (see ``available_backbones()``).
        num_classes: output logits (default 5, NORMAL-first SIPaKMeD).
        hidden:      width of the head's hidden Linear.
        dropout:     dropout in the head (between hidden ReLU and final Linear).
        pretrained:  load ImageNet weights from torchvision if True, else random init.

    Layout:
        self.features : conv trunk -> (N, feat, H, W)
        self.pool     : AdaptiveAvgPool2d(1)
        self.head     : BatchNorm1d(feat) -> Linear(feat,hidden) -> ReLU -> Dropout -> Linear(hidden,C)
    """

    def __init__(self, backbone, num_classes=5, hidden=512, dropout=0.3, pretrained=True):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"unknown backbone {backbone!r}; choose from {available_backbones()}"
            )
        self.backbone_name = backbone
        builder, weights_enum, feat = _BACKBONES[backbone]
        self.feature_dim = feat

        net = builder(weights=weights_enum if pretrained else None)

        # --- normalise the conv trunk across the two torchvision layouts ---
        if backbone.startswith("efficientnet") or backbone == "mobilenet_v3_large":
            # ``.features`` is already the conv trunk producing (N, feat, H, W).
            self.features = net.features
        elif backbone == "resnet50":
            # rebuild the conv stem .. layer4 as one Sequential (drop avgpool + fc).
            self.features = nn.Sequential(
                OrderedDict(
                    conv1=net.conv1,
                    bn1=net.bn1,
                    relu=net.relu,
                    maxpool=net.maxpool,
                    layer1=net.layer1,
                    layer2=net.layer2,
                    layer3=net.layer3,
                    layer4=net.layer4,
                )
            )
        elif backbone == "swin_t":
            self.features = _SwinNCHW(net.features, net.norm)          # (N,C,H,W) after permute
        elif backbone == "maxvit_t":
            self.features = _MaxViTTrunk(net.stem, net.blocks)        # (N,C,H,W)
        elif backbone == "densenet121":
            self.features = nn.Sequential(net.features, nn.ReLU(inplace=True))   # densenet: trunk + final ReLU
        elif backbone == "shufflenet_v2_x1_0":
            self.features = nn.Sequential(OrderedDict(
                conv1=net.conv1, maxpool=net.maxpool,
                stage2=net.stage2, stage3=net.stage3, stage4=net.stage4, conv5=net.conv5))
        else:  # pragma: no cover - guarded by registry check above
            raise ValueError(f"no trunk rule for {backbone!r}")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat),
            nn.Linear(feat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    # ------------------------------------------------------------------ API
    def features_vec(self, x):
        """Pooled feature embedding before the head: returns (N, feat)."""
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        """Return class logits (N, num_classes)."""
        return self.head(self.features_vec(x))

    def set_backbone_trainable(self, flag):
        """Freeze (flag=False) or unfreeze (flag=True) the conv trunk; head stays trainable.

        Used for stage-1 (frozen) -> stage-2 (fine-tune) teacher training (U2).
        """
        for p in self.features.parameters():
            p.requires_grad = bool(flag)
        return self

    @property
    def last_conv_module(self):
        """Last spatial conv module of the trunk, the Grad-CAM target layer.

        For efficientnet / mobilenet_v3 this is the final ``Conv2dNormActivation`` block (whose
        output is the last conv feature map); for resnet50 it is the deepest ``nn.Conv2d``.
        """
        if self.backbone_name in ("swin_t", "maxvit_t"):
            return self.features          # trunk output (N,C,H,W) is the Grad-CAM feature map
        if self.backbone_name in ("resnet50", "densenet121", "shufflenet_v2_x1_0"):
            last = None
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    last = m
            return last
        # efficientnet / mobilenet_v3: features[-1] is the top Conv2dNormActivation block.
        return self.features[-1]


# --------------------------------------------------------------------- factory
def build(name, num_classes=C.NUM_CLASSES, hidden=512, dropout=0.3, pretrained=True):
    """Factory: construct a :class:`Classifier` for the named backbone, or a published-SOTA model
    (CCanNet / CerviXpert / MaxCerVixT) when ``name`` is in models.sota.SOTA_MODELS (RQ6)."""
    from models.sota import is_sota, build_sota
    if is_sota(name):
        return build_sota(name, num_classes=num_classes, hidden=hidden, dropout=dropout, pretrained=pretrained)
    return Classifier(
        name, num_classes=num_classes, hidden=hidden, dropout=dropout, pretrained=pretrained
    )


def count_params(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_teacher(num_classes=C.NUM_CLASSES, pretrained=True):
    """EfficientNet-B4 teacher per C.TEACHER (U2)."""
    t = C.TEACHER
    return build(
        t["backbone"],
        num_classes=num_classes,
        hidden=t["hidden"],
        dropout=t["dropout"],
        pretrained=pretrained,
    )


def build_student(num_classes=C.NUM_CLASSES, pretrained=True):
    """EfficientNet-B0 student per C.STUDENT (U2)."""
    s = C.STUDENT
    return build(
        s["backbone"],
        num_classes=num_classes,
        hidden=s["hidden"],
        dropout=s["dropout"],
        pretrained=pretrained,
    )


if __name__ == "__main__":
    # smoke: build each ablation backbone + the B4 teacher, report params, confirm (N,5) output.
    torch.manual_seed(C.SEED)
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    names = list(dict.fromkeys(C.ABLATION_BACKBONES + ["efficientnet_b4"]))
    print(f"{'backbone':22s} {'params(M)':>10s} {'trainable(M)':>13s}  output")
    for name in names:
        net = build(name, pretrained=False).eval()  # pretrained=False -> no download for smoke
        total, trainable = count_params(net)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (2, C.NUM_CLASSES), f"{name}: bad output {tuple(out.shape)}"
        feat = net.features_vec(x).shape
        lc = type(net.last_conv_module).__name__
        print(
            f"{name:22s} {total/1e6:10.2f} {trainable/1e6:13.2f}  "
            f"{tuple(out.shape)} feat={tuple(feat)} last_conv={lc}"
        )
    print("OK")

"""Teacher (EfficientNet-B4) and Student (EfficientNet-B0) with the legacy head.

Backbones via **torchvision** (ImageNet weights from download.pytorch.org — reachable here;
the HuggingFace Hub that `timm` uses is blocked in this environment). Legacy head (from the
notebook): GlobalAvgPool -> BatchNorm -> Dense(512, relu) -> Dropout -> Dense(num_classes).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

# backbone name -> (constructor, weights enum, feature dim after avgpool)
_EFF = {
    "efficientnet_b4": (tvm.efficientnet_b4, tvm.EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}


class Head(nn.Module):
    def __init__(self, in_features, hidden=512, num_classes=5, dropout=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        return self.out(x)            # logits


class EffNetClassifier(nn.Module):
    """torchvision EfficientNet backbone (global-avg-pooled features) + legacy Head -> logits."""
    def __init__(self, backbone="efficientnet_b4", num_classes=5, hidden=512, dropout=0.5, pretrained=True):
        super().__init__()
        ctor, weights, feat = _EFF[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.features = net.features          # conv stack
        self.pool = net.avgpool               # AdaptiveAvgPool2d(1)
        self.head = Head(feat, hidden=hidden, num_classes=num_classes, dropout=dropout)

    def features_vec(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.head(self.features_vec(x))     # logits

    def set_backbone_trainable(self, flag: bool):
        for p in self.features.parameters():
            p.requires_grad = flag

    # alias used by Grad-CAM (last conv block lives in self.features)
    @property
    def last_conv_module(self):
        return self.features[-1]


def build_teacher(cfg, num_classes, pretrained=True):
    return EffNetClassifier(cfg["backbone"], num_classes, cfg["hidden"], cfg["dropout"], pretrained)


def build_student(cfg, num_classes, pretrained=True):
    return EffNetClassifier(cfg["backbone"], num_classes, cfg["hidden"], cfg["dropout"], pretrained)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

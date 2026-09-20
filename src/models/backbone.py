"""CNN feature extractors shared by the hybrid, baseline and surrogate models.

Two backbones are available:

* ``ResNet18Backbone`` -- a torchvision ResNet-18 (the project's main
  architecture). It exposes the output of every residual **stage**
  (``layer1``..``layer4``), which is what the layer-by-layer LNN explainer needs
  to read concepts at increasing depth.
* ``CNNBackbone`` -- a compact 3-stage net, kept as a fast CPU-only fallback and
  the default in unit tests (no weight download).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNBackbone(nn.Module):
    """3-stage conv net -> flat feature embedding.

    Input:  [B, 3, 48, 48]
    Output: [B, feature_dim]
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32),    # 48 -> 24
            _block(32, 64),   # 24 -> 12
            _block(64, 128),  # 12 -> 6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B, 128, 1, 1]
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.proj(x)


class ResNet18Backbone(nn.Module):
    """torchvision ResNet-18 that also exposes per-stage features.

    ``forward`` returns the final 512-d embedding (drop-in for ``CNNBackbone``).
    ``forward_stages`` returns the global-average-pooled feature vector after each
    residual stage -- the input to the layer-by-layer concept probes.

    Args:
        pretrained: load ImageNet weights (recommended; fast convergence).
        small_input: replace the aggressive 7x7/maxpool stem with a 3x3 stride-1
            stem (CIFAR-style) to preserve resolution on 48x48 signs. This
            reinitialises the stem, so it is best combined with more training.
    """

    STAGE_DIMS = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
    STAGES = ["layer1", "layer2", "layer3", "layer4"]

    def __init__(self, pretrained: bool = True, small_input: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        if small_input:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
            net.maxpool = nn.Identity()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512

    def forward_stages(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        def pool(t: torch.Tensor) -> torch.Tensor:
            return self.avgpool(t).flatten(1)

        return {"layer1": pool(f1), "layer2": pool(f2),
                "layer3": pool(f3), "layer4": pool(f4)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(x)["layer4"]  # [B, 512]


def build_backbone(name: str = "simple", feature_dim: int = 256,
                   pretrained: bool = True, small_input: bool = False) -> nn.Module:
    if name == "resnet18":
        return ResNet18Backbone(pretrained=pretrained, small_input=small_input)
    if name == "simple":
        return CNNBackbone(feature_dim=feature_dim)
    raise ValueError(f"unknown backbone {name!r} (use 'resnet18' or 'simple')")

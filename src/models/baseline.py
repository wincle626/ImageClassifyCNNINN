"""Standard CNN classifier baseline (no concepts, no logic).

Uses the *same* backbone as the hybrid model so the comparison isolates the
effect of the concept-bottleneck + LNN reasoning head, not the feature
extractor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "simple",
                 feature_dim: int = 256, pretrained: bool = True,
                 small_input: bool = False):
        super().__init__()
        self.backbone = build_backbone(backbone, feature_dim, pretrained,
                                       small_input)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))  # class logits [B, C]

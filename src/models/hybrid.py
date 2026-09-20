"""The hybrid CNN + LNN model (the neuro-symbolic pipeline).

    image --CNN backbone--> features --concept head--> concept probs (p in [0,1])
          --LNN reasoning layer--> per-class truth values --argmax--> class

The concept head is the "perception front end" that detects human-understandable
concepts; the LNN layer is the symbolic reasoner whose neurons are readable
logical rules over those concepts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import build_backbone
from .lnn_layer import LNNReasoningLayer


@dataclass
class HybridOutput:
    concept_logits: torch.Tensor   # [B, K]  (pre-sigmoid)
    concept_probs: torch.Tensor    # [B, K]  in [0, 1]
    class_truth: torch.Tensor      # [B, C]  in [0, 1]  (LNN truth values)
    class_logits: torch.Tensor     # [B, C]  = class_truth * temperature (for CE)


class HybridCNNLNN(nn.Module):
    def __init__(
        self,
        concept_matrix: torch.Tensor,
        backbone: str = "simple",
        feature_dim: int = 256,
        temperature: float = 8.0,
        pretrained: bool = True,
        small_input: bool = False,
    ):
        super().__init__()
        num_classes, num_concepts = concept_matrix.shape
        self.backbone = build_backbone(backbone, feature_dim, pretrained,
                                       small_input)
        self.concept_head = nn.Linear(self.backbone.feature_dim, num_concepts)
        self.reasoner = LNNReasoningLayer(concept_matrix)
        # Truth values live in [0, 1]; scale them into logit range for the
        # cross-entropy classification loss.
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> HybridOutput:
        feats = self.backbone(x)
        concept_logits = self.concept_head(feats)
        concept_probs = torch.sigmoid(concept_logits)
        raw = self.reasoner.raw_score(concept_probs)   # unclamped -> gradients
        class_truth = raw.clamp(0.0, 1.0)              # interpretable [0,1]
        class_logits = raw * self.temperature          # for cross-entropy
        return HybridOutput(
            concept_logits=concept_logits,
            concept_probs=concept_probs,
            class_truth=class_truth,
            class_logits=class_logits,
        )

"""Post-hoc LNN surrogate that *explains a black-box CNN*.

The intrinsic model (:mod:`src.models.hybrid`) builds interpretability *into* the
classifier. This module answers the complementary question the project cares
about -- **"how well can an LNN explain an existing, ordinary CNN?"** -- with a
post-hoc surrogate:

    frozen black-box CNN backbone ─► concept probe ─► concept probs
                                          │
                                          ▼
                                   LNN reasoning layer ─► class
                                          ▲
        trained to MIMIC the black-box CNN's own predictions (distillation)

The surrogate reads the *same features the black-box CNN uses*, squeezes them
through a human-concept bottleneck, and reasons with logical rules. It is trained
to reproduce the CNN's decisions, not the ground-truth labels. The key metric is
**fidelity**: the fraction of images on which the logical rules reach the same
verdict as the black box. High fidelity means the CNN's behaviour is, to that
extent, captured by readable logic -- an explanation *of the CNN*.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .baseline import BaselineCNN
from .lnn_layer import LNNReasoningLayer


@dataclass
class SurrogateOutput:
    concept_logits: torch.Tensor   # [B, K]
    concept_probs: torch.Tensor    # [B, K] in [0,1]
    class_truth: torch.Tensor      # [B, C] in [0,1]  (LNN truth values)
    class_logits: torch.Tensor     # [B, C]  = raw LNN score * temperature
    teacher_logits: torch.Tensor   # [B, C]  the black-box CNN's own logits


class PostHocLNNSurrogate(nn.Module):
    """Wraps a *frozen* trained :class:`BaselineCNN` (the black box) with a
    trainable concept probe + LNN reasoner.

    The black-box backbone and classifier are frozen; only the concept probe and
    the LNN weights are learned, so the surrogate cannot change the CNN -- it can
    only try to *explain* it.
    """

    def __init__(
        self,
        teacher: BaselineCNN,
        concept_matrix: torch.Tensor,
        temperature: float = 8.0,
    ):
        super().__init__()
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        num_classes, num_concepts = concept_matrix.shape
        feature_dim = teacher.backbone.feature_dim
        self.concept_head = nn.Linear(feature_dim, num_concepts)
        self.reasoner = LNNReasoningLayer(concept_matrix)
        self.temperature = temperature

    def train(self, mode: bool = True):
        # keep the frozen teacher in eval mode regardless of surrogate mode
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> SurrogateOutput:
        with torch.no_grad():
            feats = self.teacher.backbone(x)
            teacher_logits = self.teacher.classifier(feats)
        # detach so no gradient reaches the frozen black box
        concept_logits = self.concept_head(feats.detach())
        concept_probs = torch.sigmoid(concept_logits)
        raw = self.reasoner.raw_score(concept_probs)
        class_truth = raw.clamp(0.0, 1.0)
        class_logits = raw * self.temperature
        return SurrogateOutput(
            concept_logits=concept_logits,
            concept_probs=concept_probs,
            class_truth=class_truth,
            class_logits=class_logits,
            teacher_logits=teacher_logits,
        )

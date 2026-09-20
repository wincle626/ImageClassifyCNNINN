"""A differentiable Logical Neural Network (LNN) reasoning layer.

This layer implements the semantics of an LNN neuron -- a *weighted real-valued
(Lukasiewicz) logical operation with learnable importance weights* -- as
described by IBM's Logical Neural Networks (Riegel et al., 2020).  It is written
in pure PyTorch so the whole CNN->LNN pipeline is differentiable and trains
end-to-end on CPU/GPU, and so the project has no fragile external dependency.
(An optional bridge to the actual IBM ``lnn`` package for *symbolic* inference is
provided separately in :mod:`src.symbolic`.)

Semantics
---------
Every class ``c`` is one LNN conjunction (AND) neuron over the *literals* built
from the predicted concept probabilities ``p in [0, 1]``:

    literal_{c,k} = p_k            if concept k is in class c's definition
                  = 1 - p_k        otherwise            (a NOT literal)

The weighted Lukasiewicz conjunction of those literals is

    truth_c = clamp( bias_c - sum_k  w_{c,k} * (1 - literal_{c,k}),  0, 1 )

with ``w_{c,k} >= 0`` (enforced via softplus) the learnable importance of each
literal and ``bias_c`` a learnable offset (initialised to 1, so a *perfect*
concept match yields truth 1).  Each violated literal subtracts its weight --
exactly the "how much does breaking this rule hurt" reading that makes an LNN
interpretable.

Because a whole class is a single readable conjunction, the layer can report
*why* it fired: the literals that supported it and the ones that dragged it down.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LNNReasoningLayer(nn.Module):
    """Weighted Lukasiewicz conjunction, one neuron per class.

    Args:
        concept_matrix: ``[num_classes, num_concepts]`` binary ground-truth
            table.  Determines, per class, which concepts are positive literals
            and which are negated literals.
        active_init / inactive_init: initial importance weights for a class's
            defining (positive) concepts and for all other (negated) concepts.
            Active concepts start important; negated ones start weakly important
            so the rule mildly penalises signs that carry *extra* concepts.
    """

    def __init__(
        self,
        concept_matrix: torch.Tensor,
        active_init: float = 1.0,
        inactive_init: float = 0.15,
        learn_bias: bool = True,
    ):
        super().__init__()
        m = concept_matrix.float()
        self.register_buffer("mask", m)             # [C, K] 1 = positive literal
        num_classes, num_concepts = m.shape
        self.num_classes = num_classes
        self.num_concepts = num_concepts

        # softplus(raw_w) = weight, so weights stay non-negative.
        init = torch.where(m > 0.5,
                           torch.full_like(m, active_init),
                           torch.full_like(m, inactive_init))
        # invert softplus for initialisation: raw = log(exp(w) - 1)
        raw = torch.log(torch.expm1(init.clamp(min=1e-4)))
        self.raw_w = nn.Parameter(raw)              # [C, K]

        bias = torch.ones(num_classes)
        if learn_bias:
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("bias", bias)

    @property
    def weights(self) -> torch.Tensor:
        """Non-negative learned importance weights ``[C, K]``."""
        return F.softplus(self.raw_w)

    def literals(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Signed literal truth values ``[B, C, K]``.

        For each class the concept prob is used directly where the concept is a
        positive literal, and complemented (1 - p) where it is a NOT literal.
        """
        p = concept_probs.unsqueeze(1)              # [B, 1, K]
        mask = self.mask.unsqueeze(0)               # [1, C, K]
        return mask * p + (1.0 - mask) * (1.0 - p)  # [B, C, K]

    def raw_score(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Unclamped Lukasiewicz score ``bias - sum_k w*(1-literal)`` ``[B, C]``.

        This is what the classifier ranks and what the loss is computed on: it
        keeps gradients flowing even when a rule is badly violated (a hard clamp
        to 0 would otherwise saturate and kill the gradient at initialisation).
        The order of ``raw_score`` matches the order of the clamped truth value,
        so ``argmax`` is identical.
        """
        lit = self.literals(concept_probs)          # [B, C, K]
        w = self.weights.unsqueeze(0)               # [1, C, K]
        penalty = (w * (1.0 - lit)).sum(dim=-1)     # [B, C]
        return self.bias.unsqueeze(0) - penalty     # [B, C]

    def forward(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Return class truth values ``[B, C]`` clamped to ``[0, 1]`` -- the
        interpretable LNN truth value.  Use :meth:`raw_score` for the loss."""
        return self.raw_score(concept_probs).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Introspection helpers used by the explainer.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def contributions(self, concept_probs: torch.Tensor, cls: int) -> torch.Tensor:
        """Per-concept penalty contribution ``w * (1 - literal)`` for one class.

        Shape ``[B, K]``.  Small value = literal well satisfied (supports the
        class); large value = literal violated (argues against the class).
        """
        lit = self.literals(concept_probs)[:, cls, :]   # [B, K]
        return self.weights[cls].unsqueeze(0) * (1.0 - lit)

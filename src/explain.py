"""Turn a hybrid CNN+LNN prediction into a readable logical explanation, and
measure how *faithful* those explanations are via concept interventions.
"""

from __future__ import annotations

import torch

from .concepts import (CLASS_CONCEPTS, CLASS_NAMES, CONCEPTS, NUM_CONCEPTS)
from .models import HybridCNNLNN


def _fmt(name: str, prob: float, positive: bool) -> str:
    lit = name if positive else f"NOT {name}"
    return f"{lit} ({prob:.2f})"


@torch.no_grad()
def explain_sample(
    model: HybridCNNLNN,
    image: torch.Tensor,
    device: torch.device,
    top_support: int = 6,
    top_violation: int = 3,
) -> dict:
    """Explain the model's prediction for a single image tensor ``[3,H,W]``.

    Returns a dict with the predicted class, its LNN truth value, the concepts
    that supported it, the literals that argued against it, and a natural-language
    sentence.
    """
    model.eval()
    out = model(image.unsqueeze(0).to(device))
    truth = out.class_truth[0]                       # [C] clamped [0,1]
    pred = int(out.class_logits[0].argmax().item())  # rank on raw score
    probs = out.concept_probs[0]                     # [K]

    # penalty contribution per concept for the predicted class
    contrib = model.reasoner.contributions(
        out.concept_probs, pred)[0].cpu()            # [K]
    mask = model.reasoner.mask[pred].cpu()           # [K] 1=positive literal

    # Supporting literals: those the rule cares about (weight>~0) and that are
    # well satisfied (low penalty).  We surface the class's positive literals
    # first, ranked by how confidently the concept was detected.
    weights = model.reasoner.weights[pred].cpu()
    support = []
    for k in range(NUM_CONCEPTS):
        if mask[k] > 0.5:  # positive literal in this class's rule
            support.append((CONCEPTS[k], float(probs[k].item()), True,
                            float(weights[k])))
    support.sort(key=lambda t: -t[1] * t[3])
    support = support[:top_support]

    # Violations: literals contributing the most penalty (arguing against pred).
    order = torch.argsort(contrib, descending=True)
    violations = []
    for k in order.tolist():
        if contrib[k] <= 1e-3:
            break
        positive = mask[k] > 0.5
        violations.append((CONCEPTS[k], float(probs[k].item()), positive,
                           float(contrib[k])))
        if len(violations) >= top_violation:
            break

    support_str = " AND ".join(_fmt(n, p, pos) for n, p, pos, _ in support)
    sentence = (f'Classified as "{CLASS_NAMES[pred]}" '
                f"(truth={truth[pred].item():.2f}) because: {support_str}.")
    if violations:
        vstr = ", ".join(_fmt(n, p, pos) for n, p, pos, _ in violations)
        sentence += f" Weakened by: {vstr}."

    return {
        "pred": pred,
        "pred_name": CLASS_NAMES[pred],
        "truth": float(truth[pred].item()),
        "support": support,
        "violations": violations,
        "sentence": sentence,
        "concept_probs": probs.cpu(),
        "class_truth": truth.cpu(),
    }


@torch.no_grad()
def faithfulness_intervention(
    model: HybridCNNLNN,
    image: torch.Tensor,
    device: torch.device,
) -> dict:
    """Test whether the explanation is *causal*.

    For the predicted class, we take each of its defining (positive) concepts,
    force that concept's detected probability to 0 (a counterfactual "the sign
    does NOT have this concept"), and measure how much the class truth value
    drops.  A faithful rule should lose confidence -- often flipping the
    prediction -- when a genuinely necessary concept is removed.
    """
    model.eval()
    out = model(image.unsqueeze(0).to(device))
    base_probs = out.concept_probs.clone()           # [1, K]
    pred = int(out.class_logits[0].argmax().item())
    base_truth = float(out.class_truth[0, pred].item())  # clamped [0,1]

    effects = []
    for name in CLASS_CONCEPTS[pred]:
        k = CONCEPTS.index(name)
        probs = base_probs.clone()
        probs[0, k] = 0.0                            # ablate this concept
        raw = model.reasoner.raw_score(probs)[0]     # rank on raw score
        truth = raw.clamp(0.0, 1.0)
        new_pred = int(raw.argmax().item())
        effects.append({
            "concept": name,
            "truth_drop": base_truth - float(truth[pred].item()),
            "flipped": new_pred != pred,
            "new_pred_name": CLASS_NAMES[new_pred],
        })
    effects.sort(key=lambda e: -e["truth_drop"])
    return {
        "pred": pred,
        "pred_name": CLASS_NAMES[pred],
        "base_truth": base_truth,
        "effects": effects,
        "mean_truth_drop": (sum(e["truth_drop"] for e in effects)
                            / max(len(effects), 1)),
        "flip_rate": (sum(e["flipped"] for e in effects)
                      / max(len(effects), 1)),
    }


@torch.no_grad()
def explain_layerwise(model, image: torch.Tensor, device: torch.device,
                      top_concepts: int = 6) -> dict:
    """Produce a layer-by-layer logical trace for one image.

    For each ResNet stage, report which concepts it detects and what class its
    LNN concludes -- showing the decision forming with depth.
    """
    model.eval()
    out = model(image.unsqueeze(0).to(device))
    labels = getattr(model, "layer_labels", {})
    trace = []
    for s in model.stages:
        probs = out.concept_probs[s][0]
        pred = int(out.class_logits[s][0].argmax().item())
        truth = float(out.class_truth[s][0, pred].item())
        active = [(CONCEPTS[k], float(probs[k].item()))
                  for k in range(len(CONCEPTS)) if probs[k].item() > 0.5]
        active.sort(key=lambda t: -t[1])
        active = active[:top_concepts]
        concept_str = ", ".join(f"{n} ({p:.2f})" for n, p in active) or "(none)"
        tag = f"{s} ({labels[s]})" if s in labels else s
        trace.append({
            "stage": s,
            "label": labels.get(s, s),
            "pred": pred,
            "pred_name": CLASS_NAMES[pred],
            "truth": truth,
            "active_concepts": active,
            "sentence": (f"{tag}: detects {concept_str} "
                         f"-> {CLASS_NAMES[pred]} (truth {truth:.2f})"),
        })
    return {
        "trace": trace,
        "final_pred": trace[-1]["pred"],
        "final_name": trace[-1]["pred_name"],
    }

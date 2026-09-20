"""End-to-end evaluation: accuracy (hybrid vs baseline), concept-detection
quality, explanation faithfulness, and rendered example explanations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .concepts import (CLASS_NAMES, CONCEPT_GROUPS, CONCEPTS, NUM_CLASSES,
                       concept_matrix)
from .data import _MEAN, _STD, make_dataloaders
from .explain import (explain_layerwise, explain_sample,
                      faithfulness_intervention)
from .metrics import (comprehensiveness_sufficiency, explanation_stability,
                      rule_correctness, sanity_check, simulatability)
from .models import (BaselineCNN, HybridCNNLNN, LayerwiseCNNLNN,
                     PostHocLNNSurrogate)
from .train import (evaluate_baseline, evaluate_hybrid, evaluate_layerwise,
                    evaluate_surrogate)
from .utils import (Config, configure_backends, describe_device, ensure_dir,
                    get_device, save_json)


def load_hybrid(ckpt_path: str, device: torch.device) -> HybridCNNLNN:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = HybridCNNLNN(
        torch.from_numpy(concept_matrix()),
        backbone=cfg.get("backbone", "simple"),
        feature_dim=cfg.get("feature_dim", 256),
        temperature=cfg.get("temperature", 8.0),
        pretrained=False,  # weights come from the checkpoint
        small_input=cfg.get("small_input", False),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_baseline(ckpt_path: str, device: torch.device) -> BaselineCNN:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = BaselineCNN(
        NUM_CLASSES,
        backbone=cfg.get("backbone", "simple"),
        feature_dim=cfg.get("feature_dim", 256),
        pretrained=False,
        small_input=cfg.get("small_input", False),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_surrogate(ckpt_path: str, device: torch.device) -> PostHocLNNSurrogate:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    teacher = BaselineCNN(
        NUM_CLASSES,
        backbone=cfg.get("backbone", "simple"),
        feature_dim=cfg.get("feature_dim", 256),
        pretrained=False,
        small_input=cfg.get("small_input", False),
    )
    model = PostHocLNNSurrogate(
        teacher, torch.from_numpy(concept_matrix()),
        temperature=cfg.get("temperature", 8.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_layerwise(ckpt_path: str, device: torch.device) -> LayerwiseCNNLNN:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = LayerwiseCNNLNN(
        torch.from_numpy(concept_matrix()),
        pretrained=False,  # weights come from the checkpoint
        small_input=cfg.get("small_input", False),
        temperature=cfg.get("temperature", 8.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _denormalize(img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    x = (img.cpu() * std + mean).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


@torch.no_grad()
def per_concept_scores(model: HybridCNNLNN, loader, device) -> dict:
    """Precision/recall/F1 per concept over the test set."""
    tp = np.zeros(len(CONCEPTS))
    fp = np.zeros(len(CONCEPTS))
    fn = np.zeros(len(CONCEPTS))
    for images, _, concepts in loader:
        images = images.to(device)
        probs = model(images).concept_probs.cpu().numpy()
        pred = (probs > 0.5).astype(np.float32)
        gt = concepts.numpy()
        tp += ((pred == 1) & (gt == 1)).sum(0)
        fp += ((pred == 1) & (gt == 0)).sum(0)
        fn += ((pred == 0) & (gt == 1)).sum(0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {CONCEPTS[i]: {"precision": float(prec[i]),
                          "recall": float(rec[i]),
                          "f1": float(f1[i])}
            for i in range(len(CONCEPTS))}


def run_full_evaluation(cfg: Config, n_examples: int = 12,
                        n_faithfulness: int = 200) -> dict:
    device = get_device(cfg.device)
    configure_backends(device)
    print(f"[eval] device: {describe_device(device)}")
    out_dir = ensure_dir(cfg.out_dir)

    _, test_loader = make_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers,
        cfg.subset_fraction, cfg.seed, cfg.download,
    )

    hybrid = load_hybrid(str(Path(cfg.out_dir) / "hybrid.pt"), device)
    baseline_path = Path(cfg.out_dir) / "baseline.pt"
    baseline_acc = None
    if baseline_path.exists():
        baseline = load_baseline(str(baseline_path), device)
        baseline_acc = evaluate_baseline(baseline, test_loader, device)

    hy = evaluate_hybrid(hybrid, test_loader, device)
    concept_scores = per_concept_scores(hybrid, test_loader, device)
    macro_f1 = float(np.mean([v["f1"] for v in concept_scores.values()]))

    # ---- faithfulness over a random subset of test images -----------------
    faith = _run_faithfulness(hybrid, test_loader, device, n_faithfulness)

    # ---- explanation-quality metrics (faithfulness / plausibility / robustness)
    print("[eval] computing explanation-quality metrics ...")
    explanation_metrics = {
        "comprehensiveness_sufficiency": comprehensiveness_sufficiency(
            hybrid, test_loader, device, n=min(500, n_faithfulness * 3)),
        "rule_correctness": rule_correctness(hybrid, test_loader, device, n=1000),
        "stability": explanation_stability(hybrid, test_loader, device, n=80),
        "sanity_check": sanity_check(hybrid, test_loader, device, n=500),
        "simulatability": simulatability(hybrid, test_loader, device, n=2000),
    }

    # ---- example explanations (render to a figure) ------------------------
    example_sentences = _render_examples(hybrid, test_loader, device,
                                         out_dir, n_examples)

    # ---- post-hoc surrogate: does an LNN explain the black-box CNN? --------
    surrogate_results = None
    surrogate_path = Path(cfg.out_dir) / "surrogate.pt"
    if surrogate_path.exists():
        surrogate = load_surrogate(str(surrogate_path), device)
        sur = evaluate_surrogate(surrogate, test_loader, device)
        sur_concepts = per_concept_scores(surrogate, test_loader, device)
        sur_examples = _surrogate_examples(surrogate, test_loader, device,
                                           n_examples // 2 or 4)
        surrogate_results = {
            "fidelity": sur["fidelity"],
            "surrogate_acc": sur["surrogate_acc"],
            "teacher_acc": sur["teacher_acc"],
            "concept_macro_f1": float(np.mean([v["f1"]
                                               for v in sur_concepts.values()])),
            "examples": sur_examples,
        }

    # ---- layer-by-layer: how the decision forms with depth ----------------
    layerwise_results = None
    layerwise_path = Path(cfg.out_dir) / "layerwise.pt"
    if layerwise_path.exists():
        lw = load_layerwise(str(layerwise_path), device)
        layerwise_results = _layerwise_evaluation(lw, test_loader, device,
                                                  out_dir, n_examples)

    results = {
        "hybrid_class_acc": hy["class_acc"],
        "hybrid_concept_acc": hy["concept_acc"],
        "concept_macro_f1": macro_f1,
        "baseline_class_acc": baseline_acc,
        "accuracy_cost": (None if baseline_acc is None
                          else baseline_acc - hy["class_acc"]),
        "faithfulness": faith,
        "explanation_metrics": explanation_metrics,
        "surrogate": surrogate_results,
        "layerwise": layerwise_results,
        "per_concept": concept_scores,
        "example_sentences": example_sentences,
    }
    save_json(results, Path(out_dir) / "evaluation.json")
    _write_report(results, Path(out_dir) / "REPORT.md")
    return results


@torch.no_grad()
def _run_faithfulness(model, loader, device, n: int) -> dict:
    drops, flips, count = [], [], 0
    for images, _, _ in loader:
        for i in range(images.size(0)):
            f = faithfulness_intervention(model, images[i], device)
            drops.append(f["mean_truth_drop"])
            flips.append(f["flip_rate"])
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    return {
        "n_samples": count,
        "mean_truth_drop_when_concept_removed": float(np.mean(drops)) if drops else 0.0,
        "mean_flip_rate_when_concept_removed": float(np.mean(flips)) if flips else 0.0,
    }


def _render_examples(model, loader, device, out_dir, n_examples) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images, labels, _ = next(iter(loader))
    n = min(n_examples, images.size(0))
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    axes = np.array(axes).reshape(-1)
    sentences = []
    for i in range(n):
        exp = explain_sample(model, images[i], device)
        ax = axes[i]
        ax.imshow(_denormalize(images[i]))
        ax.axis("off")
        correct = exp["pred"] == int(labels[i].item())
        colour = "green" if correct else "red"
        support = "\n".join(
            f"  {'' if pos else 'NOT '}{name} ({p:.2f})"
            for name, p, pos, _ in exp["support"][:5])
        title = (f"pred: {exp['pred_name']}  (truth {exp['truth']:.2f})\n"
                 f"true: {CLASS_NAMES[int(labels[i].item())]}\n"
                 f"because:\n{support}")
        ax.set_title(title, fontsize=7, color=colour, loc="left")
        sentences.append(exp["sentence"])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig_path = Path(out_dir) / "example_explanations.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved example explanations -> {fig_path}")
    return sentences


@torch.no_grad()
def _surrogate_examples(model, loader, device, n: int) -> list[dict]:
    """For a few images, report the black box's verdict and the LNN's logical
    account of it, plus whether they agree."""
    images, labels, _ = next(iter(loader))
    out = model(images.to(device))
    teacher_pred = out.teacher_logits.argmax(1).cpu()
    examples = []
    for i in range(min(n, images.size(0))):
        exp = explain_sample(model, images[i], device)
        t = int(teacher_pred[i].item())
        examples.append({
            "true": CLASS_NAMES[int(labels[i].item())],
            "cnn_says": CLASS_NAMES[t],
            "lnn_agrees": exp["pred"] == t,
            "lnn_explanation": exp["sentence"],
        })
    return examples


@torch.no_grad()
def _per_stage_concept_f1(model, loader, device) -> dict:
    """F1 per (stage, concept) over the test set."""
    stages = model.stages
    K = len(CONCEPTS)
    tp = {s: np.zeros(K) for s in stages}
    fp = {s: np.zeros(K) for s in stages}
    fn = {s: np.zeros(K) for s in stages}
    for images, _, concepts in loader:
        images = images.to(device)
        out = model(images)
        gt = concepts.numpy()
        for s in stages:
            pred = (out.concept_probs[s].cpu().numpy() > 0.5).astype(np.float32)
            tp[s] += ((pred == 1) & (gt == 1)).sum(0)
            fp[s] += ((pred == 1) & (gt == 0)).sum(0)
            fn[s] += ((pred == 0) & (gt == 1)).sum(0)
    f1 = {}
    for s in stages:
        prec = tp[s] / np.maximum(tp[s] + fp[s], 1)
        rec = tp[s] / np.maximum(tp[s] + fn[s], 1)
        f1[s] = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return f1  # {stage: np.array[K]}


def _layerwise_evaluation(model, loader, device, out_dir, n_examples) -> dict:
    stage_metrics = evaluate_layerwise(model, loader, device)
    f1 = _per_stage_concept_f1(model, loader, device)

    # aggregate concept F1 by group -> the "concept emergence" table/heatmap
    groups = list(CONCEPT_GROUPS.keys())
    group_idx = {g: [CONCEPTS.index(c) for c in CONCEPT_GROUPS[g]] for g in groups}
    emergence = {s: {g: float(np.mean(f1[s][group_idx[g]])) for g in groups}
                 for s in model.stages}
    macro_f1 = {s: float(np.mean(f1[s])) for s in model.stages}

    _render_emergence_figure(model.stages, groups, emergence, out_dir)

    # per-image layer-by-layer logical traces
    images, labels, _ = next(iter(loader))
    traces = []
    for i in range(min(4, images.size(0))):
        exp = explain_layerwise(model, images[i], device)
        traces.append({
            "true": CLASS_NAMES[int(labels[i].item())],
            "final": exp["final_name"],
            "steps": [t["sentence"] for t in exp["trace"]],
        })

    return {
        "stage_acc": stage_metrics["stage_acc"],
        "stage_concept_acc": stage_metrics["stage_concept_acc"],
        "stage_macro_f1": macro_f1,
        "concept_emergence_by_group": emergence,
        "layer_labels": getattr(model, "layer_labels", {}),
        "example_traces": traces,
    }


def _render_emergence_figure(stages, groups, emergence, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.array([[emergence[s][g] for g in groups] for s in stages])
    fig, ax = plt.subplots(figsize=(1.3 * len(groups) + 2,
                                    0.42 * len(stages) + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=8)
    ax.set_ylabel("ResNet-18 conv layer (shallow → deep)", fontsize=9)
    ax.set_title("Concept-detection F1 by conv layer and concept group\n"
                 "(how concepts emerge across the 17 conv layers)", fontsize=9)
    for i in range(len(stages)):
        for j in range(len(groups)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig_path = Path(out_dir) / "layerwise_concept_emergence.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved concept-emergence heatmap -> {fig_path}")


def _write_report(r: dict, path: Path) -> None:
    lines = []
    lines.append("# CNN + LNN neuro-symbolic traffic-sign classifier — report\n")
    lines.append("## Classification accuracy (GTSRB test set)\n")
    lines.append(f"- **Hybrid (CNN concept detector + LNN reasoner):** "
                 f"{r['hybrid_class_acc']*100:.2f}%")
    if r["baseline_class_acc"] is not None:
        lines.append(f"- **Baseline (plain CNN):** "
                     f"{r['baseline_class_acc']*100:.2f}%")
        lines.append(f"- **Accuracy cost of the hybrid design:** "
                     f"{r['accuracy_cost']*100:+.2f} pts")
    else:
        lines.append("- Baseline not trained (run `train-baseline` to compare).")
    lines.append("")
    lines.append("## Concept detection (the CNN front end)\n")
    lines.append(f"- Concept-wise accuracy: {r['hybrid_concept_acc']*100:.2f}%")
    lines.append(f"- Concept macro-F1: {r['concept_macro_f1']*100:.2f}%")
    lines.append("")
    lines.append("## Explanation faithfulness (concept intervention)\n")
    f = r["faithfulness"]
    lines.append(f"Over {f['n_samples']} test images we removed each concept the "
                 "winning rule depends on and measured the effect:")
    lines.append(f"- Mean truth-value drop when a required concept is removed: "
                 f"{f['mean_truth_drop_when_concept_removed']:.3f}")
    lines.append(f"- Mean prediction flip-rate when a required concept is "
                 f"removed: {f['mean_flip_rate_when_concept_removed']*100:.1f}%")
    lines.append("\nA high drop / flip-rate means the stated reasons are the "
                 "actual causes of the decision (faithful explanations).\n")

    if r.get("explanation_metrics"):
        em = r["explanation_metrics"]
        cs = em["comprehensiveness_sufficiency"]
        rc = em["rule_correctness"]
        st = em["stability"]
        lines.append("## Explanation-quality metrics\n")
        lines.append(f"Measured over up to {cs['n']} test images "
                     f"(top-k concepts, k ∈ {cs['ks']}).\n")
        lines.append("**Faithfulness — comprehensiveness & sufficiency** "
                     "(concept ablation of the predicted class):\n")
        lines.append(f"- Base predicted-class confidence: {cs['base_confidence']:.3f}")
        lines.append(f"- **Comprehensiveness**: {cs['comprehensiveness']:.3f} "
                     "(drop when the top concepts are removed — *higher is better*)")
        lines.append(f"- **Sufficiency**: {cs['sufficiency']:.3f} "
                     "(drop when only the top concepts are kept — *closer to 0 is "
                     "better*)")
        lines.append("")
        lines.append("**Plausibility — rule correctness vs the ground-truth GTSRB "
                     "rules** (do the detected concepts match the predicted class's "
                     "known concept set?):\n")
        allp = rc["all_predictions"]
        corp = rc["correct_predictions"]
        lines.append(f"- All predictions (n={allp['n']}): precision "
                     f"{allp['precision']*100:.1f}%, recall {allp['recall']*100:.1f}%, "
                     f"F1 {allp['f1']*100:.1f}%, exact-rule-match "
                     f"{allp['exact_match']*100:.1f}%")
        lines.append(f"- Correct predictions (n={corp['n']}): precision "
                     f"{corp['precision']*100:.1f}%, recall {corp['recall']*100:.1f}%, "
                     f"F1 {corp['f1']*100:.1f}%, exact-rule-match "
                     f"{corp['exact_match']*100:.1f}%")
        lines.append("")
        lines.append("**Robustness — explanation stability** under small "
                     f"perturbations ({st['n_views']} views of {st['n']} images):\n")
        lines.append(f"- Prediction consistency: {st['prediction_consistency']*100:.1f}% "
                     "(1.0 = the class never flips)")
        lines.append(f"- Explanation Jaccard: {st['explanation_jaccard']*100:.1f}% "
                     "(overlap of the supporting-concept set across perturbations)")
        lines.append("")

        if "simulatability" in em:
            sim = em["simulatability"]
            lines.append("**Simulatability** — can a simple linear proxy reproduce "
                         "the model's prediction from the concepts alone? "
                         f"(n={sim['n']}):\n")
            lines.append(f"- From binary (human-readable) concepts: "
                         f"{sim['simulatability_binary']*100:.1f}% agreement")
            lines.append(f"- From soft concept probabilities: "
                         f"{sim['simulatability_soft']*100:.1f}% agreement")
            lines.append("\nHigh agreement means the human-readable concepts are a "
                         "sufficient basis to simulate the model's decisions.\n")

        if "sanity_check" in em:
            sc = em["sanity_check"]
            tr, rnd, dl = sc["trained"], sc["randomized_lnn"], sc["deltas"]
            verdict = "PASSED" if sc["passed"] else "NOT PASSED"
            lines.append("**Sanity check (control)** — randomising the LNN weights "
                         "should make the metrics collapse; if they don't, the "
                         "explanations are not tied to the learned logic.\n")
            lines.append(f"- Accuracy: {tr['accuracy']*100:.2f}% (trained) → "
                         f"{rnd['accuracy']*100:.2f}% (randomised LNN); "
                         f"chance ≈ {sc['chance_level']*100:.2f}%")
            lines.append(f"- Comprehensiveness: {tr['comprehensiveness']:.3f} → "
                         f"{rnd['comprehensiveness']:.3f}")
            lines.append(f"- Rule-correctness F1: {tr['rule_f1']*100:.1f}% → "
                         f"{rnd['rule_f1']*100:.1f}%")
            lines.append(f"- **Sanity check: {verdict}** "
                         "(metrics collapse when the logic is randomised).\n")

    if r.get("surrogate"):
        s = r["surrogate"]
        lines.append("## Post-hoc: can an LNN explain the black-box CNN?\n")
        lines.append("An LNN surrogate was trained to reproduce the plain CNN's "
                     "predictions through a human-concept + logic bottleneck.")
        lines.append(f"- **Fidelity (LNN agrees with the CNN):** "
                     f"{s['fidelity']*100:.2f}%")
        lines.append(f"- Surrogate accuracy vs ground truth: "
                     f"{s['surrogate_acc']*100:.2f}%")
        lines.append(f"- Black-box CNN accuracy vs ground truth: "
                     f"{s['teacher_acc']*100:.2f}%")
        lines.append(f"- Surrogate concept macro-F1: "
                     f"{s['concept_macro_f1']*100:.2f}%")
        lines.append("\nFidelity is the key number: the fraction of the CNN's own "
                     "decisions that simple logical rules over human concepts can "
                     "reproduce -- i.e. how much of the black box the LNN explains.\n")
        lines.append("Example — the CNN's verdict and the LNN's logical account:\n")
        for ex in s["examples"][:5]:
            mark = "agrees" if ex["lnn_agrees"] else "DISAGREES"
            lines.append(f"- CNN says **{ex['cnn_says']}** (true: {ex['true']}); "
                         f"LNN {mark}. {ex['lnn_explanation']}")
        lines.append("")

    if r.get("layerwise"):
        lw = r["layerwise"]
        labels = lw.get("layer_labels", {})
        lines.append("## Layer-by-layer explainability (ResNet-18 + LNN)\n")
        lines.append("ResNet-18 has 18 weight layers (1 stem conv + 16 block "
                     "convs + 1 fc). A concept probe + LNN reasoner is attached "
                     "after **each of the 17 convolutional layers**; the 18th "
                     "layer (fc) is replaced by the deepest LNN. So we can read "
                     "what the network concludes after every convolution.\n")
        lines.append("**How much of the decision is settled at each conv layer** "
                     "(per-layer classification accuracy):\n")
        for s in lw["stage_acc"]:
            lab = f" — {labels[s]}" if s in labels else ""
            lines.append(f"- `{s}`{lab}: accuracy {lw['stage_acc'][s]*100:.2f}%, "
                         f"concept macro-F1 {lw['stage_macro_f1'][s]*100:.2f}%")
        lines.append("\n**Concept emergence with depth** (F1 by concept group per "
                     "conv layer; see `layerwise_concept_emergence.png`):\n")
        groups = list(next(iter(lw["concept_emergence_by_group"].values())).keys())
        header = "| stage | " + " | ".join(groups) + " |"
        sep = "|" + "---|" * (len(groups) + 1)
        lines.append(header)
        lines.append(sep)
        for s, gd in lw["concept_emergence_by_group"].items():
            row = "| " + s + " | " + " | ".join(f"{gd[g]:.2f}" for g in groups) + " |"
            lines.append(row)
        lines.append("\nLow-level concepts (shape, colour) are detectable in early "
                     "stages; fine concepts (specific digits/symbols) only appear "
                     "in deeper stages -- an explicit account of what each layer "
                     "contributes.\n")
        lines.append("**Example layer-by-layer traces** (the decision forming):\n")
        for tr in lw["example_traces"][:3]:
            lines.append(f"- true: **{tr['true']}** -> final: **{tr['final']}**")
            for step in tr["steps"]:
                lines.append(f"    - {step}")
        lines.append("")

    lines.append("## Example explanations\n")
    lines.append("See `example_explanations.png`. Sample sentences:\n")
    for s in r["example_sentences"][:6]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Weakest-detected concepts (lowest F1)\n")
    worst = sorted(r["per_concept"].items(), key=lambda kv: kv[1]["f1"])[:8]
    for name, sc in worst:
        lines.append(f"- `{name}`: F1={sc['f1']:.2f} "
                     f"(P={sc['precision']:.2f}, R={sc['recall']:.2f})")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote report -> {path}")

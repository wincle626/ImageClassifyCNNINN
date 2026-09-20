"""Training loops for the hybrid CNN+LNN model and the CNN baseline."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .concepts import concept_matrix
from .data import make_dataloaders
from .models import BaselineCNN, HybridCNNLNN, LayerwiseCNNLNN
from .utils import (AmpHelper, Config, configure_backends, describe_device,
                    ensure_dir, get_device, save_json, set_seed)


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target).float().mean().item()


# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
def train_hybrid(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    configure_backends(device)
    amp = AmpHelper(device, cfg.amp)
    print(f"[hybrid] device: {describe_device(device)}"
          f"{'  | AMP on' if amp.enabled else ''}")
    train_loader, test_loader = make_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers,
        cfg.subset_fraction, cfg.seed, cfg.download,
    )

    m = torch.from_numpy(concept_matrix())
    model = HybridCNNLNN(m, backbone=cfg.backbone, feature_dim=cfg.feature_dim,
                         temperature=cfg.temperature, pretrained=cfg.pretrained,
                         small_input=cfg.small_input).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "cls": 0.0, "concept": 0.0, "n": 0}
        for images, labels, concepts in tqdm(
                train_loader, desc=f"[hybrid] epoch {epoch+1}/{cfg.epochs}",
                leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            opt.zero_grad()
            with amp.autocast():
                out = model(images)
                class_loss = F.cross_entropy(out.class_logits, labels)
                concept_loss = F.binary_cross_entropy_with_logits(
                    out.concept_logits, concepts)
                loss = (cfg.class_loss_weight * class_loss
                        + cfg.concept_loss_weight * concept_loss)
            amp.backward_step(loss, opt)

            bs = images.size(0)
            running["loss"] += loss.item() * bs
            running["cls"] += class_loss.item() * bs
            running["concept"] += concept_loss.item() * bs
            running["n"] += bs
        sched.step()

        val = evaluate_hybrid(model, test_loader, device)
        n = running["n"]
        rec = {
            "epoch": epoch + 1,
            "train_loss": running["loss"] / n,
            "train_class_loss": running["cls"] / n,
            "train_concept_loss": running["concept"] / n,
            "test_acc": val["class_acc"],
            "test_concept_acc": val["concept_acc"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(f"[hybrid] epoch {epoch+1}: "
              f"loss={rec['train_loss']:.3f} "
              f"test_acc={rec['test_acc']:.4f} "
              f"concept_acc={rec['test_concept_acc']:.4f} "
              f"({rec['seconds']}s)")

    out_dir = ensure_dir(cfg.out_dir)
    ckpt_path = Path(out_dir) / "hybrid.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": cfg.to_dict()}, ckpt_path)
    save_json({"history": history, "config": cfg.to_dict()},
              Path(out_dir) / "hybrid_history.json")
    print(f"[hybrid] saved checkpoint -> {ckpt_path}")
    return {"model": model, "history": history, "checkpoint": str(ckpt_path)}


@torch.no_grad()
def evaluate_hybrid(model: HybridCNNLNN, loader: DataLoader,
                    device: torch.device) -> dict:
    model.eval()
    correct = total = 0
    concept_correct = concept_total = 0
    for images, labels, concepts in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        concepts = concepts.to(device, non_blocking=True)
        out = model(images)
        correct += (out.class_logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pred_c = (out.concept_probs > 0.5).float()
        concept_correct += (pred_c == concepts).sum().item()
        concept_total += concepts.numel()
    return {
        "class_acc": correct / max(total, 1),
        "concept_acc": concept_correct / max(concept_total, 1),
    }


# --------------------------------------------------------------------------- #
# Baseline model
# --------------------------------------------------------------------------- #
def train_baseline(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    configure_backends(device)
    amp = AmpHelper(device, cfg.amp)
    print(f"[baseline] device: {describe_device(device)}"
          f"{'  | AMP on' if amp.enabled else ''}")
    train_loader, test_loader = make_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers,
        cfg.subset_fraction, cfg.seed, cfg.download,
    )

    from .concepts import NUM_CLASSES
    model = BaselineCNN(NUM_CLASSES, backbone=cfg.backbone,
                        feature_dim=cfg.feature_dim, pretrained=cfg.pretrained,
                        small_input=cfg.small_input).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        run_loss = run_n = 0
        for images, labels, _ in tqdm(
                train_loader, desc=f"[baseline] epoch {epoch+1}/{cfg.epochs}",
                leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad()
            with amp.autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            amp.backward_step(loss, opt)
            run_loss += loss.item() * images.size(0)
            run_n += images.size(0)
        sched.step()

        acc = evaluate_baseline(model, test_loader, device)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / run_n,
               "test_acc": acc, "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[baseline] epoch {epoch+1}: loss={rec['train_loss']:.3f} "
              f"test_acc={acc:.4f} ({rec['seconds']}s)")

    out_dir = ensure_dir(cfg.out_dir)
    ckpt_path = Path(out_dir) / "baseline.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": cfg.to_dict()}, ckpt_path)
    save_json({"history": history, "config": cfg.to_dict()},
              Path(out_dir) / "baseline_history.json")
    print(f"[baseline] saved checkpoint -> {ckpt_path}")
    return {"model": model, "history": history, "checkpoint": str(ckpt_path)}


@torch.no_grad()
def evaluate_baseline(model: BaselineCNN, loader: DataLoader,
                      device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Post-hoc LNN surrogate (explains a trained black-box CNN)
# --------------------------------------------------------------------------- #
def train_surrogate(cfg: Config, kd_temp: float = 3.0) -> dict:
    """Fit an LNN surrogate to a *trained* baseline CNN.

    The surrogate is trained to reproduce the black-box CNN's predictions
    (knowledge distillation) while keeping its concept probe grounded in the
    ground-truth concepts (so the rules stay readable).  Requires a trained
    ``outputs/baseline.pt`` -- run ``train-baseline`` first.
    """
    from pathlib import Path as _Path

    from .models import BaselineCNN, PostHocLNNSurrogate
    from .concepts import NUM_CLASSES

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    configure_backends(device)
    amp = AmpHelper(device, cfg.amp)
    print(f"[surrogate] device: {describe_device(device)}"
          f"{'  | AMP on' if amp.enabled else ''}")

    baseline_path = _Path(cfg.out_dir) / "baseline.pt"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"{baseline_path} not found. Train the black-box CNN first with "
            "`python run.py train-baseline`."
        )
    ckpt = torch.load(baseline_path, map_location=device)
    bcfg = ckpt.get("config", {})
    teacher = BaselineCNN(
        NUM_CLASSES,
        backbone=bcfg.get("backbone", "simple"),
        feature_dim=bcfg.get("feature_dim", cfg.feature_dim),
        pretrained=False,  # weights come from the checkpoint
        small_input=bcfg.get("small_input", False),
    )
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.to(device)

    train_loader, test_loader = make_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers,
        cfg.subset_fraction, cfg.seed, cfg.download,
    )

    m = torch.from_numpy(concept_matrix())
    model = PostHocLNNSurrogate(teacher, m, temperature=cfg.temperature).to(device)
    # only the probe + reasoner are trainable (teacher is frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        run = {"loss": 0.0, "n": 0}
        for images, _, concepts in tqdm(
                train_loader, desc=f"[surrogate] epoch {epoch+1}/{cfg.epochs}",
                leave=False):
            images = images.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            opt.zero_grad()
            with amp.autocast():
                out = model(images)
                # distillation: match the black box's soft predictions
                teacher_soft = F.softmax(out.teacher_logits / kd_temp, dim=1)
                student_log = F.log_softmax(out.class_logits / kd_temp, dim=1)
                kd_loss = F.kl_div(student_log, teacher_soft,
                                   reduction="batchmean") * (kd_temp ** 2)
                concept_loss = F.binary_cross_entropy_with_logits(
                    out.concept_logits, concepts)
                loss = kd_loss + cfg.concept_loss_weight * concept_loss
            amp.backward_step(loss, opt)
            run["loss"] += loss.item() * images.size(0)
            run["n"] += images.size(0)
        sched.step()

        val = evaluate_surrogate(model, test_loader, device)
        rec = {"epoch": epoch + 1, "train_loss": run["loss"] / run["n"],
               "fidelity": val["fidelity"], "surrogate_acc": val["surrogate_acc"],
               "teacher_acc": val["teacher_acc"],
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[surrogate] epoch {epoch+1}: loss={rec['train_loss']:.3f} "
              f"fidelity={rec['fidelity']:.4f} "
              f"surrogate_acc={rec['surrogate_acc']:.4f} "
              f"teacher_acc={rec['teacher_acc']:.4f} ({rec['seconds']}s)")

    out_dir = ensure_dir(cfg.out_dir)
    ckpt_path = _Path(out_dir) / "surrogate.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": cfg.to_dict()}, ckpt_path)
    save_json({"history": history, "config": cfg.to_dict()},
              _Path(out_dir) / "surrogate_history.json")
    print(f"[surrogate] saved checkpoint -> {ckpt_path}")
    return {"model": model, "history": history, "checkpoint": str(ckpt_path)}


@torch.no_grad()
def evaluate_surrogate(model, loader: DataLoader, device: torch.device) -> dict:
    """Fidelity (agreement with the black box), plus surrogate & teacher
    accuracy against the ground truth."""
    model.eval()
    agree = s_correct = t_correct = total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        s_pred = out.class_logits.argmax(1)
        t_pred = out.teacher_logits.argmax(1)
        agree += (s_pred == t_pred).sum().item()
        s_correct += (s_pred == labels).sum().item()
        t_correct += (t_pred == labels).sum().item()
        total += labels.size(0)
    total = max(total, 1)
    return {
        "fidelity": agree / total,          # surrogate vs black box
        "surrogate_acc": s_correct / total, # surrogate vs truth
        "teacher_acc": t_correct / total,   # black box vs truth
    }


# --------------------------------------------------------------------------- #
# Layer-by-layer model (ResNet-18 with a concept probe + LNN at every stage)
# --------------------------------------------------------------------------- #
def train_layerwise(cfg: Config) -> dict:
    """Train the layer-wise ResNet18+LNN with deep supervision: every residual
    stage learns to detect concepts and reason to the class from them."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    configure_backends(device)
    amp = AmpHelper(device, cfg.amp)
    print(f"[layerwise] device: {describe_device(device)}"
          f"{'  | AMP on' if amp.enabled else ''}")
    train_loader, test_loader = make_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers,
        cfg.subset_fraction, cfg.seed, cfg.download,
    )

    m = torch.from_numpy(concept_matrix())
    model = LayerwiseCNNLNN(m, pretrained=cfg.pretrained,
                            small_input=cfg.small_input,
                            temperature=cfg.temperature).to(device)
    # deeper layers weigh more in the classification loss (final layer decides)
    n_taps = len(model.stages)
    stage_w = {s: 0.2 + 0.8 * (i / max(n_taps - 1, 1))
               for i, s in enumerate(model.stages)}

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        run_loss = run_n = 0
        for images, labels, concepts in tqdm(
                train_loader, desc=f"[layerwise] epoch {epoch+1}/{cfg.epochs}",
                leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            opt.zero_grad()
            with amp.autocast():
                out = model(images)
                loss = 0.0
                for s in model.stages:
                    concept_loss = F.binary_cross_entropy_with_logits(
                        out.concept_logits[s], concepts)
                    class_loss = F.cross_entropy(out.class_logits[s], labels)
                    loss = loss + cfg.concept_loss_weight * concept_loss \
                        + stage_w[s] * cfg.class_loss_weight * class_loss
            amp.backward_step(loss, opt)
            run_loss += float(loss.item()) * images.size(0)
            run_n += images.size(0)
        sched.step()

        val = evaluate_layerwise(model, test_loader, device)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / run_n,
               "stage_acc": val["stage_acc"],
               "final_acc": val["stage_acc"][model.stages[-1]],
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        # show a few representative conv layers to keep the line readable
        shown = [model.stages[0], model.stages[4], model.stages[8],
                 model.stages[12], model.stages[-1]]
        acc_str = "  ".join(f"{s}={val['stage_acc'][s]:.3f}" for s in shown)
        print(f"[layerwise] epoch {epoch+1}: loss={rec['train_loss']:.3f} "
              f"| conv-layer acc ({', '.join(shown)}): {acc_str} "
              f"({rec['seconds']}s)")

    out_dir = ensure_dir(cfg.out_dir)
    ckpt_path = Path(out_dir) / "layerwise.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": cfg.to_dict()}, ckpt_path)
    save_json({"history": history, "config": cfg.to_dict()},
              Path(out_dir) / "layerwise_history.json")
    print(f"[layerwise] saved checkpoint -> {ckpt_path}")
    return {"model": model, "history": history, "checkpoint": str(ckpt_path)}


@torch.no_grad()
def evaluate_layerwise(model, loader: DataLoader, device: torch.device) -> dict:
    """Per-stage classification accuracy and per-stage concept accuracy."""
    model.eval()
    stages = model.stages
    correct = {s: 0 for s in stages}
    concept_correct = {s: 0 for s in stages}
    total = concept_total = 0
    for images, labels, concepts in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        concepts = concepts.to(device, non_blocking=True)
        out = model(images)
        for s in stages:
            correct[s] += (out.class_logits[s].argmax(1) == labels).sum().item()
            pred_c = (out.concept_probs[s] > 0.5).float()
            concept_correct[s] += (pred_c == concepts).sum().item()
        total += labels.size(0)
        concept_total += concepts.numel()
    return {
        "stage_acc": {s: correct[s] / max(total, 1) for s in stages},
        "stage_concept_acc": {s: concept_correct[s] / max(concept_total, 1)
                              for s in stages},
    }

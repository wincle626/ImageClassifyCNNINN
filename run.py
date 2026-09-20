"""Command-line entrypoint for the CNN+LNN explainable traffic-sign project.

Examples
--------
    # quick smoke test on 5% of the data, 2 epochs (CPU-friendly)
    python run.py train-hybrid   --config config.yaml --epochs 2 --subset 0.05
    python run.py train-baseline --config config.yaml --epochs 2 --subset 0.05
    python run.py evaluate       --config config.yaml --subset 0.05

    # full run
    python run.py train-hybrid --config config.yaml
    python run.py train-baseline --config config.yaml
    python run.py evaluate --config config.yaml

    # explain / inspect
    python run.py concepts                      # print the concept table
    python run.py explain --checkpoint outputs/hybrid.pt --num 8
"""

from __future__ import annotations

import argparse

from src.utils import Config


def _apply_overrides(cfg: Config, args) -> Config:
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    if getattr(args, "subset", None) is not None:
        cfg.subset_fraction = args.subset
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "data_root", None) is not None:
        cfg.data_root = args.data_root
    if getattr(args, "out_dir", None) is not None:
        cfg.out_dir = args.out_dir
    if getattr(args, "no_download", False):
        cfg.download = False
    if getattr(args, "backbone", None) is not None:
        cfg.backbone = args.backbone
    if getattr(args, "no_pretrained", False):
        cfg.pretrained = False
    if getattr(args, "small_input", False):
        cfg.small_input = True
    if getattr(args, "device", None) is not None:
        cfg.device = args.device
    if getattr(args, "no_amp", False):
        cfg.amp = False
    return cfg


def _add_common(p):
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--subset", type=float, default=None,
                   help="fraction of each split to use (0-1)")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--data-root", dest="data_root", default=None)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--backbone", choices=["resnet18", "simple"], default=None)
    p.add_argument("--no-pretrained", dest="no_pretrained", action="store_true")
    p.add_argument("--small-input", dest="small_input", action="store_true")
    p.add_argument("--device", default=None,
                   help="auto | cuda | cuda:0 | cpu (default: auto)")
    p.add_argument("--no-amp", dest="no_amp", action="store_true",
                   help="disable mixed-precision training on CUDA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train-hybrid", help="train the CNN+LNN hybrid model")
    _add_common(p)

    p = sub.add_parser("train-baseline", help="train the plain CNN baseline")
    _add_common(p)

    p = sub.add_parser(
        "train-surrogate",
        help="fit an LNN that explains the trained black-box CNN (needs baseline)")
    _add_common(p)

    p = sub.add_parser(
        "train-layerwise",
        help="train the ResNet18+LNN with a concept probe/reasoner at every stage")
    _add_common(p)

    p = sub.add_parser("evaluate", help="evaluate + generate the report")
    _add_common(p)
    p.add_argument("--examples", type=int, default=12)

    p = sub.add_parser("explain", help="print explanations for test images")
    _add_common(p)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--mode", choices=["hybrid", "surrogate", "layerwise"],
                   default="hybrid",
                   help="explain the intrinsic hybrid, the black-box CNN via its "
                        "LNN surrogate, or the ResNet layer-by-layer")

    sub.add_parser("concepts", help="print the GTSRB concept table")
    sub.add_parser("device", help="report PyTorch / CUDA / GPU status")

    args = parser.parse_args()

    if args.command == "concepts":
        _print_concepts()
        return
    if args.command == "device":
        _print_device()
        return

    cfg = _apply_overrides(Config.load(args.config), args)

    if args.command == "train-hybrid":
        from src.train import train_hybrid
        train_hybrid(cfg)
    elif args.command == "train-baseline":
        from src.train import train_baseline
        train_baseline(cfg)
    elif args.command == "train-surrogate":
        from src.train import train_surrogate
        train_surrogate(cfg)
    elif args.command == "train-layerwise":
        from src.train import train_layerwise
        train_layerwise(cfg)
    elif args.command == "evaluate":
        from src.evaluate import run_full_evaluation
        r = run_full_evaluation(cfg, n_examples=args.examples)
        print("\n=== summary ===")
        print(f"hybrid accuracy   : {r['hybrid_class_acc']*100:.2f}%")
        if r["baseline_class_acc"] is not None:
            print(f"baseline accuracy : {r['baseline_class_acc']*100:.2f}%")
            print(f"accuracy cost     : {r['accuracy_cost']*100:+.2f} pts")
        print(f"concept macro-F1  : {r['concept_macro_f1']*100:.2f}%")
        print(f"faithfulness flip : "
              f"{r['faithfulness']['mean_flip_rate_when_concept_removed']*100:.1f}%")
        if r.get("explanation_metrics"):
            em = r["explanation_metrics"]
            cs, rc, st = (em["comprehensiveness_sufficiency"],
                          em["rule_correctness"], em["stability"])
            print(f"comprehensiveness : {cs['comprehensiveness']:.3f}  "
                  f"| sufficiency: {cs['sufficiency']:.3f}")
            print(f"rule correctness  : F1 {rc['all_predictions']['f1']*100:.1f}%  "
                  f"| exact-match {rc['all_predictions']['exact_match']*100:.1f}%")
            print(f"stability         : pred {st['prediction_consistency']*100:.1f}%  "
                  f"| expl-Jaccard {st['explanation_jaccard']*100:.1f}%")
            if "simulatability" in em:
                sim = em["simulatability"]
                print(f"simulatability    : {sim['simulatability_binary']*100:.1f}% "
                      "(binary concepts -> model prediction)")
            if "sanity_check" in em:
                sc = em["sanity_check"]
                print(f"sanity check      : {'PASSED' if sc['passed'] else 'NOT PASSED'} "
                      f"(acc {sc['trained']['accuracy']*100:.1f}% → "
                      f"{sc['randomized_lnn']['accuracy']*100:.1f}% on LNN randomise)")
        if r.get("surrogate"):
            s = r["surrogate"]
            print(f"surrogate fidelity: {s['fidelity']*100:.2f}%  "
                  f"(LNN reproduces the black-box CNN's decisions)")
            print(f"  surrogate acc   : {s['surrogate_acc']*100:.2f}%  "
                  f"| black-box CNN acc: {s['teacher_acc']*100:.2f}%")
        if r.get("layerwise"):
            lw = r["layerwise"]
            keys = list(lw["stage_acc"])
            shown = [keys[0], keys[len(keys) // 2], keys[-1]]
            accs = "  ".join(f"{s}={lw['stage_acc'][s]*100:.1f}%" for s in shown)
            print(f"layer-by-layer acc (17 conv layers; {', '.join(shown)}): "
                  f"{accs}")
    elif args.command == "explain":
        _run_explain(cfg, args)


def _print_device() -> None:
    import torch
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA build   : {torch.version.cuda or 'none (CPU-only build)'}")
    avail = torch.cuda.is_available()
    print(f"CUDA available: {avail}")
    if avail:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  ({p.total_memory / 1e9:.1f} GB, "
                  f"cc {p.major}.{p.minor})")
        print(f"cuDNN        : {torch.backends.cudnn.version()}")
        print("=> training and inference will run on the GPU.")
    else:
        print("=> no CUDA GPU visible to PyTorch; runs will use the CPU.")
        print("   For GPU, install a CUDA build of PyTorch, e.g.:")
        print("   pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu124")


def _print_concepts() -> None:
    from src.concepts import (CLASS_CONCEPTS, CLASS_NAMES, CONCEPT_GROUPS,
                              NUM_CLASSES, NUM_CONCEPTS)
    print(f"{NUM_CLASSES} classes, {NUM_CONCEPTS} concepts\n")
    for group, names in CONCEPT_GROUPS.items():
        print(f"[{group}] {', '.join(names)}")
    print("\nclass -> concepts:")
    for c in range(NUM_CLASSES):
        print(f"  {c:2d} {CLASS_NAMES[c]:38s} : {', '.join(CLASS_CONCEPTS[c])}")


def _run_explain(cfg: Config, args) -> None:
    import torch

    from src.concepts import CLASS_NAMES
    from src.data import make_dataloaders
    from src.evaluate import load_hybrid, load_layerwise, load_surrogate
    from src.explain import (explain_layerwise, explain_sample,
                             faithfulness_intervention)
    from src.utils import get_device

    device = get_device(cfg.device)
    surrogate_mode = args.mode == "surrogate"
    layerwise_mode = args.mode == "layerwise"

    if layerwise_mode:
        ckpt = args.checkpoint or f"{cfg.out_dir}/layerwise.pt"
        model = load_layerwise(ckpt, device)
    elif surrogate_mode:
        ckpt = args.checkpoint or f"{cfg.out_dir}/surrogate.pt"
        model = load_surrogate(ckpt, device)
    else:
        ckpt = args.checkpoint or f"{cfg.out_dir}/hybrid.pt"
        model = load_hybrid(ckpt, device)

    _, test_loader = make_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers,
        cfg.subset_fraction, cfg.seed, cfg.download)
    images, labels, _ = next(iter(test_loader))

    if layerwise_mode:
        for i in range(min(args.num, images.size(0))):
            exp = explain_layerwise(model, images[i], device)
            print("-" * 78)
            print(f"true class : {CLASS_NAMES[int(labels[i].item())]}")
            for step in exp["trace"]:
                print("  " + step["sentence"])
            print(f"  => final: {exp['final_name']}")
        return

    teacher_pred = None
    if surrogate_mode:
        with torch.no_grad():
            teacher_pred = model(images.to(device)).teacher_logits.argmax(1).cpu()
    for i in range(min(args.num, images.size(0))):
        exp = explain_sample(model, images[i], device)
        faith = faithfulness_intervention(model, images[i], device)
        print("-" * 78)
        print(f"true class : {CLASS_NAMES[int(labels[i].item())]}")
        if surrogate_mode:
            t = int(teacher_pred[i].item())
            agree = "agrees" if exp["pred"] == t else "DISAGREES"
            print(f"black-box CNN says: {CLASS_NAMES[t]}  -> LNN {agree}")
        print(exp["sentence"])
        top = faith["effects"][0] if faith["effects"] else None
        if top:
            print(f"faithfulness: removing '{top['concept']}' drops truth by "
                  f"{top['truth_drop']:.2f}"
                  + (f" -> flips to '{top['new_pred_name']}'"
                     if top["flipped"] else ""))


if __name__ == "__main__":
    main()

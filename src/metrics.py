"""Explanation-quality metrics for the CNN+LNN models.

Three complementary families, all operating on the concept bottleneck so they
apply to the intrinsic hybrid, the post-hoc surrogate, and the layer-wise model
alike:

* **Comprehensiveness & sufficiency** (faithfulness) -- ERASER-style curves
  adapted to concepts. Rank the concepts by how much removing them changes the
  predicted-class probability, then:
    - *comprehensiveness* = drop when the top-k concepts are removed (high = the
      cited concepts really drive the decision);
    - *sufficiency* = drop when only the top-k concepts are kept (low = those
      concepts alone are enough to reproduce the decision).

* **Rule correctness** (plausibility) -- GTSRB uniquely gives us the ground-truth
  concept set of every class, so for each prediction we can check whether the
  concepts the model detected match the predicted class's *known* rule
  (precision / recall / exact-match of the explanation against the real rule).

* **Explanation stability** (robustness) -- perturb an image slightly and check
  that the prediction and the set of supporting concepts stay the same
  (Jaccard overlap).

Each function returns plain Python floats so results drop straight into the JSON
report.
"""

from __future__ import annotations

import copy

import torch

from .concepts import NUM_CLASSES, concept_matrix


def _final_head(model, x: torch.Tensor):
    """Return (concept_probs [B,K], class_logits [B,C], reasoner) for the model's
    final decision, regardless of which model type it is."""
    out = model(x)
    if hasattr(model, "reasoners"):        # layer-wise: use the deepest layer
        last = model.stages[-1]
        return out.concept_probs[last], out.class_logits[last], model.reasoners[last]
    return out.concept_probs, out.class_logits, model.reasoner


def _class_prob(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # index tensor must live on the same device as ``logits`` (matters on CUDA)
    rows = torch.arange(logits.size(0), device=logits.device)
    return torch.softmax(logits, dim=1)[rows, idx]


# --------------------------------------------------------------------------- #
# Faithfulness: comprehensiveness & sufficiency
# --------------------------------------------------------------------------- #
@torch.no_grad()
def comprehensiveness_sufficiency(model, loader, device, n: int = 500,
                                  ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    model.eval()
    comp_tot = suff_tot = base_tot = 0.0
    count = 0
    for images, _, _ in loader:
        images = images.to(device)
        probs, logits, reasoner = _final_head(model, images)
        B = probs.size(0)
        pred = logits.argmax(1)
        temp = model.temperature

        def prob_of(p):  # class prob after reasoning over concept probs p
            return _class_prob(reasoner.raw_score(p) * temp, pred)

        p_full = _class_prob(logits, pred)

        # per-concept importance = drop in predicted-class prob when ablated
        delta = torch.zeros_like(probs)
        for k in range(probs.size(1)):
            p2 = probs.clone()
            p2[:, k] = 0.0
            delta[:, k] = p_full - prob_of(p2)
        order = delta.argsort(dim=1, descending=True)

        comp_s = torch.zeros(B, device=images.device)
        suff_s = torch.zeros(B, device=images.device)
        for kk in ks:
            top = order[:, :kk]
            p_rm = probs.clone()
            p_rm.scatter_(1, top, 0.0)                    # remove top-k
            comp_s += p_full - prob_of(p_rm)
            p_keep = torch.zeros_like(probs)
            p_keep.scatter_(1, top, probs.gather(1, top))  # keep only top-k
            suff_s += p_full - prob_of(p_keep)
        comp_s /= len(ks)
        suff_s /= len(ks)

        comp_tot += comp_s.sum().item()
        suff_tot += suff_s.sum().item()
        base_tot += p_full.sum().item()
        count += B
        if count >= n:
            break
    count = max(count, 1)
    return {
        "comprehensiveness": comp_tot / count,   # higher is better
        "sufficiency": suff_tot / count,          # lower (closer to 0) is better
        "base_confidence": base_tot / count,
        "ks": list(ks),
        "n": count,
    }


# --------------------------------------------------------------------------- #
# Plausibility: rule correctness vs the ground-truth GTSRB rules
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rule_correctness(model, loader, device, n: int = 1000,
                     threshold: float = 0.5) -> dict:
    model.eval()
    M = torch.from_numpy(concept_matrix()).to(device)  # [C, K]

    def _acc():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": 0.0, "n": 0}

    allm, corr = _acc(), _acc()
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        probs, logits, _ = _final_head(model, images)
        pred = logits.argmax(1)
        detected = probs > threshold                     # [B, K]
        gt = M[pred] > 0.5                               # predicted class's rule
        inter = (detected & gt).sum(1).float()
        prec = inter / detected.sum(1).float().clamp(min=1)
        rec = inter / gt.sum(1).float().clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
        exact = (detected == gt).all(1).float()

        for key, val in (("precision", prec), ("recall", rec),
                         ("f1", f1), ("exact", exact)):
            allm[key] += val.sum().item()
        allm["n"] += images.size(0)

        mask = pred == labels
        if mask.any():
            for key, val in (("precision", prec), ("recall", rec),
                             ("f1", f1), ("exact", exact)):
                corr[key] += val[mask].sum().item()
            corr["n"] += int(mask.sum().item())
        if allm["n"] >= n:
            break

    def _finalize(d):
        m = max(d["n"], 1)
        return {"precision": d["precision"] / m, "recall": d["recall"] / m,
                "f1": d["f1"] / m, "exact_match": d["exact"] / m, "n": d["n"]}

    return {"all_predictions": _finalize(allm),
            "correct_predictions": _finalize(corr)}


# --------------------------------------------------------------------------- #
# Robustness: explanation stability under small perturbations
# --------------------------------------------------------------------------- #
@torch.no_grad()
def explanation_stability(model, loader, device, n: int = 80, n_views: int = 4,
                          noise: float = 0.04, brightness: float = 0.1,
                          seed: int = 0) -> dict:
    model.eval()
    M = torch.from_numpy(concept_matrix()).to(device)
    torch.manual_seed(seed)
    cons_tot = jac_tot = 0.0
    count = 0
    for images, _, _ in loader:
        for i in range(images.size(0)):
            x = images[i:i + 1].to(device)
            views = [x]
            for _ in range(n_views - 1):
                b = 1.0 + (torch.rand(1, device=device) * 2 - 1) * brightness
                views.append(x * b + torch.randn_like(x) * noise)
            batch = torch.cat(views, 0)
            probs, logits, _ = _final_head(model, batch)
            preds = logits.argmax(1)
            support = (probs > 0.5) & (M[preds] > 0.5)   # cited supporting concepts
            ref = support[0]

            cons = (preds == preds[0]).float().mean().item()
            js = []
            for v in range(1, batch.size(0)):
                inter = (support[v] & ref).sum().item()
                union = (support[v] | ref).sum().item()
                js.append(1.0 if union == 0 else inter / union)
            jac_tot += sum(js) / max(len(js), 1)
            cons_tot += cons
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    count = max(count, 1)
    return {
        "prediction_consistency": cons_tot / count,   # 1.0 = perturbation never flips the class
        "explanation_jaccard": jac_tot / count,        # 1.0 = identical supporting concepts
        "n": count,
        "n_views": n_views,
    }


# --------------------------------------------------------------------------- #
# Sanity-check control (model-parameter randomization test, Adebayo et al. 2018)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _quick_accuracy(model, loader, device, n: int) -> float:
    model.eval()
    correct = total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        _, logits, _ = _final_head(model, images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if total >= n:
            break
    return correct / max(total, 1)


def _randomize_reasoners(model) -> None:
    """Overwrite the LNN reasoner weights with random values (keeps the trained
    CNN + concept head intact) -- the control condition for the sanity check."""
    reasoners = ([model.reasoner] if hasattr(model, "reasoner")
                 else list(model.reasoners.values()))
    with torch.no_grad():
        for r in reasoners:
            r.raw_w.data.normal_(0.0, 1.0)
            r.bias.data.normal_(1.0, 0.5)


def sanity_check(model, loader, device, n: int = 500) -> dict:
    """Model-parameter randomization test: if the explanation metrics are truly
    tied to the learned logic, randomizing the LNN weights should make them
    collapse. Reports the metrics before and after randomizing the reasoner."""
    def _snapshot(m):
        return {
            "accuracy": _quick_accuracy(m, loader, device, n),
            "comprehensiveness": comprehensiveness_sufficiency(
                m, loader, device, n=min(300, n))["comprehensiveness"],
            "rule_f1": rule_correctness(
                m, loader, device, n=n)["all_predictions"]["f1"],
        }

    trained = _snapshot(model)
    randomized = copy.deepcopy(model)
    _randomize_reasoners(randomized)
    randomized = randomized.to(device)
    rand = _snapshot(randomized)

    deltas = {k: trained[k] - rand[k] for k in trained}
    # a healthy model loses most of its accuracy when the logic is randomized
    passed = bool(rand["accuracy"] < 0.5 * trained["accuracy"]
                  and deltas["comprehensiveness"] > 0)
    return {
        "trained": trained,
        "randomized_lnn": rand,
        "deltas": deltas,
        "chance_level": 1.0 / NUM_CLASSES,
        "passed": passed,
    }


# --------------------------------------------------------------------------- #
# Simulatability (can a simple, human-graspable proxy reproduce the decision
# from the concepts alone?)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _collect_concepts_and_preds(model, loader, device, n):
    model.eval()
    Xs, Xb, Y, count = [], [], [], 0
    for images, _, _ in loader:
        images = images.to(device)
        probs, logits, _ = _final_head(model, images)
        pred = logits.argmax(1)
        Xs.append(probs.cpu())
        Xb.append((probs > 0.5).float().cpu())
        Y.append(pred.cpu())
        count += images.size(0)
        if count >= n:
            break
    return torch.cat(Xs), torch.cat(Xb), torch.cat(Y)


def simulatability(model, loader, device, n: int = 2000, steps: int = 300,
                   lr: float = 0.05) -> dict:
    """Train a simple linear proxy to predict the *model's own prediction* from
    the concepts, and measure held-out agreement. High agreement = the concept
    explanation is a sufficient basis to simulate the model's decisions.

    Reported for binary concepts (what a human reads: present / absent) and for
    the soft concept probabilities (reference upper bound).
    """
    Xs, Xb, Y = _collect_concepts_and_preds(model, loader, device, n)
    m = Xb.size(0)
    if m < 4:
        return {"simulatability_binary": 0.0, "simulatability_soft": 0.0, "n": m}
    idx = torch.randperm(m)
    half = m // 2
    tr, te = idx[:half], idx[half:]

    def _train_proxy(X):
        clf = torch.nn.Linear(X.size(1), NUM_CLASSES)
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(clf(X[tr]), Y[tr])
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (clf(X[te]).argmax(1) == Y[te]).float().mean().item()

    return {
        "simulatability_binary": _train_proxy(Xb),  # human-readable concepts
        "simulatability_soft": _train_proxy(Xs),    # soft-probability reference
        "n": m,
        "proxy": "linear classifier: concepts -> model prediction",
    }

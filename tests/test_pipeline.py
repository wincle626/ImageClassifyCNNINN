"""Fast, data-free unit tests for the core logic. Run with: pytest -q"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.concepts import (CLASS_CONCEPTS, NUM_CLASSES, NUM_CONCEPTS,
                          concept_matrix)
from src.explain import (explain_layerwise, explain_sample,
                         faithfulness_intervention)
from src.models import (BaselineCNN, HybridCNNLNN, LayerwiseCNNLNN,
                        PostHocLNNSurrogate, ResNet18Backbone)
from src.models.lnn_layer import LNNReasoningLayer


def test_concept_vectors_unique():
    m = concept_matrix()
    assert m.shape == (NUM_CLASSES, NUM_CONCEPTS)
    keys = {tuple(row.tolist()) for row in m}
    assert len(keys) == NUM_CLASSES  # every class distinct in concept space


def test_lnn_recovers_class_from_perfect_concepts():
    """With the ground-truth concept vector fed in, the LNN's argmax truth
    value must recover the correct class -- i.e. the rules are separable."""
    m = torch.from_numpy(concept_matrix())
    layer = LNNReasoningLayer(m)
    truth = layer(m)  # feed each class's own ground-truth concept vector
    assert torch.equal(truth.argmax(dim=1), torch.arange(NUM_CLASSES))


def test_perfect_match_truth_is_one():
    m = torch.from_numpy(concept_matrix())
    layer = LNNReasoningLayer(m)
    truth = layer(m)
    # the diagonal (class c evaluated on class c's concepts) should be ~1
    diag = truth[torch.arange(NUM_CLASSES), torch.arange(NUM_CLASSES)]
    assert torch.all(diag > 0.99)


def test_hybrid_forward_shapes():
    m = torch.from_numpy(concept_matrix())
    model = HybridCNNLNN(m)
    x = torch.randn(4, 3, 48, 48)
    out = model(x)
    assert out.concept_probs.shape == (4, NUM_CONCEPTS)
    assert out.class_truth.shape == (4, NUM_CLASSES)
    assert torch.all((out.concept_probs >= 0) & (out.concept_probs <= 1))
    assert torch.all((out.class_truth >= 0) & (out.class_truth <= 1))


def test_explanation_and_faithfulness():
    m = torch.from_numpy(concept_matrix())
    model = HybridCNNLNN(m)
    device = torch.device("cpu")
    x = torch.randn(3, 48, 48)
    exp = explain_sample(model, x, device)
    assert "because" in exp["sentence"]
    assert 0.0 <= exp["truth"] <= 1.0
    faith = faithfulness_intervention(model, x, device)
    # removing a required concept must never *increase* the class truth value
    assert all(e["truth_drop"] >= -1e-6 for e in faith["effects"])


def test_surrogate_freezes_teacher_and_flows_to_probe():
    """The post-hoc surrogate must not change the black-box CNN (teacher frozen),
    but gradients must reach its own concept probe + reasoner."""
    m = torch.from_numpy(concept_matrix())
    teacher = BaselineCNN(NUM_CLASSES)
    surrogate = PostHocLNNSurrogate(teacher, m)

    # teacher params frozen, probe/reasoner trainable
    assert all(not p.requires_grad for p in surrogate.teacher.parameters())
    assert any(p.requires_grad for p in surrogate.concept_head.parameters())

    x = torch.randn(4, 3, 48, 48)
    out = surrogate(x)
    assert out.teacher_logits.shape == (4, NUM_CLASSES)
    assert out.class_logits.shape == (4, NUM_CLASSES)

    teacher_before = next(surrogate.teacher.classifier.parameters()).clone()
    out.class_logits.sum().backward()
    # teacher got no gradient
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in surrogate.teacher.parameters())
    # probe/reasoner did
    trainable_grads = [p.grad for p in surrogate.concept_head.parameters()
                       if p.grad is not None]
    assert trainable_grads and any(g.abs().sum() > 0 for g in trainable_grads)
    # teacher weights unchanged (no optimiser step, but confirm identity anyway)
    torch.testing.assert_close(
        next(surrogate.teacher.classifier.parameters()), teacher_before)


def test_resnet18_backbone_exposes_stages():
    bb = ResNet18Backbone(pretrained=False)
    x = torch.randn(2, 3, 48, 48)
    stages = bb.forward_stages(x)
    assert set(stages) == {"layer1", "layer2", "layer3", "layer4"}
    assert stages["layer1"].shape == (2, 64)
    assert stages["layer4"].shape == (2, 512)
    assert bb(x).shape == (2, 512)  # final embedding drop-in


def test_hybrid_accepts_resnet18():
    m = torch.from_numpy(concept_matrix())
    model = HybridCNNLNN(m, backbone="resnet18", pretrained=False)
    out = model(torch.randn(2, 3, 48, 48))
    assert out.class_truth.shape == (2, NUM_CLASSES)
    assert out.concept_probs.shape == (2, NUM_CONCEPTS)


def test_resnet18_has_18_weight_layers():
    from src.models.layerwise import resnet18_weight_layers
    layers = resnet18_weight_layers()
    assert len(layers) == 18                 # the "18" in ResNet-18
    assert layers[0][0] == "L1"              # stem conv
    assert layers[-1][0] == "L18"            # fc classifier


def test_layerwise_taps_every_conv_layer():
    m = torch.from_numpy(concept_matrix())
    model = LayerwiseCNNLNN(m, pretrained=False)
    # 17 convolutional layers get a probe + LNN (fc is layer 18, replaced)
    assert len(model.stages) == 17
    assert len(model.probes) == 17 and len(model.reasoners) == 17

    x = torch.randn(2, 3, 48, 48)
    out = model(x)
    for s in model.stages:
        assert out.concept_probs[s].shape == (2, NUM_CONCEPTS)
        assert out.class_logits[s].shape == (2, NUM_CLASSES)
    # final prediction is the deepest conv layer's LNN
    assert torch.equal(out.final_logits, out.class_logits[model.stages[-1]])

    # deep supervision: gradient reaches the very first conv layer's probe
    out.class_logits[model.stages[0]].sum().backward()
    first = model.stages[0]
    g = [p.grad for p in model.probes[first].parameters() if p.grad is not None]
    assert g and any(t.abs().sum() > 0 for t in g)

    # layer-by-layer trace has one step per conv layer
    exp = explain_layerwise(model, x[0], torch.device("cpu"))
    assert [t["stage"] for t in exp["trace"]] == model.stages


def test_explanation_metrics_run():
    import math

    from torch.utils.data import DataLoader, TensorDataset

    from src.metrics import (comprehensiveness_sufficiency,
                             explanation_stability, rule_correctness)

    m = torch.from_numpy(concept_matrix())
    model = HybridCNNLNN(m)  # simple backbone, random weights is fine for shapes
    B = 12
    ds = TensorDataset(torch.randn(B, 3, 48, 48),
                       torch.randint(0, NUM_CLASSES, (B,)),
                       torch.zeros(B, NUM_CONCEPTS))
    loader = DataLoader(ds, batch_size=6)
    dev = torch.device("cpu")

    cs = comprehensiveness_sufficiency(model, loader, dev, n=12, ks=(1, 3))
    assert cs["n"] == 12
    assert math.isfinite(cs["comprehensiveness"]) and math.isfinite(cs["sufficiency"])

    rc = rule_correctness(model, loader, dev, n=12)
    for group in ("all_predictions", "correct_predictions"):
        for key in ("precision", "recall", "f1", "exact_match"):
            assert 0.0 <= rc[group][key] <= 1.0

    st = explanation_stability(model, loader, dev, n=6, n_views=3)
    assert 0.0 <= st["prediction_consistency"] <= 1.0
    assert 0.0 <= st["explanation_jaccard"] <= 1.0


def test_sanity_check_and_simulatability_run():
    from torch.utils.data import DataLoader, TensorDataset

    from src.metrics import sanity_check, simulatability

    m = torch.from_numpy(concept_matrix())
    model = HybridCNNLNN(m)
    B = 24
    ds = TensorDataset(torch.randn(B, 3, 48, 48),
                       torch.randint(0, NUM_CLASSES, (B,)),
                       torch.zeros(B, NUM_CONCEPTS))
    loader = DataLoader(ds, batch_size=8)
    dev = torch.device("cpu")

    sc = sanity_check(model, loader, dev, n=24)
    assert set(sc) >= {"trained", "randomized_lnn", "deltas", "passed"}
    assert 0.0 <= sc["trained"]["accuracy"] <= 1.0
    assert isinstance(sc["passed"], bool)

    sim = simulatability(model, loader, dev, n=24, steps=20)
    assert 0.0 <= sim["simulatability_binary"] <= 1.0
    assert 0.0 <= sim["simulatability_soft"] <= 1.0


def test_gradients_flow_through_lnn():
    m = torch.from_numpy(concept_matrix())
    model = HybridCNNLNN(m)
    x = torch.randn(2, 3, 48, 48)
    out = model(x)
    loss = out.class_logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)

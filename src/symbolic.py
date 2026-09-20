"""Optional bridge to IBM's Logical Neural Network library (``pip install lnn``).

The main pipeline uses the differentiable LNN layer in
:mod:`src.models.lnn_layer` (pure PyTorch, always available).  This module is a
*demonstration* that the same rules can be executed as **hard symbolic
inference** by IBM's official ``lnn`` package: we take the CNN's detected
concepts, threshold them to logical facts, and let an IBM LNN prove the class.

It is intentionally isolated and imported lazily so the project has no hard
dependency on ``lnn`` (which pins specific versions and can be awkward to
install).  If ``lnn`` is not installed, :func:`symbolic_explanation` raises a
clear, actionable error.
"""

from __future__ import annotations

from .concepts import CLASS_CONCEPTS, CLASS_NAMES, CONCEPTS


def _require_lnn():
    try:
        import lnn  # noqa: F401
        return lnn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "IBM's LNN library is not available. Install it with "
            "`pip install lnn` to use the symbolic-inference demo. "
            "The core CNN+LNN pipeline does not require it."
        ) from exc


def symbolic_explanation(concept_probs, cls: int, threshold: float = 0.5) -> dict:
    """Run IBM-LNN symbolic inference for one class rule.

    Args:
        concept_probs: 1-D iterable of predicted concept probabilities
            (length == number of concepts, in :data:`src.concepts.CONCEPTS`
            order).
        cls: the class whose rule to evaluate.
        threshold: probability above which a concept is treated as TRUE.

    Returns a dict with the class truth-value *bounds* proven by the LNN and the
    facts that were fed in.
    """
    lnn = _require_lnn()
    from lnn import And, Model, Proposition, Fact, World

    model = Model()
    props = {}
    facts = {}
    for name in CLASS_CONCEPTS[cls]:
        k = CONCEPTS.index(name)
        p = float(concept_probs[k])
        prop = Proposition(name)
        props[name] = prop
        facts[prop] = Fact.TRUE if p >= threshold else Fact.FALSE

    rule = And(*props.values(), world=World.OPEN)
    model.add_knowledge(rule)
    model.add_data(facts)
    model.infer()

    bounds = rule.state()  # a Fact enum (e.g. Fact.TRUE / Fact.FALSE / UNKNOWN)
    return {
        "class": cls,
        "class_name": CLASS_NAMES[cls],
        "rule": " AND ".join(CLASS_CONCEPTS[cls]),
        "proven_state": str(bounds),
        "facts": {name: str(facts[props[name]]) for name in props},
    }

import numpy as np
import pytest

from mint.domains import REGISTRY, annotate, get_domain, sigmoid


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_sample_shapes_and_determinism(name):
    d = get_domain(name)
    a = d.sample(64, np.random.default_rng(3))
    b = d.sample(64, np.random.default_rng(3))
    assert a.raw.shape == (64, d.spec.n_inputs)
    assert np.array_equal(a.raw, b.raw), "sampling must be seed-deterministic"
    assert set(np.unique(a.y)) <= {0, 1}
    assert np.allclose(a.logit, d.logit(a.raw))


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_spurious_inputs_have_zero_causal_weight(name):
    """The decoy input predicts the outcome but must not cause it."""
    d = get_domain(name)
    raw = d.sample(256, np.random.default_rng(4)).raw
    base = d.logit(raw)
    perturbed = np.array(raw, copy=True)
    for i in d.spec.spurious_inputs:
        if d.spec.kind == "tabular":
            perturbed[:, i] = np.random.default_rng(5).normal(size=len(raw)) * 3
        else:
            perturbed[perturbed == i] = 0
    assert np.allclose(d.logit(perturbed), base), "spurious input leaked into the DGP"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_spurious_inputs_are_predictive(name):
    """...and it must actually correlate, otherwise no decoy units appear."""
    d = get_domain(name)
    b = d.sample(4000, np.random.default_rng(6), spurious_strength=2.0)
    i = d.spec.spurious_inputs[0]
    v = b.raw[:, i] if d.spec.kind == "tabular" else (b.raw == i).any(1).astype(float)
    assert abs(np.corrcoef(v, b.y)[0, 1]) > 0.2


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_annotation_matches_executed_counterfactual(name):
    d = get_domain(name)
    raw = d.sample(128, np.random.default_rng(7)).raw
    ann = annotate(d, raw)
    assert ann.factor_effect.shape == (128, d.spec.n_factors)
    for f in range(d.spec.n_factors):
        expected = d.logit(d.do_neutral(raw, f)) - d.logit(raw)
        assert np.allclose(ann.factor_effect[:, f], expected)
    top = np.abs(ann.factor_effect).argmax(1)
    assert np.array_equal(top, ann.factor_id)
    assert (ann.margin >= -1e-9).all()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_neutralising_a_factor_is_idempotent(name):
    d = get_domain(name)
    raw = d.sample(64, np.random.default_rng(8)).raw
    for f in range(d.spec.n_factors):
        once = d.do_neutral(raw, f)
        assert np.allclose(d.logit(d.do_neutral(once, f)), d.logit(once))


def test_factor_vocabularies_are_disjoint_across_domains():
    """Transfer would be trivial if domains shared explanation labels."""
    vocabs = [{f.name for f in get_domain(n).spec.factors} for n in sorted(REGISTRY)]
    for i, a in enumerate(vocabs):
        for b in vocabs[i + 1:]:
            assert not (a & b)

"""The semantic layer: no localisation head, no pattern leak into the control."""
import numpy as np
import pytest
import torch

from mint.semantic.encoding import D_GLOBAL, D_SIGNATURE, D_UNIT, GLOBAL_FEATURES, UNIT_FEATURES
from mint.semantic.model import (GLOBAL_PATTERN_FEATURES, PATTERN_FEATURES, SemanticInterpreter)


def _inputs(n=3, u=24, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, u, D_UNIT, generator=g), torch.randn(n, D_GLOBAL, generator=g),
            torch.ones(n, u, dtype=torch.bool), torch.randn(n, k, D_SIGNATURE, generator=g),
            torch.ones(n, k, dtype=torch.bool), torch.rand(n, k, u, generator=g))


def test_the_model_has_no_localisation_head():
    """Localisation left the learned model when the gate failed; it must not
    creep back as an auxiliary output."""
    m = SemanticInterpreter(d_model=32, n_layers=1, n_heads=2).eval()
    with torch.no_grad():
        out = m(*_inputs())
    assert set(out) == {"name", "attention"}
    assert not any(n.startswith(("causal", "mediation", "decoy", "effect_mass"))
                   for n, _ in m.named_modules())


def test_blind_control_sees_no_trace_of_the_causal_pattern():
    """Masking the per-unit features alone leaves `log_fo_mass` in the global
    context, which is a summary of the same first-order term."""
    m = SemanticInterpreter(d_model=32, n_layers=1, n_heads=2, blind=True).eval()
    ux, gx, mk, sx, sm, lo = _inputs()
    ux2 = ux.clone()
    for f in PATTERN_FEATURES:
        ux2[:, :, UNIT_FEATURES.index(f)] += 7.0
    gx2 = gx.clone()
    for f in GLOBAL_PATTERN_FEATURES:
        gx2[:, GLOBAL_FEATURES.index(f)] += 7.0
    with torch.no_grad():
        a = m(ux, gx, mk, sx, sm, lo)["name"]
        b = m(ux2, gx2, mk, sx, sm, lo)["name"]
    assert torch.allclose(a, b, atol=1e-6)
    # and it must still see architecture shape and B's own output
    gx3 = gx.clone(); gx3[:, GLOBAL_FEATURES.index("state_prob")] += 7.0
    with torch.no_grad():
        assert not torch.allclose(a, m(ux, gx3, mk, sx, sm, lo)["name"], atol=1e-5)


def test_naming_works_for_any_number_of_candidates():
    m = SemanticInterpreter(d_model=32, n_layers=1, n_heads=2).eval()
    for k in (2, 3, 4, 7, 11):
        ux, gx, mk, sx, sm, lo = _inputs(k=k)
        with torch.no_grad():
            assert m(ux, gx, mk, sx, sm, lo)["name"].shape == (3, k)


def test_candidate_order_does_not_matter():
    m = SemanticInterpreter(d_model=32, n_layers=1, n_heads=2).eval()
    ux, gx, mk, sx, sm, lo = _inputs(k=4)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = m(ux, gx, mk, sx, sm, lo)["name"]
        b = m(ux, gx, mk, sx[:, perm], sm[:, perm], lo[:, perm])["name"]
    assert torch.allclose(a[:, perm], b, atol=1e-5)


def test_the_scored_state_counterfactual_is_never_an_input():
    """The whole design rests on this: with it, naming is a lookup."""
    import inspect

    from mint.semantic import encoding
    src = inspect.getsource(encoding.causal_pattern)
    assert "first_order_cf" not in src, "the factual pattern must not read the counterfactual"
    for fn in (encoding.candidate_signatures, encoding.candidate_locus):
        sig = inspect.signature(fn).parameters
        assert "ref_idx" in sig, f"{fn.__name__} must be restricted to reference states"

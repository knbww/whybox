import numpy as np
import torch

from mint.encoding import D_GLOBAL, D_UNIT
from mint.interpreter.model import FEATURE_GROUPS, INTERNAL_FEATURES, Interpreter, feature_mask


def _inputs(n=4, u=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, u, D_UNIT, generator=g), torch.randn(n, D_GLOBAL, generator=g),
            torch.ones(n, u, dtype=torch.bool))


def test_unit_permutation_equivariance():
    m = Interpreter(d_model=32, n_layers=2, n_heads=2).eval()
    ux, gx, mk = _inputs()
    perm = torch.randperm(ux.shape[1])
    with torch.no_grad():
        a, b = m(ux, gx, mk), m(ux[:, perm], gx, mk)
    assert torch.allclose(a["causal"][:, perm], b["causal"], atol=1e-5)
    assert torch.allclose(a["mediation"][:, perm], b["mediation"], atol=1e-5)
    assert torch.allclose(a["effect_mass"], b["effect_mass"], atol=1e-5)


def test_padding_does_not_change_real_units():
    """Target models have different unit counts; padding must be inert."""
    m = Interpreter(d_model=32, n_layers=2, n_heads=2).eval()
    ux, gx, mk = _inputs(u=12)
    ux2 = torch.cat([ux, torch.randn(ux.shape[0], 7, D_UNIT)], dim=1)
    mk2 = torch.cat([mk, torch.zeros(mk.shape[0], 7, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        a, b = m(ux, gx, mk), m(ux2, gx, mk2)
    assert torch.allclose(a["causal"], b["causal"][:, :12], atol=1e-5)


def test_output_only_arm_really_is_blind_to_internals():
    m = Interpreter(d_model=32, n_layers=2, n_heads=2, drop_features=INTERNAL_FEATURES).eval()
    ux, gx, mk = _inputs()
    ux2 = ux.clone()
    keep = feature_mask(INTERNAL_FEATURES) == 0
    ux2[:, :, keep] = torch.randn_like(ux2[:, :, keep]) * 5
    with torch.no_grad():
        assert torch.allclose(m(ux, gx, mk)["causal"], m(ux2, gx, mk)["causal"], atol=1e-6)


def test_feature_groups_are_known_features():
    from mint.encoding import UNIT_FEATURES
    for group, feats in FEATURE_GROUPS.items():
        assert set(feats) <= set(UNIT_FEATURES), group


def test_vocabulary_head_is_separable_from_the_procedure():
    m = Interpreter(d_model=32, n_layers=2, n_heads=2, factor_heads={"skirmish": 7})
    proc = {id(p) for p in m.procedure_parameters()}
    assert not (proc & {id(p) for p in m.factor_heads.parameters()})
    assert len(proc) + len(list(m.factor_heads.parameters())) == len(list(m.parameters()))

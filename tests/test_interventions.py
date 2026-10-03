import numpy as np
import torch

from mint.causal import UnitLayout, counterfactual_acts, ground_truth, subset_patch_effect
from mint.domains import get_domain
from mint.models import TargetConfig, build_target, train_target


def _fixture(name, arch, seed=0):
    d = get_domain(name)
    tt = train_target(d, TargetConfig(name, arch, seed, 1.0, n_train=1500, epochs=8),
                      np.random.default_rng(seed))
    return d, tt.model


def test_full_patch_reproduces_the_counterfactual_exactly():
    """For a feed-forward B, patching every unit to its counterfactual value must
    reproduce the counterfactual output: sufficiency of the full set is 1."""
    d, m = _fixture("skirmish", {"widths": (24, 12)})
    layout = UnitLayout.of(m)
    raw = d.sample(32, np.random.default_rng(1)).raw
    x = torch.as_tensor(d.to_model_input(raw))
    with torch.no_grad():
        base = m.logits(x).numpy()
    cf = counterfactual_acts(m, d, raw)[:, :, 0]
    full = np.ones((32, layout.n_units), dtype=bool)
    got = subset_patch_effect(m, layout, x, cf, full, base) + base
    with torch.no_grad():
        want = m.logits(torch.as_tensor(d.to_model_input(d.do_neutral(raw, 0)))).numpy()
    assert np.allclose(got, want, atol=1e-4)


def test_empty_patch_is_a_no_op():
    d, m = _fixture("clinic", {"widths": (16, 16)})
    layout = UnitLayout.of(m)
    raw = d.sample(16, np.random.default_rng(2)).raw
    x = torch.as_tensor(d.to_model_input(raw))
    with torch.no_grad():
        base = m.logits(x).numpy()
    none = np.zeros((16, layout.n_units), dtype=bool)
    assert np.allclose(subset_patch_effect(m, layout, x, np.zeros_like(none, dtype=float),
                                           none, base), 0.0)


def test_single_unit_effect_matches_a_hand_rolled_ablation():
    d, m = _fixture("skirmish", {"widths": (24, 12)})
    layout = UnitLayout.of(m)
    raw, probe = d.sample(24, np.random.default_rng(3)).raw, d.sample(256, np.random.default_rng(4)).raw
    gt = ground_truth(m, d, raw, probe)
    x = torch.as_tensor(d.to_model_input(raw))
    with torch.no_grad():
        base = m.logits(x).numpy()
        for u in (0, 5, layout.n_units - 1):
            mask = np.zeros((24, layout.n_units), dtype=bool)
            mask[:, u] = True
            vals = np.repeat(gt.probe.act_mean[None], 24, axis=0)
            manual = subset_patch_effect(m, layout, x, vals, mask, base)
            assert np.allclose(manual, gt.unit_effect[:, u], atol=1e-5)


def test_ground_truth_shapes_and_finiteness():
    for name, arch in [("skirmish", {"widths": (24, 12)}),
                       ("seqworld", {"d_model": 24, "n_layers": 1, "n_heads": 4, "d_ff": 32})]:
        d, m = _fixture(name, arch)
        raw, probe = d.sample(16, np.random.default_rng(5)).raw, d.sample(128, np.random.default_rng(6)).raw
        gt = ground_truth(m, d, raw, probe)
        u, f = m.n_units(), d.spec.n_factors
        assert gt.unit_effect.shape == (16, u)
        assert gt.unit_ie.shape == (16, u, f)
        assert gt.factor_total.shape == (16, f)
        for a in (gt.unit_effect, gt.unit_ie, gt.factor_total, gt.acts, gt.grads):
            assert np.isfinite(a).all()


def test_correlation_and_causation_come_apart():
    """The whole study rests on these two rankings disagreeing."""
    d = get_domain("skirmish")
    tt = train_target(d, TargetConfig("skirmish", {"widths": (32, 16)}, 0, 2.0,
                                      n_train=3000, epochs=25), np.random.default_rng(7))
    gt = ground_truth(tt.model, d, d.sample(64, np.random.default_rng(8)).raw,
                      d.sample(512, np.random.default_rng(9)).raw)
    from mint.eval.metrics import spearman
    rho = spearman(np.abs(gt.probe.corr_out), np.abs(gt.unit_effect).mean(0))
    assert rho < 0.9, f"correlation ranking is nearly identical to the causal one (rho={rho})"


def test_full_patch_reproduces_the_counterfactual_on_a_transformer_too():
    """The invariant that used to hold only for feed-forward targets.

    Collapsing a unit to a scalar and broadcasting it back made this fail on
    every transformer -- the sufficiency denominator was unreachable, and the
    shortfall was misread as a property of the models.  See
    results/frozen/ERRATA.md item 1.
    """
    from mint.causal.interventions import capture_sites

    for name, arch in [("skirmish", {"family": "tab_transformer", "d_model": 24,
                                     "n_layers": 2, "n_heads": 4, "d_ff": 32}),
                       ("agreement", {"family": "lm", "d_model": 32, "n_layers": 2,
                                      "n_heads": 4, "d_ff": 48})]:
        d, m = _fixture(name, arch)
        layout = UnitLayout.of(m)
        raw = d.sample(16, np.random.default_rng(11)).raw
        x = torch.as_tensor(d.to_model_input(raw))
        x_cf = torch.as_tensor(d.to_model_input(d.do_neutral(raw, 0)))
        with torch.no_grad():
            want = m.logits(x_cf).numpy()
            full = torch.ones(16, layout.n_units, dtype=torch.bool)
            edits = layout.edits_from_sites(full, capture_sites(m, layout, x_cf))
            got = m.logits(x, edits=edits).numpy()
        assert np.allclose(got, want, atol=1e-4), name


def test_scalar_patching_is_degenerate_on_transformers():
    """Lock in the defect so it cannot come back unnoticed."""
    from mint.causal.interventions import capture_sites

    d, m = _fixture("agreement", {"family": "lm", "d_model": 32, "n_layers": 2,
                                  "n_heads": 4, "d_ff": 48})
    layout = UnitLayout.of(m)
    raw = d.sample(24, np.random.default_rng(12)).raw
    x = torch.as_tensor(d.to_model_input(raw))
    x_cf = torch.as_tensor(d.to_model_input(d.do_neutral(raw, 0)))
    full = torch.ones(24, layout.n_units, dtype=torch.bool)
    with torch.no_grad():
        want = m.logits(x_cf).numpy()
        sites = capture_sites(m, layout, x_cf)
        exact = m.logits(x, edits=layout.edits_from_sites(full, sites)).numpy()
        flat = layout.flatten({k: v for k, v in sites.items()}).numpy()
        scalar = m.logits(x, edits=layout.edits(full, torch.as_tensor(flat,
                                                dtype=torch.float32))).numpy()
    assert np.allclose(exact, want, atol=1e-4)
    assert not np.allclose(scalar, want, atol=1e-2)


def test_weight_norms_distinguish_sites_of_equal_width():
    """Matching parameters by shape assigned layer 0's weights to every site of
    the same width, so three unit descriptors described the wrong layer."""
    from mint.causal.interventions import probe_summary

    for name, arch in [("clinic", {"family": "mlp", "widths": (24, 24, 12)}),
                       ("agreement", {"family": "lm", "d_model": 32, "n_layers": 2,
                                      "n_heads": 4, "d_ff": 48})]:
        d, m = _fixture(name, arch)
        layout = UnitLayout.of(m)
        xp = torch.as_tensor(d.to_model_input(d.sample(64, np.random.default_rng(13)).raw))
        pr = probe_summary(m, layout, xp)
        blocks = [(s, layout.offsets[s], w) for s, w in zip(layout.sites, layout.widths)]
        same = [(a, b) for i, a in enumerate(blocks) for b in blocks[i + 1:] if a[2] == b[2]]
        assert same, f"{name}: test is vacuous without two sites of equal width"
        for (sa, oa, w), (sb, ob, _) in same:
            assert not np.allclose(pr.in_norm[oa:oa + w], pr.in_norm[ob:ob + w]), f"{sa}/{sb}"
        assert not np.allclose(pr.in_norm, 1.0), "attention heads fell through to the constant"

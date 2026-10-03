"""The explanation object: naming, localising, predicting, and being checked."""
import glob

import numpy as np
import pytest
import torch

from mint.encoding import D_FACTOR, FACTOR_FEATURES, factor_signatures
from mint.eval.explanation import evaluate_explanations, naming_baselines
from mint.interpreter.dataset import to_problem_set
from mint.interpreter.model import ENGAGEMENT_IDX, Interpreter
from mint.interpreter.train import TrainConfig, predict, train_interpreter


def _bundles(pattern: str, want: int = 12):
    for path in sorted(glob.glob(f"results/cache/{pattern}.pt"), key=lambda p: -len(p)):
        b = torch.load(path, weights_only=False)
        if len(b) == want:
            return b
    pytest.skip(f"no cached bundles matching {pattern}")


def test_signatures_carry_no_domain_content():
    banned = ("factor", "name", "label", "token", "feature", "domain")
    assert not [f for f in FACTOR_FEATURES if any(b in f for b in banned)]
    import inspect
    sig = inspect.signature(factor_signatures).parameters
    assert set(sig) == {"gt", "ref_idx", "in_play"}


def test_signature_width_is_the_same_across_domains():
    widths = set()
    for pat in ("skirmish-mlp-c*", "agreement-lm-*"):
        bs = _bundles(pat)
        s = factor_signatures(bs[0].gt, np.arange(30))
        assert s.shape[1] == D_FACTOR and np.isfinite(s).all()
        widths.add(s.shape[0])
    assert len(widths) > 1, "test is vacuous unless factor counts differ"


def test_candidate_descriptions_never_see_the_state_being_explained():
    bs = _bundles("skirmish-mlp-c*")
    ps = to_problem_set(bs[:2], ref_frac=0.375)
    n_all = len(bs[0].gt.decisive)
    n_ref = int(round(0.375 * n_all))
    assert int(ps.state_id.min()) >= n_ref, "a reference state leaked in as a problem"
    assert int(ps.state_id.max()) == n_all - 1


def test_naming_head_handles_a_different_number_of_candidates():
    """The whole point of the open vocabulary: K is not baked into the weights."""
    m = Interpreter(d_model=48, n_layers=2, n_heads=2).eval()
    ux, gx = torch.randn(2, 30, 30), torch.randn(2, 8)
    mk = torch.ones(2, 30, dtype=torch.bool)
    for k in (3, 4, 7, 11):
        fx = torch.randn(2, k, D_FACTOR)
        fl = torch.rand(2, k, 30)
        fm = torch.ones(2, k, dtype=torch.bool)
        with torch.no_grad():
            o = m(ux, gx, mk, factor_x=fx, factor_mask=fm, factor_locus=fl)
        assert o["name"].shape == (2, k) and o["carrier"].shape == (2, k, 30)


def test_locus_bias_changes_which_units_a_candidate_attends_to():
    m = Interpreter(d_model=48, n_layers=2, n_heads=2).eval()
    ux, gx = torch.randn(1, 20, 30), torch.randn(1, 8)
    mk = torch.ones(1, 20, dtype=torch.bool)
    fx, fm = torch.randn(1, 2, D_FACTOR), torch.ones(1, 2, dtype=torch.bool)
    a = torch.full((1, 2, 20), 0.05)
    b = a.clone(); b[0, 0, 7] = 5.0
    with torch.no_grad():
        oa = m(ux, gx, mk, factor_x=fx, factor_mask=fm, factor_locus=a)
        ob = m(ux, gx, mk, factor_x=fx, factor_mask=fm, factor_locus=b)
    assert ob["carrier"][0, 0, 7] > oa["carrier"][0, 0, 7]
    assert torch.allclose(oa["carrier"][0, 1], ob["carrier"][0, 1], atol=1e-4)


def test_engagement_features_are_state_conditioned():
    from mint.encoding import UNIT_FEATURES
    for i in ENGAGEMENT_IDX:
        assert UNIT_FEATURES[i].startswith(("state_", "abs_state_", "first_order",
                                            "abs_first_order"))


@pytest.mark.slow
def test_a_wrong_name_scores_near_zero_end_to_end():
    """The strict score executes the *named* counterfactual and divides by what
    actually moved B, so a confidently wrong name cannot hide."""
    bs = _bundles("skirmish-mlp-c*")
    ps = to_problem_set(bs[:3])
    m = train_interpreter(ps, TrainConfig(epochs=3, d_model=48, n_layers=2, batch_size=32),
                          {"skirmish": 7}, verbose=False)
    pr = predict(m, ps)
    truth = ps.y_factor.numpy()
    n_f = int(ps.factor_mask.numpy()[0].sum())
    wrong = (truth + 1) % n_f
    good = evaluate_explanations(ps, bs[:3], truth, "oracle", carrier=pr["named_carrier"])[0]
    bad = evaluate_explanations(ps, bs[:3], wrong, "wrong", carrier=pr["named_carrier"])[0]
    assert good["explanation.score@3"] > bad["explanation.score@3"] + 0.05
    assert bad["explanation.score@3"] < 0.15


@pytest.mark.slow
def test_retrieval_baseline_is_a_real_competitor():
    """nearest_reference must be strong, otherwise the comparison is a straw man."""
    bs = _bundles("skirmish-mlp-c*")
    ps = to_problem_set(bs[8:])
    base = naming_baselines(ps, bs[8:])
    y = ps.y_factor.numpy()
    acc = {k: float((v == y).mean()) for k, v in base.items()}
    assert acc["nearest_reference"] > acc["random"] + 0.2
    assert acc["prior"] > acc["random"]


def test_internals_blind_control_must_close_every_channel():
    """The locus profile and the retrieval posterior are computed from B's
    activations; masking the unit descriptors alone leaves the naming head
    reading the model through them.  This is the bug the semantic run exposed."""
    import torch

    from mint.interpreter.model import INTERNAL_FEATURES, Interpreter
    m = Interpreter(d_model=32, n_layers=1, n_heads=2, drop_features=INTERNAL_FEATURES).eval()
    ux, gx = torch.randn(2, 12, 30), torch.randn(2, 8)
    mk = torch.ones(2, 12, dtype=torch.bool)
    fx, fm = torch.randn(2, 3, D_FACTOR), torch.ones(2, 3, dtype=torch.bool)
    lo_a, lo_b = torch.rand(2, 3, 12), torch.rand(2, 3, 12)
    re_a, re_b = torch.rand(2, 3), torch.rand(2, 3)
    with torch.no_grad():
        base = m(ux, gx, mk, factor_x=fx, factor_mask=fm, factor_locus=lo_a, retrieval=re_a)
        via_locus = m(ux, gx, mk, factor_x=fx, factor_mask=fm, factor_locus=lo_b, retrieval=re_a)
        via_retr = m(ux, gx, mk, factor_x=fx, factor_mask=fm, factor_locus=lo_a, retrieval=re_b)
        closed_a = m(ux, gx, mk, factor_x=fx, factor_mask=fm)
        perturbed = ux.clone()
        internal = m.feat_mask == 0  # exactly the columns the control blanks
        perturbed[:, :, internal] += 3.0
        closed_b = m(perturbed, gx, mk, factor_x=fx, factor_mask=fm)
    assert not torch.allclose(base["name"], via_locus["name"], atol=1e-4)
    assert not torch.allclose(base["name"], via_retr["name"], atol=1e-4)
    # with all three channels closed, the internals cannot reach the naming head
    # (architecture shape and B's own output are still visible, by design)
    assert torch.allclose(closed_a["name"], closed_b["name"], atol=1e-6)


def test_paired_test_cannot_be_significant_with_too_few_models():
    """The design constraint that invalidated the earlier runs: at four models
    no paired sign-flip test can reach 0.05, however large the effect."""
    from mint.eval.stats import min_attainable_p, paired_sign_flip

    assert min_attainable_p(4) == pytest.approx(0.125)
    assert min_attainable_p(8) == pytest.approx(0.0078125)
    assert min_attainable_p(24) < 1e-6
    huge = np.arange(4) + 100.0
    _, p = paired_sign_flip(huge, np.zeros(4))
    assert p >= 0.125


def test_holm_bonferroni_is_monotone_and_conservative():
    from mint.eval.stats import holm_bonferroni

    out = holm_bonferroni([0.001, 0.02, 0.4, 0.5], alpha=0.05)
    adj = [o["p_adj"] for o in out]
    assert adj == sorted(adj)
    assert all(a >= p for a, p in zip(adj, [0.001, 0.02, 0.4, 0.5]))
    assert out[0]["reject"] and not out[2]["reject"]

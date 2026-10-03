"""The language target: a real LM, read as a next-token contrast."""
import numpy as np
import torch

from mint.causal import ground_truth
from mint.domains import get_domain
from mint.domains.agreement import ARE, IS, VERB
from mint.models import TargetConfig, build_target, train_target


def _lm(seed=0, epochs=12):
    d = get_domain("agreement")
    tt = train_target(d, TargetConfig("agreement", {"family": "lm", "d_model": 32, "n_layers": 2,
                                                    "n_heads": 4, "d_ff": 64}, seed, 2.0,
                                      n_train=3000, epochs=epochs, lr=3e-3),
                      np.random.default_rng(seed))
    return d, tt


def test_readout_is_the_next_token_contrast():
    d, tt = _lm()
    raw = d.sample(24, np.random.default_rng(1)).raw
    x = torch.as_tensor(d.to_model_input(raw))
    with torch.no_grad():
        got = tt.model(x)[0]
        full = tt.model.token_logits(x)[:, -1]
    assert torch.allclose(got, full[:, ARE] - full[:, IS], atol=1e-5)


def test_the_context_stops_before_the_verb():
    d = get_domain("agreement")
    raw = d.sample(32, np.random.default_rng(2)).raw
    ctx = d.to_model_input(raw)
    assert ctx.shape[1] == VERB == raw.shape[1] - 1
    assert not np.isin(ctx, [IS, ARE]).any(), "the outcome token leaked into the context"


def test_interventions_never_touch_the_outcome_token():
    d = get_domain("agreement")
    raw = d.sample(64, np.random.default_rng(3)).raw
    for f in range(d.spec.n_factors):
        assert np.array_equal(d.do_neutral(raw, f)[:, VERB], raw[:, VERB])


def test_the_lm_learns_the_grammar_not_just_the_attractor():
    """An LM that only read the decoy could not reach the Bayes rate."""
    d, tt = _lm(epochs=25)
    b = d.sample(2000, np.random.default_rng(4))
    from mint.models.train_target import _auc
    assert tt.test_auc > 0.75, tt.test_auc
    assert tt.test_auc <= _auc(b.y, b.logit) + 0.05


def test_ground_truth_on_a_language_model():
    d, tt = _lm()
    gt = ground_truth(tt.model, d, d.sample(16, np.random.default_rng(5)).raw,
                      d.sample(128, np.random.default_rng(6)).raw)
    u, f = tt.model.n_units(), d.spec.n_factors
    assert gt.unit_effect.shape == (16, u) and gt.unit_ie.shape == (16, u, f)
    assert np.isfinite(gt.unit_effect).all() and np.isfinite(gt.unit_ie).all()
    assert {r.kind for r in tt.model.units} == {"attn_head", "mlp_neuron"}


def test_tab_transformer_shares_the_ontology_but_not_the_architecture():
    d = get_domain("skirmish")
    mlp = build_target(d.spec, {"family": "mlp", "widths": (32, 16)}, 0)
    tt = build_target(d.spec, {"family": "tab_transformer", "d_model": 32, "n_layers": 2,
                               "n_heads": 4, "d_ff": 48}, 0)
    x = torch.as_tensor(d.to_model_input(d.sample(8, np.random.default_rng(7)).raw))
    assert mlp(x)[0].shape == tt(x)[0].shape
    assert {r.kind for r in mlp.units} == {"mlp_neuron"}
    assert {r.kind for r in tt.units} == {"attn_head", "mlp_neuron"}


def test_mediation_weight_drops_states_with_nothing_to_explain():
    """A state whose decisive factor barely moves B has a meaningless mediation
    target; the weighted loss must be able to ignore it."""
    import torch
    from mint.interpreter.dataset import make_targets
    d, tt = _lm()
    gt = ground_truth(tt.model, d, d.sample(64, np.random.default_rng(8)).raw,
                      d.sample(128, np.random.default_rng(9)).raw)
    loose = make_targets(gt, mediation_min_effect=0.0)
    tight = make_targets(gt, mediation_min_effect=0.25)
    assert loose.mediation_weight.sum() == 64, "the default must keep every state"
    assert 0 < tight.mediation_weight.sum() < 64
    # the agreement grammar makes some counterfactuals a literal no-op
    n = len(gt.decisive)
    assert (np.abs(gt.factor_total[np.arange(n), gt.decisive]) == 0).any()
    dropped = tight.mediation_weight == 0
    n = len(gt.decisive)
    tot = np.abs(gt.factor_total[np.arange(n), gt.decisive])
    assert tot[dropped].max() <= tot[~dropped].min() + 1e-9

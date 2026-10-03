"""`verify()` decides whether an executed claim is true of the target, so it is
the origin of the executed-validity metric. These pin the contract that an audit
found broken: a claim that proposes no actual intervention was scored as if it
had been verified, which handed `(support=none, NEGATIVE, WEAK)` a perfect
executed record and 0.665 executed validity -- above the confirmatory hybrid.
"""
import numpy as np

from mint.domains import get_domain
from mint.generative.composite import Composite, bucket, support_effect, verify
from mint.models import TargetConfig, train_target

ARCH = {"family": "lm", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64}


def _fixture(n=12):
    d = get_domain("interact")
    tt = train_target(d, TargetConfig("interact", ARCH, 0, 1.0, n_train=1200, epochs=6),
                      np.random.default_rng(0))
    raw = d.sample(n, np.random.default_rng(1)).raw
    ctx = d.to_model_input(raw)
    from mint.generative.task import _contrast
    ref = d.to_model_input(_contrast(d, raw))
    # every position the reference actually changes -- a support that must act
    sup = ctx != ref
    live = sup.any(1)
    eff = support_effect(tt.model, d, raw, sup)
    oracle = Composite(np.zeros(n, int), sup, (eff > 0).astype(int), bucket(eff), live)
    return d, tt.model, raw, oracle


def test_a_claim_that_proposes_nothing_earns_no_executed_credit():
    d, m, raw, oracle = _fixture()
    n, t = oracle.support.shape
    # the degenerate winner: no support, and the sign/strength a no-op forward yields
    empty = Composite(oracle.kind, np.zeros((n, t), bool),
                      np.zeros(n, int), np.zeros(n, int), oracle.live)
    got = verify(m, d, raw, empty, 1.0, oracle)
    assert got["verified_sign"] == 0.0
    assert got["verified_strength"] == 0.0
    assert got["executed_validity"] == 0.0


def test_a_support_that_changes_nothing_earns_no_executed_credit():
    """Naming only positions where the context already equals the reference is an
    empty intervention wearing a non-empty support."""
    d, m, raw, oracle = _fixture()
    inert = ~oracle.support & np.ones_like(oracle.support)
    n = len(oracle.live)
    claim = Composite(oracle.kind, inert, np.zeros(n, int), np.zeros(n, int), oracle.live)
    got = verify(m, d, raw, claim, 1.0, oracle)
    assert got["executed_validity"] == 0.0


def test_the_guard_does_not_penalise_a_true_claim():
    """The oracle proposes a support that really moves the target, so it must keep
    a perfect executed record -- the guard must bite vacuity, not correctness."""
    d, m, raw, oracle = _fixture()
    got = verify(m, d, raw, oracle, 1.0, oracle)
    assert got["verified_sign"] == 1.0
    assert got["verified_strength"] == 1.0
    assert got["executed_validity"] == 1.0

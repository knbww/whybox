"""The encoding must carry model structure and no domain ontology."""
import copy

import numpy as np
import torch

from mint.causal.interventions import _forward_capture, probe_summary
from mint.causal.layout import UnitLayout
from mint.domains import get_domain
from mint.encoding import D_GLOBAL, D_UNIT, UNIT_FEATURES, encode_units
from mint.models import TargetConfig, train_target


def _encode(model, x_probe, x_focal):
    layout = UnitLayout.of(model)
    probe = probe_summary(model, layout, x_probe)
    logit, acts, grads = _forward_capture(model, layout, x_focal)
    return encode_units(probe, acts, grads, logit, model.param_summary()["n_params"])


def test_encoding_is_invariant_to_input_feature_relabelling():
    """Permute B's input ontology (and its first layer to match): same function,
    same internal units, therefore the interpreter's view must be identical."""
    d = get_domain("skirmish")
    tt = train_target(d, TargetConfig("skirmish", {"widths": (24, 12)}, 0, 1.0,
                                      n_train=1500, epochs=8), np.random.default_rng(0))
    m = tt.model
    x_probe = torch.as_tensor(d.to_model_input(d.sample(256, np.random.default_rng(1)).raw))
    x_focal = torch.as_tensor(d.to_model_input(d.sample(32, np.random.default_rng(2)).raw))

    perm = torch.randperm(x_focal.shape[1])
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        m2.hidden[0].weight.copy_(m.hidden[0].weight[:, perm])
    assert torch.allclose(m(x_focal)[0], m2(x_focal[:, perm])[0], atol=1e-5)

    a_u, a_g = _encode(m, x_probe, x_focal)
    b_u, b_g = _encode(m2, x_probe[:, perm], x_focal[:, perm])
    assert np.allclose(a_u, b_u, atol=1e-5), "input ontology leaked into the unit descriptors"
    assert np.allclose(a_g, b_g, atol=1e-5), "input ontology leaked into the global context"


def test_encoding_width_is_the_same_for_every_domain_and_architecture():
    shapes = []
    for name, arch in [("skirmish", {"widths": (24, 12)}), ("clinic", {"widths": (16, 16, 8)}),
                       ("seqworld", {"d_model": 24, "n_layers": 1, "n_heads": 4, "d_ff": 32})]:
        d = get_domain(name)
        tt = train_target(d, TargetConfig(name, arch, 0, 1.0, n_train=1000, epochs=5),
                          np.random.default_rng(0))
        xp = torch.as_tensor(d.to_model_input(d.sample(128, np.random.default_rng(1)).raw))
        xf = torch.as_tensor(d.to_model_input(d.sample(16, np.random.default_rng(2)).raw))
        u, g = _encode(tt.model, xp, xf)
        assert u.shape[2] == D_UNIT and g.shape[1] == D_GLOBAL
        assert np.isfinite(u).all() and np.isfinite(g).all()
        shapes.append((u.shape[0], u.shape[1]))
    assert len({s[1] for s in shapes}) > 1, "test is vacuous unless unit counts differ"


def test_encoder_signature_cannot_see_the_domain():
    import inspect
    sig = inspect.signature(encode_units).parameters
    assert "domain" not in sig and "raw" not in sig and "x" not in sig
    banned = ("feature", "factor", "label", "token", "input_name")
    assert not [f for f in UNIT_FEATURES if any(b in f for b in banned)]


def test_first_order_descriptor_predicts_the_real_ablation_effect():
    """The descriptor is the analytic prediction of a unit's mean-ablation
    effect.  Computed from scalar summaries it is arithmetically wrong for
    multi-component units, which handicapped the strongest non-learned control
    on exactly the transformer cells.  See results/frozen/ERRATA.md item 1."""
    from mint.causal import ground_truth
    from mint.eval.metrics import spearman
    from mint.encoding import UNIT_FEATURES

    col = UNIT_FEATURES.index("first_order_ablation")
    for name, arch in [("skirmish", {"family": "mlp", "widths": (24, 12)}),
                       ("agreement", {"family": "lm", "d_model": 32, "n_layers": 2,
                                      "n_heads": 4, "d_ff": 48})]:
        d = get_domain(name)
        tt = train_target(d, TargetConfig(name, arch, 0, 1.0, n_train=1500, epochs=10),
                          np.random.default_rng(0))
        gt = ground_truth(tt.model, d, d.sample(32, np.random.default_rng(1)).raw,
                          d.sample(256, np.random.default_rng(2)).raw)
        from mint.encoding import encode_problem
        enc = encode_problem(gt, tt.model.param_summary()["n_params"], "t", name)
        rho = float(np.mean([spearman(np.abs(enc.unit_x[i, :, col]),
                                      np.abs(gt.unit_effect[i])) for i in range(32)]))
        assert rho > 0.85, f"{name}: first-order descriptor tracks the effect at only rho={rho:.3f}"

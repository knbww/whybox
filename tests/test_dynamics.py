"""The path signals must satisfy the identities that define them, or nothing read
from them can be trusted."""
import numpy as np
import torch

from mint.causal import UnitLayout
from mint.domains import get_domain
from mint.encoding.dynamics import _contrast, path_signals
from mint.models import TargetConfig, train_target


def _fixture(arch, seed=0):
    d = get_domain("skirmish")
    tt = train_target(d, TargetConfig("skirmish", arch, seed, 1.0, n_train=1500, epochs=8),
                      np.random.default_rng(seed))
    raw = d.sample(24, np.random.default_rng(3)).raw
    with torch.no_grad():
        f = lambda z: tt.model(torch.as_tensor(d.to_model_input(z), dtype=torch.float32))[0].numpy()
        total = f(_contrast(d, raw)) - f(raw)
    return d, tt.model, raw, total


def test_integrated_gradients_are_complete():
    d, m, raw, total = _fixture({"widths": (24, 12)})
    s = path_signals(m, d, raw, np.ones(36), 1.0, steps=256)
    assert np.allclose(s["ig"].sum(1), total, atol=2e-2 * max(1.0, np.abs(total).max()))
    assert np.allclose(s["total"], total, atol=1e-5)


def test_conductance_is_complete_within_every_layer():
    d, m, raw, total = _fixture({"widths": (24, 12)})
    lay = UnitLayout.of(m)
    s = path_signals(m, d, raw, np.ones(lay.n_units), 1.0, steps=256)
    for site, w in zip(lay.sites, lay.widths):
        o = lay.offsets[site]
        assert np.allclose(s["cond"][:, o:o + w].sum(1), total,
                           atol=2e-2 * max(1.0, np.abs(total).max())), site


def test_nothing_moves_where_the_reference_equals_the_state():
    """Positions no factor carries are never displaced, so every path signal there is 0."""
    d, m, raw, _ = _fixture({"widths": (24, 12)})
    s = path_signals(m, d, raw, np.ones(36), 1.0, steps=8)
    carried = {c for f in d.spec.factors for c in f.carriers}
    free = [i for i in range(d.spec.n_inputs) if i not in carried]
    assert free and np.all(s["ig"][:, free] == 0) and np.all(s["gvar"][:, free] == 0)

"""Dynamic signals of a target model along one counterfactual path.

Everything the interpreter reads today is a snapshot at the state itself: a
gradient multiplied by a displacement, taken at `x`. For a ReLU network that is
blind to anything that happens between `x` and the reference, because the
second derivative is zero almost everywhere -- a curvature term at `x` carries
nothing. What a snapshot misses has to be measured *along the path*.

The path is the one the snapshot already uses: from `x` to the reference `r`,
every declared factor set to neutral at once (`generative.task._contrast`). It is
the same path for every state and does not depend on any single factor, so nothing
here is factor-indexed. Measuring it costs `steps + 1` forward passes and `steps`
backward passes of B on interpolated inputs -- B is run, but never with one factor
removed.

Per input position i (tabular targets):
  ig      integrated gradient: (r_i - x_i) * mean over the path of df/dx_i.
          Sums over positions to f(r) - f(x) (Sundararajan et al., 2017).
  gvar    |r_i - x_i| * std over the path of df/dx_i: how much the sensitivity
          of position i changes on the way, i.e. how nonlinear its effect is.

Per internal unit u:
  cond    conductance: sum over path segments of df/da_u at the segment midpoint
          times the change in a_u across the segment (Dhamdhere et al., 2018).
          Within any layer that separates input from output it sums to f(r) - f(x).
  dact    a_u(r) - a_u(x), in units of the unit's own spread on the probe set.

Per state:
  total   f(r) - f(x), in units of B's own logit spread on the probe set.
"""
from __future__ import annotations

import numpy as np
import torch

from ..causal.layout import UnitLayout


def _contrast(domain, raw: np.ndarray) -> np.ndarray:
    out = raw.copy()
    for f in range(domain.spec.n_factors):
        out = domain.do_neutral(out, f)
    return out


def _reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """(N, W, *components) -> (N, W), summing components as a first-order term does."""
    return t if t.dim() == 2 else t.sum(dim=tuple(range(2, t.dim())))


def path_signals(model, domain, raw: np.ndarray, act_std: np.ndarray, logit_std: float,
                 steps: int = 16) -> dict[str, np.ndarray]:
    if domain.spec.kind != "tabular":
        raise NotImplementedError("position signals are defined for tabular targets; "
                                  "a token target needs the gradient through its embedding")
    model.eval()
    layout = UnitLayout.of(model)
    x = torch.as_tensor(domain.to_model_input(raw), dtype=torch.float32)
    r = torch.as_tensor(domain.to_model_input(_contrast(domain, raw)), dtype=torch.float32)
    delta = r - x

    # activations and outputs at the segment endpoints: alpha = 0, 1/steps, ..., 1
    ends_logit, ends_act = [], []
    with torch.no_grad():
        for k in range(steps + 1):
            lg, acts = model(x + (k / steps) * delta, capture=True)
            ends_logit.append(lg)
            ends_act.append({s: acts[s].detach() for s in layout.sites})

    gx = []
    cond = {s: torch.zeros(len(x), w) for s, w in zip(layout.sites, layout.widths)}
    for k in range(steps):
        xm = (x + ((k + 0.5) / steps) * delta).requires_grad_(True)
        lg, acts = model(xm, capture=True)
        grads = torch.autograd.grad(lg.sum(), [xm] + [acts[s] for s in layout.sites],
                                    allow_unused=True)
        gx.append(grads[0].detach())
        for s, g in zip(layout.sites, grads[1:]):
            if g is None:
                continue
            cond[s] += _reduce_sum(g.detach() * (ends_act[k + 1][s] - ends_act[k][s]))
    gx = torch.stack(gx)                                           # (steps, N, T)

    dact = torch.cat([(ends_act[-1][s] - ends_act[0][s]) if ends_act[0][s].dim() == 2 else
                      (ends_act[-1][s] - ends_act[0][s]).mean(dim=tuple(range(2, ends_act[0][s].dim())))
                      for s in layout.sites], dim=1)
    return {
        "ig": (delta * gx.mean(0)).numpy(),
        "gvar": (delta.abs() * gx.std(0)).numpy(),
        "cond": torch.cat([cond[s] for s in layout.sites], dim=1).numpy(),
        "dact": (dact.numpy() / (np.asarray(act_std)[None, :] + 1e-6)),
        "total": ((ends_logit[-1] - ends_logit[0]).numpy() / max(logit_std, 1e-9)),
    }

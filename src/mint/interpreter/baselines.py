"""Scoring rules the interpreter has to beat, all reading the same encoding."""
from __future__ import annotations

import numpy as np
import torch

from ..encoding import UNIT_FEATURES

_F = {name: i for i, name in enumerate(UNIT_FEATURES)}


def _feat(unit_x: torch.Tensor, name: str) -> torch.Tensor:
    return unit_x[..., _F[name]]


def score_random(unit_x, global_x, mask, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(unit_x.shape[:2], generator=g)


def score_correlation(unit_x, global_x, mask) -> torch.Tensor:
    """Pure predictive evidence: |corr(activation, output)| over the probe set."""
    return _feat(unit_x, "abs_corr_out")


def score_activation(unit_x, global_x, mask) -> torch.Tensor:
    return _feat(unit_x, "abs_state_act_z")


def score_gradient(unit_x, global_x, mask) -> torch.Tensor:
    return _feat(unit_x, "state_grad_scaled").abs()


def score_gradxact(unit_x, global_x, mask) -> torch.Tensor:
    return _feat(unit_x, "state_gradact_scaled").abs()


def score_first_order(unit_x, global_x, mask) -> torch.Tensor:
    """(a - a_mean) * dlogit/da: the analytic first-order guess at the ablation
    effect.  The strongest non-learned baseline, and a hard one to beat."""
    return _feat(unit_x, "first_order_ablation").abs()


def score_weight_norm(unit_x, global_x, mask) -> torch.Tensor:
    return _feat(unit_x, "out_norm_rank")


BASELINES = {
    "random": score_random,
    "correlation": score_correlation,
    "activation": score_activation,
    "gradient": score_gradient,
    "grad_x_act": score_gradxact,
    "first_order": score_first_order,
    "weight_norm": score_weight_norm,
}


# --- assisted condition ------------------------------------------------------
# These use measurements the strict condition forbids: the candidate locus
# profile is a mean indirect effect over reference states, i.e. an executed
# patching result.  Under the assisted condition every method may use it, so it
# has to be in the baseline suite -- without it the comparison is not honest.

def score_locus_retrieved(ps) -> np.ndarray:
    """Rank units by the locus profile of the factor retrieval picks.

    No learned parameters.  This is the baseline the audit found missing.
    """
    fl = ps.factor_locus.numpy()
    fr = ps.factor_retrieval.numpy()
    fm = ps.factor_mask.numpy()
    pick = (fr + np.where(fm, 0.0, -np.inf)).argmax(1)
    s = fl[np.arange(len(pick)), pick].astype(np.float64).copy()
    s[~ps.mask.numpy()] = -1e9
    return s


def score_locus_oracle_name(ps) -> np.ndarray:
    """The same lookup handed the correct name: the ceiling of the lookup."""
    fl = ps.factor_locus.numpy()
    s = fl[np.arange(len(ps)), ps.y_factor.numpy()].astype(np.float64).copy()
    s[~ps.mask.numpy()] = -1e9
    return s


ASSISTED_BASELINES = {
    "locus_retrieved": score_locus_retrieved,
    "locus_oracle_name": score_locus_oracle_name,
}

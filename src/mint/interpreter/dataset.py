"""Building interpreter training data: populations of target models + ground truth."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..causal.interventions import CausalGroundTruth, ground_truth
from ..causal.layout import UnitLayout
from ..domains import Domain, get_domain
from ..encoding import D_FACTOR, EncodedProblem, encode_problem, factor_signatures
from ..encoding.model_state import tied_rank
from ..models import TargetConfig, TrainedTarget, population, train_target


@dataclass
class Bundle:
    """One target model B with everything measured about it."""

    trained: TrainedTarget
    gt: CausalGroundTruth
    enc: EncodedProblem
    focal_raw: np.ndarray
    domain: str

    @property
    def n_units(self) -> int:
        return self.enc.n_units

    @property
    def tag(self) -> str:
        return self.trained.cfg.tag()

    @property
    def family(self) -> str:
        return self.trained.cfg.family

    @property
    def cell(self) -> str:
        """Evaluation cell: a domain *and* an architecture family.  Keeping the
        two axes separate is what lets a transfer gap be attributed to one."""
        return f"{self.domain}/{self.family}"


DECOY_MIN_GAP = 0.15  # minimum rank gap for a unit to count as a decoy
RETRIEVAL_NEIGHBOURS = 8  # reference states voting in the retrieval posterior


@dataclass
class Targets:
    """Supervision derived from executed interventions on B."""

    effect: np.ndarray  # (N, U) |ablation effect|, max-normalised per state
    mediation: np.ndarray  # (N, U) |IE via the human-decisive factor| / |factor total|
    decoy: np.ndarray  # (N, U) binary: predictive but not causal
    decoy_score: np.ndarray  # (N, U) continuous rank(|corr|) - rank(|effect|)
    factor: np.ndarray  # (N,) human factor label (domain-specific vocabulary)
    effect_mass: np.ndarray  # (N,) total |effect| relative to probe logit spread
    mediation_weight: np.ndarray  # (N,) 0 where the decisive factor barely moves B


def make_targets(gt: CausalGroundTruth, mediation_min_effect: float = 0.0) -> Targets:
    """`mediation_min_effect` drops states whose decisive factor barely moves B
    from the mediation loss: there is nothing to explain there, and the
    normalised target is pure noise -- and in a language domain the counterfactual
    can be a literal no-op, so B sees the same sentence twice.  It mirrors the
    filter the executed protocol already applies at evaluation time.  Default 0.0
    keeps every state, which is what the first runs used; 0.25 matches the
    evaluation filter and is the better setting for new runs."""
    eff = np.abs(gt.unit_effect)
    eff_n = eff / (eff.max(1, keepdims=True) + 1e-8)

    n = len(gt.decisive)
    ie = np.abs(gt.unit_ie[np.arange(n), :, gt.decisive])
    tot = np.abs(gt.factor_total[np.arange(n), gt.decisive])[:, None]
    med = np.clip(ie / (tot + 1e-6), 0.0, 3.0)
    med = med / (med.max(1, keepdims=True) + 1e-8)

    # A decoy is a unit that *predicts* B's output well but does not *cause* it.
    # Labelled by the gap between the two within-model rankings, with a fixed
    # 10% budget so every model contributes positives (see docs/PROTOCOL.md).
    u = eff.shape[1]
    corr_rank = tied_rank(np.abs(gt.probe.corr_out))
    eff_rank = tied_rank(eff.mean(0))
    dscore = corr_rank - eff_rank
    thresh = max(float(np.quantile(dscore, 0.90)), DECOY_MIN_GAP)
    decoy = (dscore >= thresh).astype(np.float32)
    n_state = len(gt.decisive)
    dec_total = np.abs(gt.factor_total[np.arange(n_state), gt.decisive])
    weight = (np.ones(n_state, dtype=np.float32) if mediation_min_effect <= 0 else
              (dec_total > mediation_min_effect * (gt.probe.logit_std + 1e-6)).astype(np.float32))
    return Targets(
        effect=eff_n.astype(np.float32),
        mediation=med.astype(np.float32),
        decoy=np.repeat(decoy[None], n, axis=0),
        decoy_score=np.repeat(dscore[None].astype(np.float32), n, axis=0),
        factor=gt.decisive.astype(np.int64),
        effect_mass=(eff.sum(1) / (gt.probe.logit_std + 1e-6)).astype(np.float32),
        mediation_weight=weight,
    )


def build_bundles(domain_name: str, cfgs: list[TargetConfig], n_focal: int = 160,
                  n_probe: int = 512, seed: int = 0, verbose: bool = True) -> list[Bundle]:
    domain = get_domain(domain_name)
    rng = np.random.default_rng(seed)
    out: list[Bundle] = []
    for i, cfg in enumerate(cfgs):
        t0 = time.time()
        tt = train_target(domain, cfg, rng)
        # focal / probe states come from the *clean* distribution: the interpreter
        # is asked about a model, not about the model's training leak
        focal = domain.sample(n_focal, rng, spurious_strength=cfg.spurious_strength).raw
        probe = domain.sample(n_probe, rng, spurious_strength=cfg.spurious_strength).raw
        gt = ground_truth(tt.model, domain, focal, probe)
        enc = encode_problem(gt, tt.model.param_summary()["n_params"], cfg.tag(), domain_name)
        out.append(Bundle(tt, gt, enc, focal, domain_name))
        if verbose:
            print(f"  [{i + 1}/{len(cfgs)}] {cfg.tag():34s} auc={tt.test_auc:.3f} "
                  f"units={tt.model.n_units():3d} ({time.time() - t0:.1f}s)", flush=True)
    return out


@dataclass
class ProblemSet:
    """Padded tensors over many (model, state) problems, ready for batching."""

    unit_x: torch.Tensor  # (M, U_max, D)
    global_x: torch.Tensor  # (M, G)
    mask: torch.Tensor  # (M, U_max) bool
    y_effect: torch.Tensor
    y_mediation: torch.Tensor
    y_decoy: torch.Tensor
    y_factor: torch.Tensor
    y_factor_effect: torch.Tensor  # (M, K_max) signed effect of each candidate on B
    factor_x: torch.Tensor  # (M, K_max, D_FACTOR) domain-neutral candidate signatures
    factor_locus: torch.Tensor  # (M, K_max, U_max) where each candidate lives in B
    factor_retrieval: torch.Tensor  # (M, K_max) kernel-retrieval posterior over candidates
    train_factor_mask: torch.Tensor  # (M, K_max) candidates visible during training
    w_name: torch.Tensor  # (M,) 0 for states whose decisive candidate is held out
    factor_mask: torch.Tensor  # (M, K_max)
    w_mediation: torch.Tensor
    domain_id: torch.Tensor
    cell_id: torch.Tensor
    bundle_id: torch.Tensor
    state_id: torch.Tensor
    domains: list[str]
    cells: list[str]

    def __len__(self) -> int:
        return self.unit_x.shape[0]

    def subset(self, idx: torch.Tensor) -> "ProblemSet":
        return ProblemSet(
            self.unit_x[idx], self.global_x[idx], self.mask[idx], self.y_effect[idx],
            self.y_mediation[idx], self.y_decoy[idx], self.y_factor[idx],
            self.y_factor_effect[idx], self.factor_x[idx], self.factor_locus[idx],
            self.factor_retrieval[idx], self.train_factor_mask[idx], self.w_name[idx],
            self.factor_mask[idx],
            self.w_mediation[idx], self.domain_id[idx], self.cell_id[idx], self.bundle_id[idx],
            self.state_id[idx], self.domains, self.cells,
        )


def to_problem_set(bundles: list[Bundle], domains: list[str] | None = None,
                   cells: list[str] | None = None, bundle_offset: int = 0,
                   mediation_min_effect: float = 0.0, ref_frac: float = 0.375,
                   holdout_factors: dict[str, tuple[str, ...]] | None = None,
                   retrieval_tau: float = 0.5) -> ProblemSet:
    """`ref_frac` of each model's states build the candidate signatures; the rest
    become problems.  The two sets are disjoint, so the answer to "which factor
    is in play here" is never sitting inside the candidate description."""
    domains = domains or sorted({b.domain for b in bundles})
    cells = cells or sorted({b.cell for b in bundles})
    u_max = max(b.n_units for b in bundles)
    k_max = max(len(b.gt.factor_total[0]) for b in bundles)
    holdout_factors = holdout_factors or {}
    ux, gx, mk, ye, ym, yd, yf, fe, fx, fl, fr, ftm, wn, fm, wm, di, ci, bi, si = (
        [] for _ in range(19))
    for k, b in enumerate(bundles):
        t = make_targets(b.gt, mediation_min_effect)
        n_all, u = b.enc.unit_x.shape[:2]
        n_ref = max(2, int(round(ref_frac * n_all)))
        ref_idx = np.arange(n_ref)
        keep = np.arange(n_ref, n_all)
        n = len(keep)
        pad, kpad = u_max - u, k_max - b.gt.factor_total.shape[1]
        p2 = lambda a, v=0.0: np.pad(a[keep], ((0, 0), (0, pad)), constant_values=v)
        ux.append(np.pad(b.enc.unit_x[keep], ((0, 0), (0, pad), (0, 0))))
        gx.append(b.enc.global_x[keep])
        mk.append(p2(np.ones((n_all, u), dtype=bool)).astype(bool))
        ye.append(p2(t.effect)); ym.append(p2(t.mediation))
        yd.append(p2(t.decoy)); yf.append(t.factor[keep]); wm.append(t.mediation_weight[keep])

        sig = factor_signatures(b.gt, ref_idx)  # (F, D_FACTOR)
        fx.append(np.repeat(np.pad(sig, ((0, kpad), (0, 0)))[None], n, axis=0))
        # where the candidate lives: the mean per-unit indirect effect over the
        # reference states, as a distribution over this model's own units.  Unit
        # identity is meaningful inside one model and meaningless across models,
        # which is exactly the scope this is used at.
        locus = np.abs(b.gt.unit_ie[ref_idx]).mean(0).T  # (F, U)
        locus = locus / (locus.sum(1, keepdims=True) + 1e-9)
        fl.append(np.repeat(np.pad(locus, ((0, kpad), (0, pad)))[None], n, axis=0))
        fmask = np.zeros((n, k_max), dtype=bool)
        fmask[:, :sig.shape[0]] = True
        fm.append(fmask)
        scale = b.gt.probe.logit_std + 1e-6
        fe.append(np.pad(b.gt.factor_total[keep] / scale, ((0, 0), (0, kpad))))

        # kernel retrieval over the reference states, in B's own activation space:
        # the non-parametric baseline the learned head has to improve on, handed
        # to it as a feature so it can rerank rather than start from nothing
        az = (b.gt.acts - b.gt.probe.act_mean) / (b.gt.probe.act_std + 1e-6)
        az = az / (np.linalg.norm(az, axis=1, keepdims=True) + 1e-9)
        sim = az[keep] @ az[ref_idx].T
        top = np.argsort(-sim, axis=1)[:, :RETRIEVAL_NEIGHBOURS]
        s_top = np.take_along_axis(sim, top, axis=1)
        # temperature adapted to the local spread, so the vote is a neighbourhood
        # vote rather than a restatement of the reference prior
        temp = retrieval_tau * (s_top.std(1, keepdims=True) + 1e-6)
        w_ret = np.exp((s_top - s_top.max(1, keepdims=True)) / temp)
        w_ret /= w_ret.sum(1, keepdims=True) + 1e-9
        votes = b.gt.decisive[ref_idx][top]  # (n, m)
        post = np.zeros((n, k_max))
        for g in range(b.gt.factor_total.shape[1]):
            post[:, g] = (w_ret * (votes == g)).sum(1)
        fr.append(post)

        # unseen-factor holdout: these candidates are invisible while training
        hold = [get_domain(b.domain).spec.factor_index(x)
                for x in holdout_factors.get(b.domain, ())]
        tmask = fmask.copy()
        if hold:
            tmask[:, hold] = False
        ftm.append(tmask)
        wn.append((~np.isin(t.factor[keep], hold)).astype(np.float32))

        di.append(np.full(n, domains.index(b.domain)))
        ci.append(np.full(n, cells.index(b.cell)))
        bi.append(np.full(n, k + bundle_offset))
        si.append(keep)
    cat = lambda xs, dt: torch.as_tensor(np.concatenate(xs, 0), dtype=dt)
    return ProblemSet(
        unit_x=cat(ux, torch.float32), global_x=cat(gx, torch.float32),
        mask=cat(mk, torch.bool), y_effect=cat(ye, torch.float32),
        y_mediation=cat(ym, torch.float32), y_decoy=cat(yd, torch.float32),
        y_factor=cat(yf, torch.long), y_factor_effect=cat(fe, torch.float32),
        factor_x=cat(fx, torch.float32), factor_locus=cat(fl, torch.float32),
        factor_retrieval=cat(fr, torch.float32), train_factor_mask=cat(ftm, torch.bool),
        w_name=cat(wn, torch.float32), factor_mask=cat(fm, torch.bool),
        w_mediation=cat(wm, torch.float32), domain_id=cat(di, torch.long),
        cell_id=cat(ci, torch.long), bundle_id=cat(bi, torch.long),
        state_id=cat(si, torch.long), domains=domains, cells=cells,
    )

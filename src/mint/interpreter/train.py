"""Training interpreter A on human-grounded, intervention-validated supervision."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import ProblemSet
from .model import Interpreter, listwise_loss, masked_mse


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    tau: float = 0.15
    lambda_causal: float = 1.0
    lambda_mediation: float = 1.0
    lambda_decoy: float = 0.3
    lambda_vocab: float = 0.3  # legacy closed-vocabulary head
    lambda_name: float = 1.0  # open-vocabulary naming from candidate signatures
    lambda_effect: float = 0.5  # signed magnitude claim for each candidate
    lambda_carrier: float = 0.5  # per-candidate localisation
    # Factor frequencies are very skewed (man advantage decides 58% of CS2-like
    # rounds).  Plain cross-entropy collapses onto that prior, which is exactly
    # what the naming task must beat, so weight classes inversely by frequency.
    # 0 = plain cross-entropy, 1 = fully inverse-frequency, 0.5 = square root.
    # Full balancing buys minority-class coverage but costs end-to-end
    # explanation score; 0.5 is within noise of plain CE on the score and much
    # better on coverage.  See docs/RESULTS.md.
    name_balance_power: float = 0.5
    seed: int = 0
    drop_features: tuple[str, ...] = ()
    # Blank the candidate locus profile: does naming survive on the aggregate
    # signature alone, without being told where each candidate lives in B?
    drop_locus: bool = False
    # Hand the retrieval posterior to the naming head so it reranks instead of
    # competing with it.  False makes the head a purely learned mapping.
    use_retrieval: bool = True
    name: str = "interpreter"


def _batches(n: int, domain_id: torch.Tensor, bs: int, g: torch.Generator):
    """Batches are domain-homogeneous so the per-domain vocabulary head applies."""
    for d in domain_id.unique().tolist():
        idx = torch.nonzero(domain_id == d, as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        for i in range(0, len(idx), bs):
            yield d, idx[i:i + bs]


def train_interpreter(train_set: ProblemSet, cfg: TrainConfig,
                      factor_heads: dict[str, int], val_set: ProblemSet | None = None,
                      verbose: bool = True) -> Interpreter:
    torch.manual_seed(cfg.seed)
    model = Interpreter(cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dropout,
                        factor_heads=factor_heads, drop_features=cfg.drop_features)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(cfg.epochs * max(1, len(train_set) // cfg.batch_size), 1),
        pct_start=0.2,
    )
    g = torch.Generator().manual_seed(cfg.seed)
    domains = train_set.domains
    for ep in range(cfg.epochs):
        model.train()
        stats = np.zeros(7)
        nb = 0
        for d, idx in _batches(len(train_set), train_set.domain_id, cfg.batch_size, g):
            b = train_set.subset(idx)
            dom = domains[d]
            out = model(b.unit_x, b.global_x, b.mask, domain=dom,
                        factor_x=b.factor_x, factor_mask=b.train_factor_mask,
                        factor_locus=None if cfg.drop_locus else b.factor_locus,
                        retrieval=b.factor_retrieval if cfg.use_retrieval else None)
            l_c = listwise_loss(out["causal"], b.y_effect, b.mask, cfg.tau) \
                + masked_mse(out["causal"], b.y_effect, b.mask)
            w = b.w_mediation
            l_m = listwise_loss(out["mediation"], b.y_mediation, b.mask, cfg.tau, w) \
                + masked_mse(out["mediation"], b.y_mediation, b.mask, w)
            l_d = (F.binary_cross_entropy_with_logits(out["decoy"], b.y_decoy, reduction="none")
                   * b.mask).sum() / b.mask.sum().clamp(min=1)
            l_v = F.cross_entropy(out["factor"], b.y_factor) if "factor" in out \
                else torch.zeros((), dtype=torch.float32)

            # the explanation itself: name the cause, say how big it is, and say
            # which internal variables carry it
            cnt = torch.bincount(b.y_factor, minlength=out["name"].shape[1]).float()
            wts = cnt.clamp(min=1).pow(-cfg.name_balance_power)[b.y_factor] * b.w_name
            per = F.cross_entropy(out["name"], b.y_factor, reduction="none")
            l_n = (per * wts).sum() / wts.sum().clamp(min=1e-6)
            fm = b.factor_mask.float()
            l_e = ((torch.tanh(out["factor_effect"]) - torch.tanh(b.y_factor_effect)) ** 2
                   * fm).sum() / fm.sum().clamp(min=1)
            carrier = out["carrier"][torch.arange(len(b.y_factor)), b.y_factor]
            l_k = listwise_loss(carrier, b.y_mediation, b.mask, cfg.tau, b.w_mediation)

            loss = (cfg.lambda_causal * l_c + cfg.lambda_mediation * l_m
                    + cfg.lambda_decoy * l_d + cfg.lambda_vocab * l_v
                    + cfg.lambda_name * l_n + cfg.lambda_effect * l_e
                    + cfg.lambda_carrier * l_k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            stats += [float(v.detach()) for v in (l_c, l_m, l_d, l_v, l_n, l_e, l_k)]
            nb += 1
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            s = stats / max(nb, 1)
            msg = (f"  ep {ep:3d}  causal={s[0]:.3f} mediation={s[1]:.3f} decoy={s[2]:.3f} "
                   f"name={s[4]:.3f} effect={s[5]:.3f} carrier={s[6]:.3f}")
            if val_set is not None and len(val_set):
                msg += f"  val_causal_rho={quick_rho(model, val_set):.3f}"
            print(msg, flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model: Interpreter, ps: ProblemSet, bs: int = 128,
            drop_locus: bool = False, use_retrieval: bool = True) -> dict[str, np.ndarray]:
    model.eval()
    keys = ("causal", "mediation", "decoy", "name", "factor_effect", "named_carrier")
    outs: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for i in range(0, len(ps), bs):
        b = ps.subset(torch.arange(i, min(i + bs, len(ps))))
        o = model(b.unit_x, b.global_x, b.mask, factor_x=b.factor_x,
                  factor_mask=b.factor_mask,
                  factor_locus=None if drop_locus else b.factor_locus,
                  retrieval=b.factor_retrieval if use_retrieval else None)
        named = o["name"].argmax(1)
        o["named_carrier"] = o["carrier"][torch.arange(len(named)), named]
        for k in keys:
            v = o[k].clone().float()
            if k in ("causal", "mediation", "named_carrier"):
                v[~b.mask] = -1e9
            elif k == "name":
                v[~b.factor_mask] = -1e9
            outs[k].append(v.numpy())
    return {k: np.concatenate(v, 0) for k, v in outs.items()}


@torch.no_grad()
def quick_rho(model: Interpreter, ps: ProblemSet, n: int = 256) -> float:
    from ..eval.metrics import spearman
    idx = torch.arange(min(n, len(ps)))
    b = ps.subset(idx)
    was_training = model.training
    model.eval()
    o = model(b.unit_x, b.global_x, b.mask)["causal"].numpy()
    if was_training:
        model.train()
    y, m = b.y_effect.numpy(), b.mask.numpy()
    return float(np.mean([spearman(o[i][m[i]], y[i][m[i]]) for i in range(len(idx))]))

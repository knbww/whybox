"""Fit a population of target models B on a domain.

The population is deliberately heterogeneous: different widths/depths, seeds,
training budgets, and -- most importantly -- different `spurious_strength` in
the training distribution.  Models trained on a strongly leaking distribution
learn to read the spurious input, which manufactures internal units that
*correlate* with the output without *causing* it.  Those are the units the
interpreter has to learn to discount.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from ..domains import Domain
from .target import TargetModel, build_target


# Architecture families available per domain kind.  Having two families for the
# same tabular ontology is what lets us separate "different domain" from
# "different architecture" in the transfer analysis.
ARCH_PRESETS: dict[str, list[dict]] = {
    "mlp": [{"family": "mlp", "widths": (32, 16)},
            {"family": "mlp", "widths": (24, 24, 12)},
            {"family": "mlp", "widths": (48,)}],
    "tab_transformer": [
        {"family": "tab_transformer", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 48},
        {"family": "tab_transformer", "d_model": 24, "n_layers": 2, "n_heads": 4, "d_ff": 40},
        {"family": "tab_transformer", "d_model": 32, "n_layers": 3, "n_heads": 2, "d_ff": 48}],
    "transformer": [
        {"family": "transformer", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 48},
        {"family": "transformer", "d_model": 24, "n_layers": 1, "n_heads": 4, "d_ff": 40},
        {"family": "transformer", "d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 32}],
    "lm": [{"family": "lm", "d_model": 48, "n_layers": 2, "n_heads": 4, "d_ff": 96},
           {"family": "lm", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64},
           {"family": "lm", "d_model": 48, "n_layers": 3, "n_heads": 4, "d_ff": 96}],
}
DEFAULT_FAMILY = {"tabular": "mlp", "sequence": "transformer", "language": "lm"}


@dataclass(frozen=True)
class TargetConfig:
    domain: str
    arch: dict
    seed: int
    spurious_strength: float = 1.0
    n_train: int = 6000
    epochs: int = 60
    lr: float = 3e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    @property
    def family(self) -> str:
        return self.arch.get("family", "mlp")

    def tag(self) -> str:
        a = self.arch
        shape = "x".join(str(v) for v in a.get("widths", (a.get("d_model"), a.get("n_layers"))))
        return f"{self.domain}/{self.family}-{shape}-s{self.seed}-sp{self.spurious_strength:.1f}"


@dataclass
class TrainedTarget:
    model: TargetModel
    cfg: TargetConfig
    train_auc: float
    test_auc: float
    calib_err: float

    def meta(self) -> dict:
        return {**asdict(self.cfg), "tag": self.cfg.tag(), "test_auc": self.test_auc,
                "calib_err": self.calib_err, "n_units": self.model.n_units()}


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.5
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def train_target(domain: Domain, cfg: TargetConfig, rng: np.random.Generator) -> TrainedTarget:
    tr = domain.sample(cfg.n_train, rng, spurious_strength=cfg.spurious_strength)
    te = domain.sample(2000, rng, spurious_strength=cfg.spurious_strength)
    xtr = torch.as_tensor(domain.to_model_input(tr.raw))
    xte = torch.as_tensor(domain.to_model_input(te.raw))
    ytr = torch.as_tensor(tr.y, dtype=torch.float32)
    yte_np = te.y

    model = build_target(domain.spec, cfg.arch, cfg.seed)
    # A language target is trained as a language model -- next-token prediction
    # over the whole sentence -- not as a classifier of the outcome.  The binary
    # readout used for interpretation is a *contrast between two next tokens*,
    # which the LM objective never sees as a label.
    is_lm = hasattr(model, "token_logits")
    full = torch.as_tensor(tr.raw, dtype=torch.long) if is_lm else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n, bs = len(ytr), 256
    g = torch.Generator().manual_seed(cfg.seed)
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if is_lm:
                seq = full[b]
                tok = model.token_logits(seq[:, :-1])
                loss = F.cross_entropy(tok.reshape(-1, tok.shape[-1]), seq[:, 1:].reshape(-1),
                                       label_smoothing=cfg.label_smoothing)
            else:
                logit, _ = model(xtr[b])
                target = ytr[b] * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
                loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        str_ = model(xtr)[0].numpy()
        ste = model(xte)[0].numpy()
    p = 1 / (1 + np.exp(-ste))
    bins = np.clip((p * 10).astype(int), 0, 9)
    calib = float(np.mean([abs(p[bins == b].mean() - yte_np[bins == b].mean())
                           for b in range(10) if (bins == b).sum() > 20] or [0.0]))
    return TrainedTarget(model, cfg, _auc(tr.y, str_), _auc(yte_np, ste), calib)


def population(domain: Domain, seeds=(0, 1, 2, 3), spurious=(0.0, 1.0, 2.0),
               archs=None, epochs: int = 60, n_train: int = 6000,
               weight_decay: float = 1e-4, label_smoothing: float = 0.0,
               family: str | None = None) -> list[TargetConfig]:
    """A grid of target-model configurations for one domain and architecture family."""
    if archs is None:
        archs = ARCH_PRESETS[family or DEFAULT_FAMILY[domain.spec.kind]]
    cfgs = []
    for s in seeds:
        for sp in spurious:
            arch = archs[s % len(archs)]
            cfgs.append(TargetConfig(domain.spec.name, arch, s, sp, n_train=n_train, epochs=epochs,
                                 weight_decay=weight_decay, label_smoothing=label_smoothing))
    return cfgs

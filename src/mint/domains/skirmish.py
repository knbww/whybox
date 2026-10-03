"""SKIRMISH -- a CS2-style round win-probability domain (source domain).

Ontology: tactical state of a 5v5 round.  The generative process is a
structured log-odds model with genuine factor interactions (bomb control only
matters late; time pressure only bites when you are down a man and the bomb is
not planted).  Two inputs are *spurious*: `star_rating` is generated to
correlate with the outcome but carries zero causal weight, and `server_tick`
is pure nuisance.  Target models trained here will happily latch onto
`star_rating`, which is exactly what gives us correlational-but-not-causal
internal units to test the interpreter against.
"""
from __future__ import annotations

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

A_ALIVE, E_ALIVE, BOMB, SITE, UTIL, EQUIP, TIME, INFO, STAR, TICK = range(10)

SPEC = DomainSpec(
    name="skirmish",
    kind="tabular",
    n_inputs=10,
    input_names=(
        "allies_alive", "enemies_alive", "bomb_planted", "site_control",
        "utility_diff", "equipment_diff", "time_left", "info_advantage",
        "star_rating", "server_tick",
    ),
    factors=(
        FactorSpec("MAN_ADVANTAGE", (A_ALIVE, E_ALIVE), (3.0, 3.0), "player-count delta (the 1v5)"),
        FactorSpec("BOMB_CONTROL", (BOMB,), (0.0,), "bomb planted / defused state"),
        FactorSpec("MAP_CONTROL", (SITE,), (0.5,), "site and map control"),
        FactorSpec("UTILITY", (UTIL,), (0.0,), "grenade/utility differential"),
        FactorSpec("EQUIPMENT", (EQUIP,), (0.0,), "buy / economy differential"),
        FactorSpec("TIME_PRESSURE", (TIME,), (0.6,), "clock running out"),
        FactorSpec("INFO", (INFO,), (0.5,), "information advantage"),
    ),
    spurious_inputs=(STAR, TICK),
)


class Skirmish:
    spec = SPEC

    def __init__(self) -> None:
        self._ref_mu = np.zeros(self.spec.n_inputs)
        self._ref_sd = np.ones(self.spec.n_inputs)
        self._ref_mu, self._ref_sd = self._reference_stats()

    # ---- generative process -------------------------------------------------
    def logit(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        man = x[:, A_ALIVE] - x[:, E_ALIVE]
        late = 1.0 - x[:, TIME]
        out = (
            0.85 * man
            + (2.6 * x[:, BOMB] - 1.1) * late
            + 2.20 * (x[:, SITE] - 0.5)
            + 0.50 * x[:, UTIL]
            + 2.40 * x[:, EQUIP]
            + 1.40 * (x[:, INFO] - 0.5)
        )
        # interaction: the clock only kills you when you are behind and unplanted
        squeeze = np.clip(0.35 - x[:, TIME], 0.0, None) / 0.35
        out -= 1.60 * squeeze * (1.0 - x[:, BOMB]) * (man <= 0)
        return out

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, self.spec.n_inputs), dtype=np.float64)
        x[:, A_ALIVE] = rng.integers(0, 6, n)
        x[:, E_ALIVE] = rng.integers(0, 6, n)
        x[:, BOMB] = (rng.random(n) < 0.35).astype(float)
        x[:, SITE] = np.clip(rng.beta(2.0, 2.0, n), 0.02, 0.98)
        x[:, UTIL] = np.clip(rng.normal(0.0, 1.8, n), -5, 5)
        x[:, EQUIP] = np.clip(rng.normal(0.0, 0.45, n), -1, 1)
        x[:, TIME] = np.clip(rng.beta(2.2, 1.6, n), 0.0, 1.0)
        x[:, INFO] = np.clip(rng.beta(2.0, 2.0, n), 0.02, 0.98)
        x[:, TICK] = rng.normal(0.0, 1.0, n)

        p = sigmoid(self.logit(x))
        # spurious: reads like a skill stat, is really a leak of the outcome
        x[:, STAR] = 1.0 + spurious_strength * 0.8 * (p - 0.5) + rng.normal(0, 0.12, n)
        y = (rng.random(n) < p).astype(np.int64)
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain=self.spec.name)

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        f = self.spec.factors[factor]
        out = np.array(raw, dtype=np.float64, copy=True)
        for idx, val in zip(f.carriers, f.neutral):
            out[:, idx] = val
        return out

    # ---- model interface ----------------------------------------------------
    def _reference_stats(self) -> tuple[np.ndarray, np.ndarray]:
        ref = self.sample(8192, np.random.default_rng(0)).raw
        return ref.mean(0), ref.std(0) + 1e-6

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return ((np.atleast_2d(raw) - self._ref_mu) / self._ref_sd).astype(np.float32)

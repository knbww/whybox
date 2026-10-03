"""CLINIC -- inpatient deterioration risk (transfer domain, tabular).

Different ontology, different factor vocabulary, different feature scales and
distributions from SKIRMISH; same *shape* of problem.  `admission_hour` is the
spurious input (it correlates with risk through triage practice, not biology).
"""
from __future__ import annotations

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

AGE, SPO2, RESP, SBP, HR, LACT, WBC, COMORB, HOUR, BED = range(10)

SPEC = DomainSpec(
    name="clinic",
    kind="tabular",
    n_inputs=10,
    input_names=(
        "age", "spo2", "resp_rate", "sbp", "heart_rate", "lactate",
        "wbc", "comorbidity_count", "admission_hour", "bed_index",
    ),
    factors=(
        FactorSpec("HYPOXIA", (SPO2, RESP), (97.0, 16.0), "oxygenation / respiratory failure"),
        FactorSpec("HEMODYNAMIC", (SBP, HR), (120.0, 78.0), "blood pressure and rate"),
        FactorSpec("PERFUSION", (LACT,), (1.0,), "tissue perfusion (lactate)"),
        FactorSpec("INFLAMMATION", (WBC,), (8.0,), "white-cell response"),
        FactorSpec("FRAILTY", (AGE, COMORB), (55.0, 0.0), "age and comorbidity burden"),
    ),
    spurious_inputs=(HOUR, BED),
)


class Clinic:
    spec = SPEC

    def __init__(self) -> None:
        self._ref_mu = np.zeros(self.spec.n_inputs)
        self._ref_sd = np.ones(self.spec.n_inputs)
        ref = self.sample(8192, np.random.default_rng(0)).raw
        self._ref_mu, self._ref_sd = ref.mean(0), ref.std(0) + 1e-6

    def logit(self, raw: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        hypox = 1.10 * (97.0 - x[:, SPO2]) / 6.0 + 0.60 * (x[:, RESP] - 16.0) / 7.0
        hemo = 0.95 * (120.0 - x[:, SBP]) / 22.0 + 0.45 * (x[:, HR] - 78.0) / 22.0
        perf = 1.05 * (x[:, LACT] - 1.0) / 1.8
        infl = 0.80 * (np.abs(x[:, WBC] - 8.0) - 2.5) / 5.0
        frail = 0.70 * (x[:, AGE] - 55.0) / 18.0 + 0.55 * x[:, COMORB] / 1.8
        shock = 1.30 * ((x[:, LACT] > 3.0) & (x[:, SBP] < 95.0))  # interaction
        return hypox + hemo + perf + infl + frail + shock - 1.5

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, self.spec.n_inputs), dtype=np.float64)
        x[:, AGE] = np.clip(rng.normal(62, 17, n), 18, 96)
        x[:, SPO2] = np.clip(rng.normal(95.5, 3.2, n), 74, 100)
        x[:, RESP] = np.clip(rng.normal(18, 5.0, n), 8, 44)
        x[:, SBP] = np.clip(rng.normal(122, 22, n), 62, 200)
        x[:, HR] = np.clip(rng.normal(84, 19, n), 38, 170)
        x[:, LACT] = np.clip(rng.gamma(2.0, 0.9, n), 0.3, 12)
        x[:, WBC] = np.clip(rng.gamma(4.0, 2.3, n), 1.0, 32)
        x[:, COMORB] = rng.integers(0, 6, n)
        x[:, BED] = rng.integers(1, 60, n)

        p = sigmoid(self.logit(x))
        x[:, HOUR] = np.clip(12.0 + spurious_strength * 9.0 * (p - 0.5) + rng.normal(0, 2.2, n), 0, 23)
        y = (rng.random(n) < p).astype(np.int64)
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain=self.spec.name)

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        f = self.spec.factors[factor]
        out = np.array(raw, dtype=np.float64, copy=True)
        for idx, val in zip(f.carriers, f.neutral):
            out[:, idx] = val
        return out

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return ((np.atleast_2d(raw) - self._ref_mu) / self._ref_sd).astype(np.float32)

"""CAR -- UCI Car Evaluation (Bohanec & Rajkovic, 1988): a real dataset with a known rule.

The 1728 rows are every combination of six attributes, each labelled by the published
hierarchical decision model (DEX). The world's answer to any intervention is therefore a
row of the table itself: nothing is simulated and nothing is extrapolated.

The causes are the concepts of that published hierarchy, not single inputs:

    PRICE    buying price + maintenance cost
    COMFORT  doors + seats + luggage boot
    SAFETY   estimated safety

Neutral values are the middle values (medium price, medium maintenance, 4 doors, 4 seats,
medium boot, medium safety); do(cause := neutral) writes them into the cause's inputs.

The data are used as published: no column is added and no label is changed. The outcome
is the dataset's own four-level acceptability read as a number (unacc 0, acc 1, good 2,
vgood 3). `logit` returns that level for any combination, so the world's response to a
suppression is measured on the same scale as a target model's answer.

A state is held as the integer codes of its six values (stored as floats, as every
tabular domain's raw state is); a target model reads them through `to_model_input`.
"""
from __future__ import annotations

import numpy as np

from ..realdata.tasks import CAR as TABLE
from ..realdata.tasks import all_codes
from .base import DomainSpec, FactorSpec, StateBatch

BUYING, MAINT, DOORS, PERSONS, LUG_BOOT, SAFETY = range(6)

SPEC = DomainSpec(
    name="car",
    kind="tabular",
    n_inputs=6,
    input_names=TABLE.inputs,
    factors=(
        FactorSpec("PRICE", (BUYING, MAINT), (1.0, 1.0), "buying price and maintenance cost"),
        FactorSpec("COMFORT", (DOORS, PERSONS, LUG_BOOT), (2.0, 1.0, 1.0),
                   "doors, seats and luggage boot"),
        FactorSpec("SAFETY", (SAFETY,), (1.0,), "estimated safety"),
    ),
)


class Car:
    spec = SPEC

    def __init__(self) -> None:
        for f in SPEC.factors:
            for c, v in zip(f.carriers, f.neutral):
                if TABLE.neutral[c] != int(v):
                    raise ValueError(f"{f.name}: neutral value of {TABLE.inputs[c]} differs "
                                     f"from the dictionary of features")
        codes = all_codes(TABLE)
        self.states = codes.astype(np.float64)          # all 1728 combinations
        self._level = np.full([len(v) for v in TABLE.values], -1, dtype=np.int64)
        self._level[tuple(codes.T)] = TABLE.world(codes).astype(np.int64)
        if (self._level < 0).any():
            raise ValueError("the table does not cover every combination")

    @staticmethod
    def codes(raw: np.ndarray) -> np.ndarray:
        c = np.rint(np.atleast_2d(np.asarray(raw, dtype=np.float64))).astype(np.int64)
        hi = np.array([len(v) for v in TABLE.values])
        if (c < 0).any() or (c >= hi).any():
            raise ValueError("a code lies outside its attribute's values")
        return c

    # ---- the world -------------------------------------------------------------
    def logit(self, raw: np.ndarray) -> np.ndarray:
        """The published acceptability level 0..3 of each state."""
        return self._level[tuple(self.codes(raw).T)].astype(np.float64)

    def sample(self, n: int, rng: np.random.Generator,
               spurious_strength: float = 0.0) -> StateBatch:
        """Rows of the table, uniformly. There is no decoy: the data are used as published."""
        if spurious_strength != 0.0:
            raise ValueError("car has no decoy input; spurious_strength must be 0")
        x = self.states[rng.integers(0, len(self.states), n)]
        level = self.logit(x)
        return StateBatch(raw=x, logit=level, y=level.astype(np.int64), domain=SPEC.name)

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        f = SPEC.factors[factor]
        out = np.array(raw, dtype=np.float64, copy=True)
        for idx, val in zip(f.carriers, f.neutral):
            out[:, idx] = val
        return out

    # ---- model interface ----------------------------------------------------
    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return TABLE.encode(self.codes(raw))

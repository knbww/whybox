"""Real datasets whose generating rule is known, read as causal worlds.

Each task is a published dataset in which the label is a known function of the
inputs, so the world's answer to any intervention can be computed exactly:

    balance    UCI Balance Scale (Siegler, 1976): the scale tips towards the side
               with the larger weight * distance; 625 rows, all 5^4 combinations.
    car        UCI Car Evaluation (Bohanec & Rajkovic, 1988): a hierarchical
               decision model; the 1728 rows cover the whole attribute space, so
               the table itself is the rule.
    tictactoe  UCI Tic-Tac-Toe Endgame (Aha, 1991): x wins if it holds one of the
               eight lines; 958 legal endgame boards.

An input is held as an integer code into its ordered list of values. A model B
reads the codes standardised (`encode`); the world reads them through `world`.

Neutral values say what "this input is not pushing the decision" means -- a
middle weight, a medium price, an empty square. They define the suppression of an
input on B and on the world, and, in the paper's method (mint.domains.car), the
reference point of the first-order signals.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data" / "uci"


@dataclass(frozen=True)
class Task:
    name: str
    file: str
    inputs: tuple[str, ...]                 # names of the inputs, in file order
    display: tuple[str, ...]                # the same, as a person would say them
    values: tuple[tuple[str, ...], ...]     # per input: its values, in order
    neutral: tuple[int, ...]                # per input: code of the neutral value (check only)
    classes: tuple[str, ...]                # labels as written in the file
    label_first: bool = False               # the class is the first column, not the last
    score_meaning: str = ""
    shown: tuple[tuple[str, ...], ...] = ()  # per input: its values as a person would say them

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    # ---- the data ------------------------------------------------------------
    @cached_property
    def table(self) -> tuple[np.ndarray, np.ndarray]:
        """(codes (N, n_inputs) int64, label (N,) int64 index into `classes`)."""
        rows = [r.strip().split(",") for r in (DATA / self.file).read_text().splitlines()
                if r.strip()]
        if self.label_first:
            rows = [r[1:] + r[:1] for r in rows]
        codes = np.array([[self.values[i].index(v) for i, v in enumerate(r[:-1])]
                          for r in rows], dtype=np.int64)
        label = np.array([self.classes.index(r[-1]) for r in rows], dtype=np.int64)
        return codes, label

    def encode(self, codes: np.ndarray) -> np.ndarray:
        """What B reads: each code standardised over its own value range."""
        codes = np.atleast_2d(np.asarray(codes, dtype=np.float64))
        n = np.array([len(v) for v in self.values], dtype=np.float64)
        mid = (n - 1) / 2.0
        sd = np.sqrt((n ** 2 - 1) / 12.0)       # sd of a uniform code 0..n-1
        return ((codes - mid) / sd).astype(np.float32)

    # ---- the world -------------------------------------------------------------
    def world(self, codes: np.ndarray) -> np.ndarray:
        """The world's score for any codes, including combinations made by an
        intervention. Larger means more of what B is asked to predict."""
        raise NotImplementedError

    def label_of_score(self, score: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def value(self, i: int, code: int) -> str:
        vals = self.shown[i] if self.shown else self.values[i]
        return vals[int(code)]

    def show(self, i: int, code: int) -> str:
        return f"«{self.display[i]}» = {self.value(i, code)}"


class Balance(Task):
    def world(self, codes):
        c = np.atleast_2d(codes) + 1                 # weights and distances 1..5
        return (c[:, 0] * c[:, 1] - c[:, 2] * c[:, 3]).astype(np.float64)

    def label_of_score(self, score):
        return np.where(score > 0, self.classes.index("L"),
                        np.where(score < 0, self.classes.index("R"), self.classes.index("B")))


class Car(Task):
    @cached_property
    def _lookup(self) -> dict[tuple[int, ...], int]:
        codes, label = self.table
        return {tuple(map(int, c)): int(l) for c, l in zip(codes, label)}

    def world(self, codes):
        # ordinal acceptability: unacc 0, acc 1, good 2, vgood 3
        order = {self.classes.index(c): k for k, c in enumerate(("unacc", "acc", "good", "vgood"))}
        return np.array([order[self._lookup[tuple(map(int, c))]]
                         for c in np.atleast_2d(codes)], dtype=np.float64)

    def label_of_score(self, score):
        inv = {k: self.classes.index(c) for k, c in enumerate(("unacc", "acc", "good", "vgood"))}
        return np.array([inv[int(s)] for s in score])


LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


class TicTacToe(Task):
    def world(self, codes):
        x = np.atleast_2d(codes) == self.values[0].index("x")
        return np.array([float(any(r[list(L)].all() for L in LINES)) for r in x])

    def label_of_score(self, score):
        return np.where(score > 0, self.classes.index("positive"), self.classes.index("negative"))


BALANCE = Balance(
    name="balance", file="balance-scale.data", label_first=True,
    inputs=("left_weight", "left_distance", "right_weight", "right_distance"),
    display=("левый груз", "левое плечо", "правый груз", "правое плечо"),
    values=(("1", "2", "3", "4", "5"),) * 4,
    neutral=(2, 2, 2, 2),                       # the middle value, 3
    classes=("L", "B", "R"),
    score_meaning="на сколько весы склоняются влево, а не вправо")

CAR = Car(
    name="car", file="car.data",
    inputs=("buying", "maint", "doors", "persons", "lug_boot", "safety"),
    display=("цена покупки", "стоимость обслуживания", "число дверей", "число мест",
             "багажник", "безопасность"),
    values=(("low", "med", "high", "vhigh"), ("low", "med", "high", "vhigh"),
            ("2", "3", "4", "5more"), ("2", "4", "more"), ("small", "med", "big"),
            ("low", "med", "high")),
    neutral=(1, 1, 2, 1, 1, 1),                 # med, med, 4 doors, 4 seats, med, med
    classes=("unacc", "acc", "good", "vgood"),
    score_meaning="насколько машина приемлема",
    shown=(("низкая", "средняя", "высокая", "очень высокая"),
           ("низкая", "средняя", "высокая", "очень высокая"),
           ("2", "3", "4", "5 и больше"), ("2", "4", "больше 4"),
           ("маленький", "средний", "большой"), ("низкая", "средняя", "высокая")))

TICTACTOE = TicTacToe(
    name="tictactoe", file="tic-tac-toe.data",
    inputs=("top_left", "top_middle", "top_right", "middle_left", "center", "middle_right",
            "bottom_left", "bottom_middle", "bottom_right"),
    display=("левая верхняя клетка", "средняя верхняя клетка", "правая верхняя клетка",
             "левая средняя клетка", "центр", "правая средняя клетка",
             "левая нижняя клетка", "средняя нижняя клетка", "правая нижняя клетка"),
    values=(("o", "b", "x"),) * 9,              # o < empty < x, read by B as -1, 0, +1
    neutral=(1,) * 9,                           # an empty square
    classes=("positive", "negative"),
    score_meaning="насколько крестики выиграли",
    shown=(("нолик", "пусто", "крестик"),) * 9)

TASKS = {t.name: t for t in (BALANCE, CAR, TICTACTOE)}


def get_task(name: str) -> Task:
    return TASKS[name]


def all_codes(task: Task) -> np.ndarray:
    """Every combination of input values: the space any intervention lands in."""
    return np.array(list(itertools.product(*[range(len(v)) for v in task.values])),
                    dtype=np.int64)

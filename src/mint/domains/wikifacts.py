"""WIKIFACTS -- claim verification over real Wikidata attributes.

The first domain whose semantic labels are natural and concrete rather than
invented. The entities and their attribute values come from a live Wikidata
query cached under `data/wikidata/` together with the SPARQL text and the fetch
date, so every fact in the task can be re-checked against the source.

A claim names an entity and lists all eight of its attributes at fixed
positions; at most two values have been swapped for another entity's. The model
answers whether the claim holds.

Two formats were measured before this one was kept. Naming the relation
explicitly and asserting only three of them per claim -- which reads more like
natural language -- turned out to be much *harder* for a small model
(AUC 0.529 against a Bayes rate of 0.711, essentially chance), because it adds a
third composition step: read the relation token, retrieve the entity's value for
that relation, then compare. At fixed positions the model can learn eight
independent "is slot k right for this entity" detectors and combine them.
The **cause** of a `false` answer is the attribute that was swapped, and its
human name is the Wikidata property itself -- capital, official language,
currency, continent, ISO code, calling code, driving side, top-level domain.
Ten narrow labels including the two structural ones, all of them things a person
can check.

The counterfactual is exact and per-factor: restore that one attribute to the
value in the snapshot. Nothing else moves.

One honest limitation. Several of these properties are multi-valued in Wikidata
(a country may have more than one official language); the query keeps one value
per entity deterministically. The "correct" value is therefore the value **in
this snapshot**, not a claim about the world. Reproducibility is provided by the
stored query and date, not by the snapshot being exhaustive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import DomainSpec, FactorSpec, StateBatch, sigmoid

DATA = Path(__file__).resolve().parents[3] / "data" / "wikidata"
ATTRS = ("capital", "language", "currency", "continent",
         "iso", "calling_code", "driving_side", "tld")
# how much each wrong attribute pushes the verdict toward false
WEIGHT = dict(capital=2.2, currency=1.9, language=1.7, continent=1.5,
              iso=1.2, tld=1.0, calling_code=0.85, driving_side=0.7)
LABELS = dict(capital="CAPITAL", language="OFFICIAL_LANGUAGE", currency="CURRENCY",
              continent="CONTINENT", iso="ISO_CODE", calling_code="CALLING_CODE",
              driving_side="DRIVING_SIDE", tld="TOP_LEVEL_DOMAIN")

ENTITY = 0
VAL_SLOTS = tuple(range(1, 9))  # one fixed slot per attribute, in ATTRS order
NOTE, VERDICT = 9, 10


def _load() -> list[dict]:
    base = {r["entity"]: r for r in json.loads((DATA / "countries.json").read_text())["rows"]}
    extra = {r["entity"]: r for r in json.loads((DATA / "country_extras.json").read_text())["rows"]}
    rows = [{**base[e], **extra[e]} for e in sorted(base) if e in extra]
    if not rows:
        raise RuntimeError("no entity has all attributes; re-run scripts/fetch_wikidata.py")
    return rows


class _Vocab:
    def __init__(self, rows: list[dict]):
        self.tok: dict[str, int] = {"<pad>": 0, "<note_a>": 1, "<note_b>": 2,
                                    "false": 3, "true": 4}
        self.by_attr: dict[str, list[int]] = {}
        for name, key in [("entity", "entity")] + [(a, a) for a in ATTRS]:
            ids = []
            for v in sorted({r[key] for r in rows}):
                t = f"{name}:{v}"
                self.tok.setdefault(t, len(self.tok))
                ids.append(self.tok[t])
            self.by_attr[name] = ids
        self.inv = {v: k for k, v in self.tok.items()}

    def id(self, attr: str, value: str) -> int:
        return self.tok[f"{attr}:{value}"]

    def __len__(self) -> int:
        return len(self.tok)


class WikiFacts:
    def __init__(self) -> None:
        self.rows = _load()
        self.v = _Vocab(self.rows)
        self.truth = np.array([[self.v.id(a, r[a]) for a in ATTRS] for r in self.rows])
        self.ent_ids = np.array([self.v.id("entity", r["entity"]) for r in self.rows])
        self.spec = DomainSpec(
            name="wikifacts", kind="language", n_inputs=11,
            input_names=("entity",) + ATTRS + ("note", "verdict"),
            factors=tuple(FactorSpec(LABELS[a], (VAL_SLOTS[i],), (0.0,),
                                     f"Wikidata property: {a}") for i, a in enumerate(ATTRS)),
            spurious_inputs=(self.v.tok["<note_b>"],),
            vocab_size=len(self.v), contrast=(self.v.tok["false"], self.v.tok["true"]),
        )

    # ---- generative process -------------------------------------------------
    def _entity_index(self, x: np.ndarray) -> np.ndarray:
        order = np.argsort(self.ent_ids)
        return order[np.clip(np.searchsorted(self.ent_ids[order], x[:, ENTITY]),
                             0, len(order) - 1)]

    def _wrong(self, raw: np.ndarray) -> np.ndarray:
        """(n, 8) which attributes disagree with the snapshot."""
        x = np.atleast_2d(np.asarray(raw, dtype=np.int64))
        return x[:, list(VAL_SLOTS)] != self.truth[self._entity_index(x)]

    def logit(self, raw: np.ndarray) -> np.ndarray:
        w = np.array([WEIGHT[a] for a in ATTRS])
        return 2.2 - (self._wrong(raw) * w).sum(1)

    def sample(self, n: int, rng: np.random.Generator, spurious_strength: float = 1.0) -> StateBatch:
        x = np.zeros((n, 11), dtype=np.int64)
        ei = rng.integers(0, len(self.rows), n)
        x[:, ENTITY] = self.ent_ids[ei]
        x[:, 1:9] = self.truth[ei]
        n_wrong = rng.choice([0, 1, 2], n, p=[0.08, 0.74, 0.18])
        for i in range(n):
            for a in rng.choice(len(ATTRS), int(n_wrong[i]), replace=False):
                other = rng.integers(0, len(self.rows))
                while self.truth[other, a] == self.truth[ei[i], a]:
                    other = rng.integers(0, len(self.rows))
                x[i, VAL_SLOTS[a]] = self.truth[other, a]
        p = sigmoid(self.logit(x))
        lean = rng.random(n) < np.clip(0.5 + spurious_strength * 0.45 * (p - 0.5) * 2, 0.02, 0.98)
        x[:, NOTE] = np.where(lean, self.v.tok["<note_b>"], self.v.tok["<note_a>"])
        y = (rng.random(n) < p).astype(np.int64)
        x[:, VERDICT] = np.where(y == 1, self.v.tok["true"], self.v.tok["false"])
        return StateBatch(raw=x, logit=self.logit(x), y=y, domain="wikifacts")

    def do_neutral(self, raw: np.ndarray, factor: int) -> np.ndarray:
        """Restore one attribute to the snapshot value. Nothing else moves."""
        x = np.array(raw, dtype=np.int64, copy=True)
        x[:, VAL_SLOTS[factor]] = self.truth[self._entity_index(x), factor]
        return x

    def to_model_input(self, raw: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(raw, dtype=np.int64))[:, :VERDICT]

    def to_sentence(self, row: np.ndarray) -> str:
        return " ".join(self.v.inv[int(t)].split(":", 1)[-1] for t in row)

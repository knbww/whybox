#!/usr/bin/env python
"""Naming a cause from its description in words, with no table of carriers.

Diagnostic. No protocol was frozen before this ran.

Every result in this project so far answers by scoring B's input positions; the
name is attached afterwards from a table that says which positions belong to
which cause. Three measurements today say that this is the binding constraint
rather than the signals:

  * zeroing B's per-unit and global streams changes nothing in any of the three
    worlds, so the whole reading already comes from the per-position vector;
  * on an external model the interpreter's answers coincided with the analytic
    pointer on 100% of states, because with one carrier per cause and one number
    per position, naming *is* the argmax of that number and there is nothing left
    to learn;
  * widening the cause vocabulary from four to seven was the only change that
    moved the disagreement cell.

So the readout moves up a level. A candidate arrives as a sentence -- "the
premise carries not", "number of the head noun" -- embedded by a small sentence
encoder. Each of B's input positions arrives as its influence on B's answer
*together with the embedding of the word sitting there*. The interpreter has to
find, by itself, which words a description is about, and score the candidate by
what B's answer does at those positions. No carrier table enters at any point,
at training or at evaluation.

The target's seven causes are NOT all unseen. An audit found that GPT-2's
SUBJECT_NUMBER description is word for word `agreement`'s and its COORDINATION
description is close to it (cosine 0.88), and that the head nouns are the same
words `agreement` uses. Those two causes are most of GPT-2's labels. The run
therefore reports, per target cause, the closest source description, and splits
accuracy into causes with a near-duplicate in the sources and genuinely new ones.

Three arms, and the second of them is the control that matters:

  full            influence per position + the word at that position
  text_only       the influence features zeroed. If this scores above the
                  constant, the descriptions and the words alone give the answer
                  and nothing about B is being read. Amortised explainers are
                  known to hide the answer inside the explanation
                  (Jethani et al., AISTATS 2021), and this project has already
                  found three leaks of that family.
  influence_only  the word embeddings zeroed, so a description cannot be
                  grounded in anything. This is the old readout without its
                  carrier table, and it should be near chance.

Writes `results/described_causes.json`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mint.domains import annotate, get_domain
from mint.domains.agreement import WORDS as AGREEMENT_WORDS
from mint.domains.langtasks import ENTAIL_WORDS, POLARITY_WORDS

import external_lm as E
from external_lm import Target, all_neutral, ground_truth, sample, sim_lm_path_signals
from follows_b import MARGIN, load_population, pos_features, prepare

# Three source ontologies rather than one. A description channel trained against
# a single vocabulary of four causes has no reason to become a matching function:
# it can be a four-way classifier and still fit. Variety across ontologies is what
# forces it to read the words. `agreement` also carries two causes on one token,
# which a readout over input positions cannot separate at all and a readout over
# described candidates can.
SOURCES = {
    "entail": ("results/cache/entail-lm-b75d8fee9e.pt", ENTAIL_WORDS),
    "agreement": ("results/cache/agreement-lm-bb9b473848.pt", AGREEMENT_WORDS),
    "polarity": ("results/cache/polarity-lm-f25776064c.pt", POLARITY_WORDS),
}
N_MODELS, N_FIT = 24, 12
SEEDS, STEPS, EPOCHS = 3, 16, 40
N_FOCAL, N_PROBE, DATA_SEED = 160, 256, 0
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ARMS = ("full", "text_only", "influence_only")
# Candidate sets seen during training. With a fixed set of four the description
# channel is constant across every example, so nothing forces the model to read
# it: a four-way classifier over positions fits the data just as well and then
# means nothing when seven unseen descriptions arrive. `random` varies the set --
# the true cause plus a sample of other causes and of descriptions borrowed from
# unrelated domains, in a shuffled order -- so the only way to be right is to
# match the description against the words.
N_CAND = 8
SEEN_COSINE = 0.85   # a target description this close to a source one is not new


class Text:
    """Sentence embeddings, cached by string. Read-only, never fine-tuned."""

    def __init__(self, name: str = TEXT_MODEL):
        from transformers import AutoModel, AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            b = self.tk(chunk, return_tensors="pt", padding=True, truncation=True)
            h = self.m(**b).last_hidden_state
            mask = b["attention_mask"][..., None].float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            v = F.normalize(v, dim=-1).numpy()
            for t, row in zip(chunk, v):
                self.cache[t] = row
        return np.stack([self.cache[t] for t in texts])


class Namer(nn.Module):
    """Positions in, one score per described candidate out.

    A candidate's score is a soft maximum of its affinity with the positions, so
    the model says "this description matches somewhere, and here is what B's
    answer does there" rather than scoring positions and being told afterwards
    whose they are.
    """

    def __init__(self, d_inf: int, d_text: int, d: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.pos_in = nn.Sequential(nn.Linear(d_inf + d, d), nn.GELU(), nn.Linear(d, d))
        self.word_in = nn.Linear(d_text, d)
        self.cand_in = nn.Sequential(nn.Linear(d_text, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.scale = d ** -0.5

    def forward(self, inf, word, cand):
        # inf (N,T,d_inf)  word (N,T,d_text)  cand (N,K,d_text)
        h = self.enc(self.pos_in(torch.cat([inf, self.word_in(word)], -1)))
        q = self.cand_in(cand)
        return torch.logsumexp(torch.einsum("nkd,ntd->nkt", q, h) * self.scale, dim=-1)


def sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(spec) -> list[str]:
    return [f"{f.name.lower().replace('_', ' ')}: {f.description}" for f in spec.factors]


def source_packs(name: str, text: Text) -> tuple[list[dict], np.ndarray]:
    path, table = SOURCES[name]
    d = get_domain(name)
    desc = text(describe(d.spec)).astype(np.float32)
    bundles = torch.load(path, weights_only=False)[:N_MODELS]
    out = []
    for b in bundles:
        p = prepare(b, d)
        ig = sim_lm_path_signals(b.trained.model, d, b.focal_raw, STEPS)["ig"]
        ids = np.asarray(d.to_model_input(b.focal_raw))
        words = np.stack([text([table[int(t)] for t in row]) for row in ids])
        # where every world effect is zero the "world's cause" is index 0 by a stable
        # argsort, not a cause; those states are split off from genuine disagreement
        no_world = np.abs(annotate(d, b.focal_raw).factor_effect).max(1) == 0
        out.append({"domain": name, "inf": pos_features(ig), "word": words.astype(np.float32),
                    "y": p["y"], "world": p["world"], "margin": p["margin"],
                    "no_world": no_world})
    return out, desc


def gpt2_pack(text: Text, target: Target, template: str) -> tuple[dict, np.ndarray]:
    E.TEMPLATE = template
    rng = np.random.default_rng(DATA_SEED)
    focal, probe = sample(N_FOCAL, rng), sample(N_PROBE, rng)
    gt = ground_truth(target, focal, probe)
    ids = target.ids(focal)
    ref = target.ids([all_neutral(s) for s in focal])
    ig = target.position_path_signals(ids, ref, STEPS)["ig"]
    words = np.stack([text(E.words(s)) for s in focal])
    desc = text([f"{nm.lower().replace('_', ' ')}: {d}" for nm, _, d in E.factors()])
    pack = {"inf": pos_features(ig), "word": words.astype(np.float32),
            "y": gt["y"], "world": gt["world"], "margin": gt["margin"],
            "world_group": gt["world_group"], "world_single": gt["world_single"]}
    return pack, desc.astype(np.float32)


POOL_DOMAINS = ("skirmish", "clinic", "agreement", "polarity", "interact")


def distractor_pool(text: Text, exclude: str) -> np.ndarray:
    """Descriptions of causes from other domains -- never from `exclude`, the domain
    the example comes from. An earlier version drew from all five, so for an
    `agreement` or `polarity` example the true description came back as a
    "negative" in about a fifth of examples and the target was ambiguous."""
    out = []
    for name in POOL_DOMAINS:
        if name == exclude:
            continue
        try:
            spec = get_domain(name).spec
        except KeyError:
            continue
        out += [f"{f.name.lower().replace('_', ' ')}: {f.description}" for f in spec.factors]
    return text(out).astype(np.float32)


def draw_candidates(y: np.ndarray, desc: np.ndarray, pool: np.ndarray,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(N, N_CAND, d) candidate descriptions and (N,) the index of the true one."""
    n, k_src = len(y), len(desc)
    cand = np.zeros((n, N_CAND, desc.shape[1]), np.float32)
    lab = rng.integers(0, N_CAND, n)
    for i in range(n):
        others = [desc[j] for j in range(k_src) if j != y[i]]
        others += [pool[j] for j in rng.choice(len(pool), N_CAND, replace=False)]
        pick = rng.permutation(len(others))[:N_CAND - 1]
        rest = [others[j] for j in pick]
        cand[i, lab[i]] = desc[y[i]]
        cand[i, [j for j in range(N_CAND) if j != lab[i]]] = np.stack(rest)
    return cand, lab


GPT2_CELLS = ("ALL", "SEEN", "UNSEEN", "SINGLE_AGREE", "SINGLE_DISAGREE", "NONE", "OVER")


def gpt2_masks(pack: dict, keep: np.ndarray, seen: np.ndarray) -> dict[str, np.ndarray]:
    """SEEN / UNSEEN by whether the true cause's description has a near-duplicate in
    the sources; the grammar groups as in `external_lm.grammar_group`."""
    y = pack["y"][keep]
    g, w = pack["world_group"][keep], pack["world_single"][keep]
    single = g == "single"
    return {"ALL": np.ones(len(y), bool), "SEEN": seen[y], "UNSEEN": ~seen[y],
            "SINGLE_AGREE": single & (w == y), "SINGLE_DISAGREE": single & (w != y),
            "NONE": g == "none", "OVER": g == "over"}


def views(pack: dict, arm: str, keep: np.ndarray):
    inf = pack["inf"][keep]
    word = pack["word"][keep]
    if arm == "text_only":
        inf = np.zeros_like(inf)
    elif arm == "influence_only":
        word = np.zeros_like(word)
    return torch.as_tensor(inf), torch.as_tensor(word)


def _domain_tensors(packs, desc, arm, pool, rng):
    infs, words, ys = [], [], []
    for p in packs:
        k = p["margin"] > MARGIN
        a, b = views(p, arm, k)
        infs.append(a); words.append(b); ys.append(p["y"][k])
    inf = torch.cat(infs); word = torch.cat(words)
    y_src = np.concatenate(ys)
    if pool is None:
        y = torch.as_tensor(y_src, dtype=torch.long)
        cand = torch.as_tensor(desc)[None].expand(len(y), -1, -1)
    else:
        c, lab = draw_candidates(y_src, desc, pool, rng)
        y = torch.as_tensor(lab, dtype=torch.long)
        cand = torch.as_tensor(c)
    return inf, word, cand, y


def fit(by_domain: dict, arm: str, seed: int, pool: np.ndarray | None = None):
    """`by_domain` maps a source name to (fit packs, descriptions). Batches are
    drawn within one domain at a time, so sentences of different length never
    have to be padded together."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = {nm: _domain_tensors(pk, ds, arm, None if pool is None else pool[nm], rng)
            for nm, (pk, ds) in by_domain.items()}
    any_inf, _, any_cand, _ = next(iter(data.values()))
    m = Namer(any_inf.shape[-1], any_cand.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        m.train()
        batches = []
        for nm, (inf, word, cand, y) in data.items():
            perm = torch.randperm(len(y), generator=g)
            batches += [(nm, perm[i:i + 64]) for i in range(0, len(perm), 64)]
        for j in torch.randperm(len(batches), generator=g).tolist():
            nm, b_ = batches[j]
            inf, word, cand, y = data[nm]
            loss = F.cross_entropy(m(inf[b_], word[b_], cand[b_]), y[b_])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def predict(m, pack: dict, desc: np.ndarray, arm: str, keep: np.ndarray) -> np.ndarray:
    inf, word = views(pack, arm, keep)
    cand = torch.as_tensor(desc)[None].expand(len(inf), -1, -1)
    return m(inf, word, cand).argmax(-1).numpy()


def cells(pred, y, agree) -> dict[str, float]:
    ok = (pred == y).astype(float)
    return {"ALL": float(ok.mean()),
            "AGREE": float(ok[agree].mean()) if agree.any() else float("nan"),
            "DISAGREE": float(ok[~agree].mean()) if (~agree).any() else float("nan")}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="random", choices=("fixed", "random"))
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--out", default="results/described_causes_multi.json")
    a_ = ap.parse_args(argv)
    torch.set_num_threads(4)
    t0 = time.time()
    names = [n.strip() for n in a_.sources.split(",") if n.strip()]

    text = Text()
    src = {}
    for nm in names:
        packs, desc = source_packs(nm, text)
        src[nm] = (packs, desc)
        print(f"источник {nm}: {len(packs)} моделей, {len(desc)} описанных причин, "
              f"{packs[0]['inf'].shape[1]} позиций ({time.time() - t0:.0f}s)", flush=True)

    target = Target("gpt2")
    tgt, tgt_desc = gpt2_pack(text, target, "wide")
    src_all = np.concatenate([d for _, d in src.values()])
    nearest = (tgt_desc @ src_all.T).max(1)            # unit-norm embeddings: cosine
    seen = nearest >= SEEN_COSINE
    names_t = [nm for nm, _, _ in E.factors()]
    print(f"цель gpt2/wide: {len(tgt_desc)} описанных причин; ближайшее описание в "
          f"источниках: " + ", ".join(f"{n} {c:.2f}" for n, c in zip(names_t, nearest)) +
          f"; считаются виденными (cos >= {SEEN_COSINE}): {int(seen.sum())} "
          f"({time.time() - t0:.0f}s)", flush=True)

    fit_by_domain = {nm: (packs[:N_FIT], desc) for nm, (packs, desc) in src.items()}
    k_t = tgt["margin"] > MARGIN
    y_t, agree_t = tgt["y"][k_t], tgt["world"][k_t] == tgt["y"][k_t]

    pool = ({nm: distractor_pool(text, exclude=nm) for nm in names}
            if a_.candidates == "random" else None)
    if pool is not None:
        print("пул отвлекающих описаний без собственной области: " +
              ", ".join(f"{nm} {len(v)}" for nm, v in pool.items()), flush=True)

    rows = []
    for arm in ARMS:
        models = [fit(fit_by_domain, arm, s, pool) for s in range(SEEDS)]
        print(f"  [{arm}] обучен ({time.time() - t0:.0f}s)", flush=True)
        line = []
        for nm, (packs, desc) in src.items():
            per = {}
            for p in packs[N_FIT:]:
                k = p["margin"] > MARGIN
                y, agree = p["y"][k], p["world"][k] == p["y"][k]
                nw = p["no_world"][k]
                acc = np.mean([(predict(m, p, desc, arm, k) == y) for m in models], 0)
                for sub, msk in (("ALL", np.ones_like(agree)), ("AGREE", agree),
                                 ("DISAGREE", ~agree),
                                 ("DISAGREE_GENUINE", ~agree & ~nw),
                                 ("DISAGREE_NO_WORLD_CAUSE", ~agree & nw)):
                    per.setdefault(sub, []).append(
                        float(acc[msk].mean()) if msk.sum() else np.nan)
            for sub, v in per.items():
                vv = [x for x in v if np.isfinite(x)]
                rows.append({"arm": arm, "target": f"{nm} (12 отложенных)", "cell": sub,
                             "n_models": len(vv), "accuracy": float(np.mean(vv)),
                             "chance": 1.0 / len(desc),
                             "per_model": [{"accuracy": x} for x in vv]})
            line.append(f"{nm} {np.nanmean(per['ALL']):.4f}/"
                        f"{np.nanmean(per['DISAGREE_GENUINE']):.4f}")
        gm = gpt2_masks(tgt, k_t, seen)
        gseeds = []
        for m in models:
            ok = (predict(m, tgt, tgt_desc, arm, k_t) == y_t).astype(float)
            gseeds.append({c: float(ok[msk].mean()) if msk.any() else float("nan")
                           for c, msk in gm.items()})
        for c, msk in gm.items():
            rows.append({"arm": arm, "target": "gpt2/wide", "cell": c, "n_models": 1,
                         "n_states": int(msk.sum()), "chance": 1.0 / len(tgt_desc),
                         "accuracy": float(np.mean([a[c] for a in gseeds])),
                         "per_seed": [a[c] for a in gseeds]})
        print(f"  [{arm}] " + "  ".join(line) + "   |   gpt2/wide " + "  ".join(
            f"{c} {np.mean([a[c] for a in gseeds]):.4f}" for c in GPT2_CELLS), flush=True)

    gm = gpt2_masks(tgt, k_t, seen)
    K = len(tgt_desc)
    floors = {
        "random": {c: 1.0 / K for c in gm},
        "grammar_oracle": {c: (float((tgt["world_single"][k_t][msk] == y_t[msk]).mean())
                               if msk.any() and c.startswith("SINGLE") else float("nan"))
                           for c, msk in gm.items()},
        # chosen with the answers in hand: an upper envelope, not a baseline
        "best_constant_envelope": {c: (float(max(np.mean(y_t[msk] == f) for f in range(K)))
                                       if msk.any() else float("nan"))
                                   for c, msk in gm.items()},
    }
    for nm, d_ in floors.items():
        for c, v in d_.items():
            rows.append({"arm": nm, "target": "gpt2/wide", "cell": c, "n_models": 0,
                         "n_states": int(gm[c].sum()), "accuracy": v, "chance": 1.0 / K})

    src_cells = ("ALL", "DISAGREE_GENUINE", "DISAGREE_NO_WORLD_CAUSE")
    print("\nисточники (отложенные модели): " + "   ".join(f"{c}" for c in src_cells))
    for arm in ARMS:
        parts = []
        for nm in names:
            a = {r["cell"]: r["accuracy"] for r in rows
                 if r["arm"] == arm and r["target"].startswith(nm)}
            parts.append(f"{nm} " + "/".join(f"{a[c]:.4f}" for c in src_cells))
        print(f"  {arm:16s}" + "   ".join(parts))
    print(f"\ngpt2/wide, состояний: " + ", ".join(f"{c} {int(m.sum())}" for c, m in gm.items()))
    print(f"{'арм':24s}" + "".join(f"{c:>17s}" for c in GPT2_CELLS))
    for arm in ARMS:
        g = {r["cell"]: r["accuracy"] for r in rows
             if r["arm"] == arm and r["target"] == "gpt2/wide"}
        print(f"{arm:24s}" + "".join(f"{g[c]:>17.4f}" for c in GPT2_CELLS))
    for nm, d_ in floors.items():
        print(f"{nm:24s}" + "".join(f"{d_[c]:>17.4f}" for c in GPT2_CELLS))

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"commit": git("rev-parse", "HEAD"),
                          "dirty": bool(git("status", "--porcelain")),
                          "role": "diagnostic, no protocol frozen before the run",
                          "sources": {nm: {"path": SOURCES[nm][0],
                                           "sha256": sha256(SOURCES[nm][0])} for nm in names},
                          "target": "gpt2", "template": "wide",
                          "text_encoder": TEXT_MODEL, "arms": list(ARMS),
                          "readout": "candidate-level, no carrier table at train or eval",
                          "candidates": a_.candidates, "n_candidates_train": N_CAND,
                          "pool": "other domains only, per source",
                          "seen_cosine": SEEN_COSINE,
                          "target_nearest_source_cosine": [float(x) for x in nearest],
                          "margin": MARGIN, "interpreter_seeds": list(range(SEEDS)),
                          "steps": STEPS, "torch": torch.__version__},
           "rows": rows, "wall_seconds": time.time() - t0}
    with open(a_.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a_.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

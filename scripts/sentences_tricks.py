#!/usr/bin/env python
"""Pretrained language models on full sentences with a trick: relative clauses.

Section 3.7 of the paper read GPT-2 and Qwen on a short template ("The nurse near the
new window ..."). This run follows the advice to read them on full sentences with a
trick, by section 3.7's own method, on sentences whose relative clause holds a noun
that can pull the verb's number the wrong way:

    rc_obj   "The quiet nurse that the critics like ..."   the attractor is the clause's
                                                           subject, with its own verb
    rc_subj  "The quiet nurse that likes the critics ..."  the attractor is the clause's
                                                           object
    wide     section 3.7's long template, two prepositional attractors, for comparison

PROTOCOL. Committed to the repository before the run.
---------------------------------------------------------------------------
Question.   Does the interpreter, trained only on this project's toy logic language
            models, name the factor a pretrained model relies on when it chooses the
            number of the verb in such sentences, without any further training?
Method.     Section 3.7's, unchanged except for the sentences. Named factors at fixed
            positions; suppression by the declared neutral word (a noun's singular form,
            together with the verb that agrees with it; the adjective "new"); B's label is
            the factor whose suppression moves B's logit(" are") - logit(" is") most, in
            units of its spread on 256 probe sentences, scored only where the leader beats
            the next factor by 0.10 (the margin rule of every earlier run).
Sentences.  rc_obj, rc_subj and wide, 160 per template and 256 probe sentences, random
            stream 0, the number of every noun drawn independently (attract = 0), so about
            half the attractors differ in number from the subject.
Interpreter. external_gpt2's recipe: the 12 fitted entail targets, the integrated
            gradient of each position along the straight path in embedding space from the
            sentence to its all-neutral reference (16 segments), unit and global streams
            zeroed, Pointer, 40 epochs, seeds 0, 1, 2. Trained once and saved to
            results/sentences_tricks_interpreter.pt, so that a run on other hardware reads
            with exactly these weights.
Targets.    gpt2 and Qwen/Qwen2.5-0.5B-Instruct here. Larger models (for example on
            Colab) run through --models under this same protocol, each run in its own
            result file.
Cells.      ALL; SINGLE_AGREE (plural subject, B rests on SUBJECT_NUMBER); SINGLE_DISAGREE
            (plural subject, B rests on another factor: B departs from the grammar); NONE
            (singular subject: the grammar names no cause); OVER (wide only: a plural and
            coordinated subject).
Floors.     random 1/K; live (uniform over the factors whose suppression changes B at
            all at that sentence); grammar oracle in the SINGLE cells.
Test.       Per (model, template): paired sign flip over sentences of the interpreter's
            correctness (mean over seeds) against the live floor's expected correctness,
            20000 random flips; Holm over all (model, template) pairs of one run.
Read-out.   Per (model, template): "above the live floor" if that comparison rejects at
            0.05 after Holm with a positive difference; otherwise "not above the live floor".
Reported, no test. The completeness residual of the path, |sum over positions of ig -
            (f(ref) - f(x))| in B's spread, mean and 95th percentile; state counts and every
            arm in every cell.
Run.        PYTHONPATH=.:src:scripts .venv/bin/python scripts/sentences_tricks.py
            With a git checkout it refuses to run unless this file equals the committed
            blob and no tracked file is modified; without one (e.g. Colab) it records the
            file's sha256 for comparison with the committed protocol. It never overwrites a
            result. `--smoke`: one template, GPT-2, 24 sentences, 2 epochs, nothing saved.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

import external_gpt2 as G
import external_lm as E
from follows_b import MARGIN, Pointer
from mint.eval.stats import holm_bonferroni, paired_sign_flip

WEIGHTS = "results/sentences_tricks_interpreter.pt"
OUT = "results/sentences_tricks.json"
TEMPLATES = ("rc_obj", "rc_subj", "wide")
MODELS = ("gpt2", "Qwen/Qwen2.5-0.5B-Instruct")
N_FOCAL, N_PROBE, SEED, SEEDS = 160, 256, 0, (0, 1, 2)
CELLS = ("ALL", "SINGLE_AGREE", "SINGLE_DISAGREE", "NONE", "OVER")


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def interpreters(weights: str, smoke: bool) -> tuple[list[Pointer], int, dict]:
    """The interpreter of section 3.7, trained on the toy logic models or loaded."""
    if os.path.exists(weights) and not smoke:
        w = torch.load(weights, weights_only=False)
        models = []
        for sd in w["state_dicts"]:
            m = Pointer(*w["dims"])
            m.load_state_dict(sd)
            models.append(m.eval())
        return models, w["u_max"], {"weights": weights, "sha256": sha256(weights),
                                    "trained_at_commit": w["commit"]}
    from mint.domains import get_domain
    src = G.source_packs("ig")
    u_max = max(p["unit_x"].shape[1] for p in src)
    car = [list(f.carriers) for f in get_domain(G.SOURCE).spec.factors]
    seeds, epochs = ((0,), 2) if smoke else (SEEDS, 40)
    models = [G.train(src[:G.N_FIT], car, u_max, s, epochs=epochs) for s in seeds]
    if smoke:
        return models, u_max, {"weights": None}
    dims = (src[0]["unit_x"].shape[-1], src[0]["pos_x"].shape[-1], src[0]["global_x"].shape[-1])
    torch.save({"state_dicts": [m.state_dict() for m in models], "dims": dims, "u_max": u_max,
                "seeds": list(seeds), "epochs": epochs, "commit": git("rev-parse", "HEAD")},
               weights)
    return models, u_max, {"weights": weights, "sha256": sha256(weights),
                           "trained_at_commit": git("rev-parse", "HEAD")}


def one(target: E.Target, template: str, models, u_max: int, n: int) -> dict:
    E.TEMPLATE = template
    rng = np.random.default_rng(SEED)
    focal = E.sample(n, rng, attract=0.0)
    probe = E.sample(N_PROBE, rng, attract=0.0)
    tgt = G.target_pack("ig", target, focal, probe)
    keep = tgt["margin"] > MARGIN
    masks = G.ext_masks(tgt, keep)
    car = [list(c) for _, c, _ in E.factors()]
    y = tgt["y"][keep]
    ok = np.mean([G.name(m, tgt, car, u_max, keep) == y for m in models], 0)
    live = (np.abs(tgt["factor_total"]) > 1e-9)[keep]
    live_p = np.where(live[np.arange(len(y)), y], 1.0 / np.maximum(live.sum(1), 1), 0.0)
    K = len(E.factors())
    base = target.contrast(target.ids(focal))
    f_ref = target.contrast(target.ids([E.all_neutral(s) for s in focal]))
    resid = np.abs(tgt["fo"].sum(1) - (f_ref - base)) / max(tgt["logit_std"], 1e-9)
    single = tgt["world_single"][keep]
    arms = {"interpreter": G.ext_cells(ok, masks),
            "live": G.ext_cells(live_p, masks),
            "random": {c: 1.0 / K for c in CELLS},
            "grammar_oracle": {c: (float((single[m] == y[m]).mean())
                                   if m.any() and c.startswith("SINGLE") else float("nan"))
                               for c, m in masks.items()}}
    delta, p = paired_sign_flip(ok, live_p)
    example = None
    if (masks["NONE"] | masks["SINGLE_DISAGREE"]).any():
        j = int(np.flatnonzero(masks["NONE"] | masks["SINGLE_DISAGREE"])[0])
        s = [f for f, kk in zip(focal, keep) if kk][j]
        example = {"sentence": " ".join(E.words(s)) + " ...",
                   "B_rests_on": E.factors()[int(y[j])][0],
                   "interpreter_named": [E.factors()[int(G.name(m, tgt, car, u_max, keep)[j])][0]
                                         for m in models]}
    return {"n_states": {c: int(m.sum()) for c, m in masks.items()},
            "n_sentences": len(focal), "factors": [f for f, _, _ in E.factors()],
            "arms": arms, "test": {"delta": delta, "p": p},
            "completeness_residual": [float(resid.mean()), float(np.percentile(resid, 95))],
            "logit_std": tgt["logit_std"], "example": example}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--templates", nargs="+", default=list(TEMPLATES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    t0 = time.time()
    here = os.path.relpath(__file__)
    prov = {"script": here, "script_sha256": sha256(__file__),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if a.smoke:
        a.models, a.templates, n = ["gpt2"], ["rc_obj"], 24
    else:
        n = N_FOCAL
        if os.path.exists(a.out):
            print(f"{a.out} already exists; a run is never overwritten. Refusing.")
            return 2
        if os.path.isdir(".git"):
            committed = git("hash-object", here) == git("rev-parse", f"HEAD:{here}")
            dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
            if not committed or dirty:
                print("the protocol must be committed before the run and the tree clean. Stopping.")
                return 3
            prov |= {"commit": git("rev-parse", "HEAD"),
                     "protocol_commit": git("log", "-1", "--format=%H", "--", here)}
        else:
            print("no git checkout: recording this file's sha256 for comparison with the "
                  "committed protocol")
    models, u_max, wprov = interpreters(a.weights, a.smoke)
    print(f"interpreter ready ({time.time() - t0:.0f}s)", flush=True)

    results, pv, keys = {}, [], []
    dtype = getattr(torch, a.dtype)
    for name in a.models:
        target = E.Target(name, dtype=dtype, device=a.device)
        print(f"{name}: {target.n_params / 1e6:.0f}M parameters ({time.time() - t0:.0f}s)",
              flush=True)
        for t in a.templates:
            r = one(target, t, models, u_max, n)
            results.setdefault(name, {})[t] = r
            pv.append(r["test"]["p"])
            keys.append((name, t))
            c = r["arms"]
            print(f"  {t:8s} states {r['n_states']}  interpreter "
                  + "  ".join(f"{k} {c['interpreter'][k]:.3f}" for k in CELLS
                              if r["n_states"][k] > 0)
                  + f"  | live ALL {c['live']['ALL']:.3f}, chance {c['random']['ALL']:.3f}"
                  + f"  | completeness {r['completeness_residual'][0]:.4f}"
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        del target
    for (name, t), adj in zip(keys, holm_bonferroni(pv, 0.05)):
        r = results[name][t]
        r["test"].update(adj)
        r["read_out"] = ("above the live floor" if adj["reject"] and r["test"]["delta"] > 0
                         else "not above the live floor")
    print("\nread-out:")
    for name, t in keys:
        r = results[name][t]
        print(f"  {name:30s}{t:9s}{r['read_out']}  (delta {r['test']['delta']:+.3f}, "
              f"p_adj {r['test']['p_adj']:.4f})")
    if a.smoke:
        print(f"smoke run done, nothing written ({time.time() - t0:.0f}s)")
        return 0
    out = {"provenance": prov | {"interpreter": wprov, "device": a.device, "dtype": a.dtype,
                                 "n_focal": n, "n_probe": N_PROBE, "seed": SEED,
                                 "margin": MARGIN, "attract": 0.0, "steps": G.STEPS,
                                 "torch": torch.__version__, "numpy": np.__version__,
                                 "python": platform.python_version()},
           "results": results, "wall_seconds": time.time() - t0}
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

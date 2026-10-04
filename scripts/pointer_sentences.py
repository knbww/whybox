#!/usr/bin/env python
"""Section 3.7's pretrained language models: the integrated-gradient pointer, and the
interpreter on the sentences where the pointer is wrong.

PROTOCOL. Committed to the repository before this script is run.
---------------------------------------------------------------------------
Why.        The author's decision of 2026-10-04: section 3.7 reports, next to the
            interpreter, the pointer built from the same signal the interpreter reads. The
            read-out files of section 3.7 (results/sentences_tricks*.json) keep accuracies
            only, so the signal is computed again.
Pointer.    For every factor f with positions C_f: the sum over C_f of the integrated
            gradient of B's output, logit(" are") minus logit(" is"), along the 16-segment
            path from the sentence to its all-neutral version; the pointer names the factor
            with the largest absolute sum. Ties go to the first factor.
Method.     scripts/sentences_tricks.py's pipeline unchanged: random stream 0, 160 focal
            and 256 probe sentences, attract 0, templates rc_obj, rc_subj and wide, 16 path
            segments, margin 0.10, the saved interpreter results/sentences_tricks_interpreter.pt
            (sha256 82a6eb7e...). Cells as section 3.7: ALL, SINGLE_AGREE, SINGLE_DISAGREE,
            NONE, OVER.
CONTROL.    Per model and template, the recomputed state counts and interpreter accuracy in
            every cell are compared with the read-out file of that model. On the same device
            type as the read-out run and on a CPU: equal to 1e-9. Otherwise (GPU kernels are
            not bit-reproducible): every count and every cell within two sentences. A model
            that fails is reported as failed and its pointer is not scored.
Reported.   Per model, template and cell: the pointer's accuracy, the interpreter's accuracy
            on sentences where the pointer is wrong, the pointer's accuracy on sentences where
            the interpreter is wrong (fewer than two of its three seeds right), and the
            counts; pooled over the models and templates of the run. Descriptive; no test;
            nothing enters a read-out.
Models.     gpt2 and Qwen/Qwen2.5-0.5B-Instruct on a CPU (their read-outs ran on a CPU);
            gpt2-large, Qwen/Qwen2.5-1.5B-Instruct and openai-community/gpt2-xl (float32),
            Qwen/Qwen2.5-3B-Instruct (float16), Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B and
            Qwen/Qwen3-14B (4-bit) on a GPU with the settings of their read-out runs.
Run.        PYTHONPATH=.:src:scripts python scripts/pointer_sentences.py --models ... --out ...
            With a git checkout it refuses to run unless this file equals the committed blob
            and no tracked file is modified; it never overwrites a result. `--smoke`: gpt2,
            rc_obj, 24 sentences, no CONTROL, nothing written.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

import external_gpt2 as G
import external_lm as E
import sentences_tricks as ST
from follows_b import MARGIN

READOUT = {"gpt2": "results/sentences_tricks.json",
           "Qwen/Qwen2.5-0.5B-Instruct": "results/sentences_tricks.json",
           "gpt2-large": "results/sentences_tricks_colab.json",
           "Qwen/Qwen2.5-1.5B-Instruct": "results/sentences_tricks_colab.json",
           "Qwen/Qwen2.5-3B-Instruct": "results/sentences_tricks_colab_3b.json",
           "openai-community/gpt2-xl": "results/sentences_tricks_colab_gpt2xl.json",
           "Qwen/Qwen2.5-7B-Instruct": "results/sentences_tricks_colab_qwen25_7b.json",
           "Qwen/Qwen3-8B": "results/sentences_tricks_colab_qwen3_8b.json",
           "Qwen/Qwen3-14B": "results/sentences_tricks_colab_qwen3_14b.json"}


def acc(ok, m) -> float:
    return float(np.asarray(ok, float)[m].mean()) if m.any() else float("nan")


def one(target, template: str, models, u_max: int, n: int) -> dict:
    E.TEMPLATE = template
    rng = np.random.default_rng(ST.SEED)
    focal = E.sample(n, rng, attract=0.0)
    probe = E.sample(ST.N_PROBE, rng, attract=0.0)
    tgt = G.target_pack("ig", target, focal, probe)
    keep = tgt["margin"] > MARGIN
    masks = G.ext_masks(tgt, keep)
    car = [list(c) for _, c, _ in E.factors()]
    y = tgt["y"][keep]
    ok = np.mean([G.name(m, tgt, car, u_max, keep) == y for m in models], 0)
    fo = tgt["fo"][keep]
    pointer = np.stack([np.abs(fo[:, c].sum(1)) for c in car], 1).argmax(1)
    pw, iw = pointer != y, ok < 0.5
    return {"n_states": {c: int(m.sum()) for c, m in masks.items()},
            "interpreter": {c: acc(ok, m) for c, m in masks.items()},
            "pointer": {c: acc(pointer == y, m) for c, m in masks.items()},
            "interp_where_pointer_wrong": {c: acc(ok, m & pw) for c, m in masks.items()},
            "pointer_where_interp_wrong": {c: acc(pointer == y, m & iw) for c, m in masks.items()},
            "n_pointer_wrong": {c: int((m & pw).sum()) for c, m in masks.items()},
            "n_interp_wrong": {c: int((m & iw).sum()) for c, m in masks.items()},
            "_ok": ok, "_pointer_ok": (pointer == y).astype(float), "_masks": masks}


def control(r: dict, ref: dict, exact: bool) -> list[str]:
    bad = []
    for c, n in r["n_states"].items():
        rn = ref["n_states"][c]
        if (n != rn) if exact else abs(n - rn) > 2:
            bad.append(f"n_states {c}: {n} vs {rn}")
        a, b = r["interpreter"][c], ref["arms"]["interpreter"][c]
        if not (np.isfinite(a) or np.isfinite(b)):
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            if n > 0 and rn > 0:
                bad.append(f"interpreter {c}: {a} vs {b}")
            continue
        tol = 1e-9 if exact else 2.0 / max(min(n, rn), 1) + 1e-9
        if abs(a - b) > tol:
            bad.append(f"interpreter {c}: {a:.4f} vs {b:.4f}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B-Instruct"])
    ap.add_argument("--templates", nargs="+", default=list(ST.TEMPLATES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out", default="results/pointer_sentences.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    t0 = time.time()
    here = os.path.relpath(__file__)
    prov = {"script": here, "script_sha256": ST.sha256(__file__),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if a.smoke:
        a.models, a.templates, n = ["gpt2"], ["rc_obj"], 24
    else:
        n = ST.N_FOCAL
        if os.path.exists(a.out):
            print(f"{a.out} already exists; a run is never overwritten. Refusing.")
            return 2
        unknown = [m for m in a.models if m not in READOUT]
        if unknown:
            print(f"no read-out file for {unknown}; this protocol covers section 3.7's models only.")
            return 2
        if os.path.isdir(".git"):
            committed = ST.git("hash-object", here) == ST.git("rev-parse", f"HEAD:{here}")
            dirty = bool(ST.git("status", "--porcelain", "--untracked-files=no"))
            if not committed or dirty:
                print("the protocol must be committed before the run and the tree clean. Stopping.")
                return 3
            prov |= {"commit": ST.git("rev-parse", "HEAD"),
                     "protocol_commit": ST.git("log", "-1", "--format=%H", "--", here)}
        else:
            print("no git checkout: recording this file's sha256 for comparison with the "
                  "committed protocol")
    models, u_max, wprov = ST.interpreters(ST.WEIGHTS, a.smoke)
    G.STEPS = 16
    print(f"interpreter ready ({time.time() - t0:.0f}s)", flush=True)

    results, pooled = {}, {"ok": [], "pointer_ok": [], "masks": []}
    dtype = getattr(torch, a.dtype)
    for name in a.models:
        ref_file = None if a.smoke else json.load(open(READOUT[name]))
        exact = (not a.smoke and a.device == "cpu"
                 and ref_file["provenance"].get("device", "cpu") == "cpu")
        target = E.Target(name, dtype=dtype, device=a.device, load_in_4bit=a.load_in_4bit)
        print(f"{name}: {target.n_params / 1e6:.0f}M parameters ({time.time() - t0:.0f}s)", flush=True)
        for t in a.templates:
            r = one(target, t, models, u_max, n)
            bad = [] if a.smoke else control(r, ref_file["results"][name][t], exact)
            ok, pok, masks = r.pop("_ok"), r.pop("_pointer_ok"), r.pop("_masks")
            r["control"] = {"exact": exact, "failures": bad, "passed": not bad}
            if bad:
                r = {"control": r["control"], "n_states": r["n_states"]}
                print(f"  {t:8s} CONTROL FAILED: {bad}", flush=True)
            else:
                pooled["ok"].append(ok); pooled["pointer_ok"].append(pok); pooled["masks"].append(masks)
                print(f"  {t:8s} interpreter ALL {r['interpreter']['ALL']:.3f}  pointer ALL "
                      f"{r['pointer']['ALL']:.3f}  interpreter where pointer wrong "
                      f"{r['interp_where_pointer_wrong']['ALL']:.3f} (n {r['n_pointer_wrong']['ALL']})"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
            results.setdefault(name, {})[t] = r
        del target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pool = {}
    if pooled["ok"]:
        for c in pooled["masks"][0]:
            m = np.concatenate([x[c] for x in pooled["masks"]])
            ok = np.concatenate(pooled["ok"]); pok = np.concatenate(pooled["pointer_ok"])
            pool[c] = {"n": int(m.sum()), "interpreter": acc(ok, m), "pointer": acc(pok, m),
                       "interp_where_pointer_wrong": acc(ok, m & (pok < 0.5)),
                       "pointer_where_interp_wrong": acc(pok, m & (ok < 0.5)),
                       "n_pointer_wrong": int((m & (pok < 0.5)).sum()),
                       "n_interp_wrong": int((m & (ok < 0.5)).sum())}
        print("pooled:", json.dumps(pool, default=float))
    if a.smoke:
        print(f"smoke run done, nothing written ({time.time() - t0:.0f}s)")
        return 0
    out = {"provenance": prov | {"interpreter": wprov, "device": a.device, "dtype": a.dtype,
                                 "load_in_4bit": a.load_in_4bit, "n_focal": n,
                                 "n_probe": ST.N_PROBE, "seed": ST.SEED, "margin": MARGIN,
                                 "steps": G.STEPS, "torch": torch.__version__,
                                 "numpy": np.__version__, "python": platform.python_version()},
           "results": results, "pooled": pool, "wall_seconds": time.time() - t0}
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

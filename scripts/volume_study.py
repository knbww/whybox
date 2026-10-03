#!/usr/bin/env python
"""Does naming the cause improve with more training volume, at a fixed interpreter?

Protocol: `docs/PREREG_VOLUME.md`. The full run refuses to start unless that file
and `results/volume_manifest.json` are committed and unmodified.

Why this run exists. A diagnostic on the twelve fitted models only, made before the
protocol, found that the confirmed interpreter can memorise its training states
(0.999 without regularisation) while its accuracy on fitted models it was not
trained on falls from 0.73 to 0.64 as training runs longer. That is the signature
of too little training data, not of too few parameters.

Two axes of volume, matched in total states at every rung. Nothing else moves: the
interpreter is `follows_b.Pointer` at its confirmed size (0.515M) and
hyperparameters, the causes and the world are unchanged, and every cell is scored
on the same twelve pinned held-out target models and states as
`results/follows_b.json`.

    rung   total states   states axis         models axis
    0          1920       12 models x 160     (shared)
    1          3840       12 x 320            24 x 160
    2          7680       12 x 640            48 x 160
    3         15360       12 x 1280           96 x 160

Training length is chosen per cell on twelve separate validation models that are
never trained on and never scored: train up to MAX_EPOCHS, keep the checkpoint with
the best validation accuracy, stop after PATIENCE epochs without improvement. A
fixed epoch count would be unfair to the larger rungs, because epoch count alone
moves accuracy on unseen models by 0.09 on this task.

    python -m scripts.volume_study                 # full run, protocol must be committed
    python -m scripts.volume_study --smoke         # 2 models, 2 epochs, end-to-end
    python -m scripts.volume_study --probe         # peak memory and s/epoch of the largest cell
"""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mint.domains import get_domain
from mint.eval.stats import holm_bonferroni, paired_sign_flip
from scripts.follows_b import MARGIN, Pointer, load_population, prepare, recover_name, tensors

DOMAIN = "skirmish"
PREREG = "docs/PREREG_VOLUME.md"
FULL = dict(manifest="results/volume_manifest.json", out="results/volume_study.json",
            base_models=12, base_states=160, rungs=(0, 1, 2, 3), seeds=(0, 1, 2),
            max_epochs=200, patience=20, n_held=12, workers=5, threads=4)
SMOKE = dict(manifest="results/cache/volume_smoke/manifest.json",
             out="results/cache/volume_smoke/volume_study_smoke.json",
             base_models=1, base_states=8, rungs=(0, 1), seeds=(0,),
             max_epochs=2, patience=1, n_held=2, workers=2, threads=2)
BATCH, LR, WD = 64, 1e-3, 1e-2          # follows_b.train_pointer, unchanged
FAMILY = [("states", "ALL"), ("states", "DISAGREE"), ("models", "ALL"), ("models", "DISAGREE")]
SUBSETS = ("ALL", "AGREE", "DISAGREE")

_G: dict = {}                            # filled by load_all, in the parent and in every worker


def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def committed_and_clean(path: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path]).returncode == 0
    return tracked and clean


def load_manifest(path: str) -> dict:
    m = json.loads(Path(path).read_text())
    for name, f in m["files"].items():
        h = hashlib.sha256()
        with open(f["path"], "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != f["sha256"]:
            raise RuntimeError(f"{f['path']}: sha256 does not match {path}")
    return m


def head(pack: dict, n: int) -> dict:
    """The first n states of a model -- nested, so rungs differ only in count."""
    return {k: v[:n] for k, v in pack.items()}


def cells(c: dict) -> list[tuple[str, int, list[dict]]]:
    """(axis, rung, training packs). Rung 0 is one cell shared by both axes."""
    fit, new = _G["fit"], _G["new"]
    bm, bs = c["base_models"], c["base_states"]
    out = [("base", 0, [head(p, bs) for p in fit[:bm]])]
    for r in c["rungs"][1:]:
        out.append(("states", r, [head(p, bs * 2 ** r) for p in fit[:bm]]))
        extra = bm * (2 ** r - 1)
        if extra > len(new):
            raise ValueError(f"models axis rung {r} needs {extra} new models, have {len(new)}")
        out.append(("models", r, [head(p, bs) for p in fit[:bm]] + [head(p, bs) for p in new[:extra]]))
    return out


def batched_names(m, tr, spec, chunk: int = 1024) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([recover_name(m(*(t[i:i + chunk] for t in tr[:4])).numpy(), spec)
                               for i in range(0, len(tr[5]), chunk)])


def fit(tr, va, spec, seed: int, max_epochs: int, patience: int, quiet_probe: bool = False):
    """`follows_b.train_pointer`, plus checkpoint selection on the validation models."""
    torch.manual_seed(seed)
    ux, um, px, gx, carr, _ = tr
    m = Pointer(ux.shape[-1], px.shape[-1], gx.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, since, ep, t_epoch = -1.0, 0, None, 0, 0, []
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        m.train()
        perm = torch.randperm(len(carr), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = F.binary_cross_entropy_with_logits(m(ux[b], um[b], px[b], gx[b]), carr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        acc = float((batched_names(m, va, spec) == va[5]).mean())
        t_epoch.append(time.time() - t0)
        if acc > best:
            best, best_ep, best_state, since = acc, ep, copy.deepcopy(m.state_dict()), 0
        else:
            since += 1
        if since >= patience:
            break
    m.load_state_dict(best_state)
    m.eval()
    log = {"best_epoch": best_ep, "epochs_run": ep, "s_per_epoch": float(np.mean(t_epoch))}
    if not quiet_probe:
        log |= {"val_accuracy": best,
                "train_accuracy": float((batched_names(m, tr, spec) == tr[5]).mean())}
    return m, log


def _worker_init(c: dict, smoke: bool) -> None:
    # Spawned, not forked: a pool forked after torch has run in the parent hangs
    # (caught by the smoke test), so each worker loads and prepares its own copy.
    torch.set_num_threads(c["threads"])
    load_all(c, smoke)


def job(args):
    axis, rung, seed, c = args
    spec, u_max = _G["spec"], _G["u_max"]
    packs = next(p for a, r, p in _G["cells"] if a == axis and r == rung)
    tr = tensors(packs, u_max, spec, MARGIN)
    va = tensors(_G["val"], u_max, spec, MARGIN)
    t0 = time.time()
    m, log = fit(tr, va, spec, seed, c["max_epochs"], c["patience"])
    preds = [batched_names(m, tensors([p], u_max, spec, MARGIN), spec) for p in _G["held"]]
    log |= {"axis": axis, "rung": rung, "seed": seed, "n_models": len(packs),
            "n_states": int(sum(len(p["y"]) for p in packs)), "n_train_states": int(len(tr[5])),
            "wall_seconds": time.time() - t0,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    print(f"  done {axis:6s} rung {rung} seed {seed}: {log['n_train_states']} states, "
          f"epoch {log['best_epoch']}/{log['epochs_run']} ({log['wall_seconds']:.0f}s)", flush=True)
    return (axis, rung, seed), preds, log


def per_model(preds_by_seed: list[list[np.ndarray]]) -> dict[str, list[float]]:
    """Accuracy per held-out model, averaged over interpreter seeds before pairing."""
    out = {s: [] for s in SUBSETS}
    for j, p in enumerate(_G["held"]):
        k = p["margin"] > MARGIN
        y, agree = p["y"][k], p["world"][k] == p["y"][k]
        acc = np.mean([seed_preds[j] == y for seed_preds in preds_by_seed], axis=0)
        for s, msk in (("ALL", np.ones_like(agree)), ("AGREE", agree), ("DISAGREE", ~agree)):
            out[s].append(float(acc[msk].mean()) if msk.sum() else float("nan"))
    return out


def load_all(c: dict, smoke: bool) -> dict:
    d = get_domain(DOMAIN)
    man = load_manifest(c["manifest"])
    if man["smoke"] != smoke:
        raise RuntimeError(f"{c['manifest']} was built with smoke={man['smoke']}")
    held_b, held_path, held_sha = load_population(DOMAIN, 24)
    held_b = held_b[12:12 + c["n_held"]]
    assert {b.trained.cfg.seed for b in held_b} <= {4, 5, 6, 7}
    f = man["files"]
    _G.update(spec=d.spec, manifest=man, held_cache=(held_path, held_sha),
              fit=[prepare(b, d) for b in torch.load(f["fit_states"]["path"], weights_only=False)],
              new=[prepare(b, d) for b in torch.load(f["new_models"]["path"], weights_only=False)],
              val=[prepare(b, d) for b in torch.load(f["val_models"]["path"], weights_only=False)],
              held=[prepare(b, d) for b in held_b])
    train_seeds = set(f["fit_states"]["seeds"]) | set(f["new_models"]["seeds"]) | set(f["val_models"]["seeds"])
    assert not train_seeds & {4, 5, 6, 7}, "held-out seed in a training or validation population"
    _G["u_max"] = max(p["unit_x"].shape[1] for k in ("fit", "new", "val", "held") for p in _G[k])
    _G["cells"] = cells(c)
    return man


def probe(c: dict) -> int:
    """Peak memory and seconds per epoch of the largest cell, two epochs, nothing scored."""
    torch.set_num_threads(c["threads"])
    t0 = time.time()
    load_all(c, smoke=False)
    rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    spec, u_max = _G["spec"], _G["u_max"]
    top = max(c["rungs"])
    for axis in ("states", "models"):
        packs = next(p for a, r, p in _G["cells"] if a == axis and r == top)
        tr = tensors(packs, u_max, spec, MARGIN)
        va = tensors(_G["val"], u_max, spec, MARGIN)
        _, log = fit(tr, va, spec, 0, 2, 99, quiet_probe=True)
        print(f"{axis} rung {top}: {len(tr[5])} training states, {log['s_per_epoch']:.1f} s/epoch "
              f"at {c['threads']} threads", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak RSS: {rss_loaded:.0f} MB after loading, {rss:.0f} MB after training "
          f"({time.time() - t0:.0f}s)")
    return 0


def main(smoke: bool = False) -> int:
    c = SMOKE if smoke else FULL
    torch.set_num_threads(c["threads"])
    if not smoke:
        for path in (PREREG, c["manifest"], "scripts/volume_study.py", "scripts/volume_build.py",
                     "scripts/follows_b.py"):
            if not committed_and_clean(path):
                print(f"refusing to run: {path} is not committed, or has uncommitted changes")
                return 2
    t0 = time.time()
    man = load_all(c, smoke)
    print(f"loaded: {len(_G['fit'])} fitted, {len(_G['new'])} new, {len(_G['val'])} validation, "
          f"{len(_G['held'])} held-out models ({time.time() - t0:.0f}s)", flush=True)

    jobs = [(a, r, s, c) for a, r, _ in _G["cells"] for s in c["seeds"]]
    with mp.get_context("spawn").Pool(c["workers"], initializer=_worker_init,
                                      initargs=(c, smoke)) as pool:
        done = pool.map(job, jobs, chunksize=1)
    preds = {key: p for key, p, _ in done}
    logs = [log for _, _, log in done]

    rows, curve = [], {}
    for axis in ("states", "models"):
        for r in c["rungs"]:
            src = "base" if r == 0 else axis
            pm = per_model([preds[(src, r, s)] for s in c["seeds"]])
            curve[(axis, r)] = pm
            for sub in SUBSETS:
                v = [x for x in pm[sub] if np.isfinite(x)]
                rows.append({"axis": axis, "rung": r, "cell": sub, "n_models": len(v),
                             "total_states": c["base_models"] * c["base_states"] * 2 ** r,
                             "per_model": [{"accuracy": x} for x in pm[sub]],
                             "accuracy": float(np.mean(v)) if v else float("nan")})

    top = max(c["rungs"])
    stats, pv = [], []
    for axis, sub in FAMILY:
        a, b = np.array(curve[(axis, top)][sub]), np.array(curve[(axis, 0)][sub])
        delta, p = paired_sign_flip(a, b)
        stats.append({"role": "primary", "axis": axis, "cell": sub,
                      "comparison": f"rung {top} vs rung 0", "n_models": int(np.isfinite(a - b).sum()),
                      "delta": delta, "a": float(np.nanmean(a)), "b": float(np.nanmean(b))})
        pv.append(p)
    for r, adj in zip(stats, holm_bonferroni(pv, 0.05)):
        r.update(adj)
        r["reading"] = ("rises" if r["reject"] and r["delta"] > 0 else
                        "falls" if r["reject"] and r["delta"] < 0 else "no detectable change")

    secondary = []
    for axis, sub in FAMILY:
        acc = np.array([curve[(axis, r)][sub] for r in c["rungs"]])          # (rungs, models)
        slopes = [float(np.polyfit(np.array(c["rungs"], float), acc[:, j], 1)[0])
                  if np.isfinite(acc[:, j]).all() else float("nan") for j in range(acc.shape[1])]
        mean, p = paired_sign_flip(np.array(slopes), np.zeros(len(slopes)))
        secondary.append({"role": "secondary", "axis": axis, "cell": sub,
                          "statistic": "slope per doubling of volume, per held-out model",
                          "per_model": slopes, "mean": mean, "p_unadjusted": p})
    diagnostic = []
    for sub in ("ALL", "DISAGREE"):
        delta, p = paired_sign_flip(np.array(curve[("models", top)][sub]),
                                    np.array(curve[("states", top)][sub]))
        diagnostic.append({"role": "diagnostic", "comparison": f"models axis vs states axis at rung {top}",
                           "cell": sub, "delta": delta, "p_unadjusted": p})

    print(f"\n{'axis':8s}{'rung':>5s}{'states':>8s}{'ALL':>9s}{'DISAGREE':>10s}")
    for axis in ("states", "models"):
        for r in c["rungs"]:
            get = lambda sub: next(x["accuracy"] for x in rows
                                   if x["axis"] == axis and x["rung"] == r and x["cell"] == sub)
            print(f"{axis:8s}{r:>5d}{c['base_models'] * c['base_states'] * 2 ** r:>8d}"
                  f"{get('ALL'):>9.4f}{get('DISAGREE'):>10.4f}")
    print(f"\n{'primary':30s}{'delta':>9s}{'p':>9s}{'p_adj':>9s}  reading")
    for r in stats:
        print(f"{r['axis'] + ' ' + r['cell']:30s}{r['delta']:>+9.4f}{r['p']:>9.4f}{r['p_adj']:>9.4f}  {r['reading']}")

    out = {"provenance": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
                         "prereg": PREREG, "smoke": smoke, "domain": DOMAIN, "margin": MARGIN,
                         "held_out_cache": {"path": _G["held_cache"][0], "sha256": _G["held_cache"][1]},
                         "volume_manifest": c["manifest"], "volume_files": man["files"],
                         "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                         "optimiser": {"batch": BATCH, "lr": LR, "weight_decay": WD},
                         "interpreter_seeds": list(c["seeds"]), "torch": torch.__version__},
           "rows": rows, "comparisons": stats, "secondary": secondary, "diagnostic": diagnostic,
           "training": logs, "wall_seconds": time.time() - t0}
    Path(c["out"]).write_text(json.dumps(out, indent=1))
    print(f"wrote {c['out']} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    sys.exit(probe(FULL) if "--probe" in argv else main(smoke="--smoke" in argv))

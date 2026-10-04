#!/usr/bin/env python
"""Build the target models of the Car Evaluation study. Nothing about the interpreter is
computed here.

Car Evaluation (UCI; Bohanec & Rajkovic, 1988) is used as published: the 1728 rows, the
six attributes, the four-level label. A target model B reads the six attributes
(mint.domains.car) and answers the acceptability level as a number, unacc 0, acc 1,
good 2, vgood 3, trained with squared error on a random 70% of the rows (1210); the other
518 rows are its test set. Its answer is read as the nearest level.

Populations (each initialisation draws its own split; three architectures per
initialisation, the paper's MLP presets (32, 16), (24, 24, 12), (48,)):

    fit    initialisations 0..11,    stream 1001, 160 focal rows each   (36 targets)
    held   initialisations 100..111, stream 1002, all 1728 rows focal   (36 targets)
    skirmish_held  the simulated tactical world, for the reverse transfer:
           initialisations 60..71 x decoy strengths 0, 1, 2, stream 606, the exact recipe
           of scripts/fresh_build.py (36 targets)

Probe states: 512 rows of the table per target. The builder prints each target's own test
accuracy, a property of B on its own task; no cause label, no cell size and no arm is
computed or printed for the fit or held populations.

    PYTHONPATH=.:src:scripts .venv/bin/python scripts/car_build.py            # writes
        results/cache/car/{fit,held,skirmish_held}.pt and results/car_manifest.json
    ... car_build.py --smoke     # 2 models per population, 2 epochs, results/cache/car_smoke
    ... car_build.py --pilot     # initialisations 900..902 only, never used again: B's
                                 # accuracy per epoch budget and the size of the cells
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mint.causal.interventions import ground_truth
from mint.domains import annotate, get_domain
from mint.encoding import encode_problem
from mint.interpreter.dataset import Bundle
from mint.models.train_target import ARCH_PRESETS, TargetConfig, TrainedTarget, _auc
from mint.models.target import build_target

N_TRAIN, N_PROBE, N_FOCAL_FIT = 1210, 512, 160
EPOCHS, LR, WD, BATCH = 300, 3e-3, 1e-4, 64
ARCHS = ARCH_PRESETS["mlp"]
FIT_INITS, FIT_STREAM = tuple(range(0, 12)), 1001
HELD_INITS, HELD_STREAM = tuple(range(100, 112)), 1002
SK_INITS, SK_STREAM = tuple(range(60, 72)), 606
PILOT_INITS, PILOT_STREAM = (900, 901, 902), 9009
OUT, MANIFEST = Path("results/cache/car"), Path("results/car_manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split(init: int, stream: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng([stream, init]).permutation(1728)
    return np.sort(perm[:N_TRAIN]), np.sort(perm[N_TRAIN:])


def train_car(domain, cfg: TargetConfig, train: np.ndarray, test: np.ndarray):
    """B answers the acceptability level as a number; squared error, AdamW, batch 64."""
    x = torch.as_tensor(domain.to_model_input(domain.states))
    level = torch.as_tensor(domain.logit(domain.states), dtype=torch.float32)
    model = build_target(domain.spec, cfg.arch, cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed)
    tr = torch.as_tensor(train)
    for _ in range(cfg.epochs):
        perm = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = F.mse_loss(model(x[b])[0], level[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        s = model(x)[0].numpy()
    lv = level.numpy()
    pred = np.clip(np.rint(s), 0, 3)
    record = {"train_accuracy": float((pred[train] == lv[train]).mean()),
              "test_accuracy": float((pred[test] == lv[test]).mean()),
              "test_mae": float(np.abs(s[test] - lv[test]).mean()),
              "test_majority_accuracy": float((lv[test] == 0).mean())}
    acceptable = (lv > 0).astype(int)
    tt = TrainedTarget(model, cfg, _auc(acceptable[train], s[train]),
                       _auc(acceptable[test], s[test]), float("nan"))
    return tt, record


def population(domain, inits, stream, n_focal: int | None, epochs: int, archs=ARCHS):
    """n_focal None: every row of the table is a focal state."""
    bundles, records = [], []
    for init in inits:
        train, test = split(init, stream)
        for a, arch in enumerate(archs):
            t0 = time.time()
            cfg = TargetConfig("car", arch, init, spurious_strength=0.0, n_train=N_TRAIN,
                               epochs=epochs, lr=LR, weight_decay=WD)
            tt, rec = train_car(domain, cfg, train, test)
            rng = np.random.default_rng([stream, init, a])
            focal = (domain.states if n_focal is None
                     else domain.states[np.sort(rng.choice(1728, n_focal, replace=False))])
            probe = domain.states[np.sort(rng.choice(1728, N_PROBE, replace=False))]
            gt = ground_truth(tt.model, domain, focal, probe)
            enc = encode_problem(gt, tt.model.param_summary()["n_params"], cfg.tag(), "car")
            bundles.append(Bundle(tt, gt, enc, focal, "car"))
            records.append({"initialisation": init, "arch": a, "widths": list(arch["widths"]),
                            "train_rows": train.tolist(), **rec})
            print(f"  car init {init} arch {a} {str(arch['widths']):14s} test accuracy "
                  f"{rec['test_accuracy']:.3f} (majority {rec['test_majority_accuracy']:.3f}) "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return bundles, records


def pilot() -> int:
    """Initialisations 900..902, used for nothing else: does B solve the task at a given
    epoch budget, and how large are the cells the study needs."""
    d = get_domain("car")
    for epochs in (100, 300):
        bundles, recs = population(d, PILOT_INITS, PILOT_STREAM, None, epochs)
        acc = [r["test_accuracy"] for r in recs]
        print(f"epochs {epochs}: test accuracy {np.mean(acc):.3f} (min {np.min(acc):.3f})")
        causes, keep = [], []
        for b in bundles:
            a = np.abs(b.gt.factor_total) / max(b.gt.probe.logit_std, 1e-9)
            s = np.sort(a, 1)
            causes.append(a.argmax(1)); keep.append(s[:, -1] - s[:, -2] > 0.10)
        ann = annotate(d, d.states)
        named = ann.margin > 0
        for i in range(len(bundles)):
            others = [causes[j] for j in range(len(bundles)) if j != i]
            votes = np.stack([np.bincount(c, minlength=3) for c in np.stack(others, 1)])
            typ = votes.argmax(1)
            k = keep[i]
            dis = named & (causes[i] != ann.factor_id)
            print(f"   target {i}: kept {k.mean():.3f}; world names a cause {named[k].mean():.3f}; "
                  f"departs from world {dis[k].mean():.3f}; from the others {(causes[i] != typ)[k].mean():.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    if a.pilot:
        return pilot()
    out, manifest = (Path("results/cache/car_smoke"), Path("results/cache/car_smoke/manifest.json")) \
        if a.smoke else (OUT, MANIFEST)
    if not a.smoke and (out.exists() or manifest.exists()):
        print(f"{out} or {manifest} already exists; the Car populations are built once.")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = get_domain("car")
    if a.smoke:
        fit_inits, held_inits, sk_inits, epochs, archs = (0, 1), (100, 101), (60, 61), 2, ARCHS[:1]
    else:
        fit_inits, held_inits, sk_inits, epochs, archs = FIT_INITS, HELD_INITS, SK_INITS, EPOCHS, ARCHS
    files, records = {}, {}
    for name, inits, stream, n_focal in (("fit", fit_inits, FIT_STREAM, N_FOCAL_FIT),
                                         ("held", held_inits, HELD_STREAM, None)):
        print(f"{name} ({time.time() - t0:.0f}s)", flush=True)
        bundles, recs = population(d, inits, stream, n_focal, epochs, archs)
        path = out / f"{name}.pt"
        torch.save({"bundles": bundles, "records": recs}, path)
        files[name] = {"path": str(path), "sha256": sha256(path), "n_models": len(bundles),
                       "initialisations": list(inits), "stream": stream,
                       "focal": "all 1728 rows" if n_focal is None else n_focal}
        records[name] = [{k: v for k, v in r.items() if k != "train_rows"} for r in recs]

    print(f"skirmish_held ({time.time() - t0:.0f}s)", flush=True)
    from scripts.volume_build import FULL, SMOKE, new_population
    c = dict(SMOKE if a.smoke else FULL)
    if a.smoke:
        c["spurious"] = (0.0,)
    sk = new_population(get_domain("skirmish"), c, sk_inits, SK_STREAM)
    path = out / "skirmish_held.pt"
    torch.save(sk, path)
    files["skirmish_held"] = {"path": str(path), "sha256": sha256(path), "n_models": len(sk),
                              "initialisations": list(sk_inits), "stream": SK_STREAM,
                              "strengths": list(c["spurious"]),
                              "recipe": {k: c[k] for k in ("n_focal", "n_probe", "n_train",
                                                           "epochs")}}
    records["skirmish_held"] = [{"initialisation": b.trained.cfg.seed,
                                 "strength": float(b.trained.cfg.spurious_strength),
                                 "test_auc": float(b.trained.test_auc)} for b in sk]
    man = {"dataset": "UCI Car Evaluation (Bohanec & Rajkovic, 1988), data/uci/car.data",
           "data_sha256": sha256(Path("data/uci/car.data")),
           "recipe": {"target": "acceptability level 0..3 as a number, squared error",
                      "n_train": N_TRAIN, "n_test": 1728 - N_TRAIN, "epochs": epochs,
                      "lr": LR, "weight_decay": WD, "batch": BATCH, "n_probe": N_PROBE,
                      "archs": [list(x["widths"]) for x in archs]},
           "files": files, "targets": records, "smoke": a.smoke,
           "scored_before_confirmatory_run": False}
    manifest.write_text(json.dumps(man, indent=1))
    print(f"wrote {manifest} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

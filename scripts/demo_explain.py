#!/usr/bin/env python
"""The whole chain on one live state, in words.

    стимул -> модель B -> её ML-сигналы -> интерпретатор A -> причина -> проверка на B

A reads only B's signals -- no raw inputs, no feature names, no list of causes -- and
names the cause B relies on. That is the only thing A asserts. Everything after it
is **measured on B, live**: how far B's output moves when each cause is removed,
which removal moves it most, and what removing A's cause does to B's prediction.
The roles are printed, so a measurement is never mistaken for a prediction of A.

Four variants, one per established result, each next to its committed number:

    base       the main hypothesis: A names the cause B relies on    results/follows_b.json
    dynamic    the same, A also reads B's dynamics along a path      results/signals_study.json
    transfer   A fitted only on medical models, reads tactical ones   results/transfer_b.json
    lm         the target is a small language model                   results/statement_lm.json

The state is chosen by a rule fixed before A's answer is seen, from labels only:
by default the first scored state of the first held-out model -- an ordinary state,
the case the main hypothesis is about. `--disagree` takes the first state where the
cause B relies on differs from the world's instead. `--state K` walks either list.

Run:  .venv/bin/python -m scripts.demo_explain [--variant V] [--disagree] [--state K] [--retrain]
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mint.domains import get_domain
from mint.encoding.dynamics import path_signals
from scripts.follows_b import (MARGIN, Pointer, carriers_in, load_population, prepare,
                               recover_name, tensors, train_pointer)
from scripts.signals_study import arm_view, signal_pack
from scripts.statement import _as_input, executed
from scripts.volume_study import fit, load_manifest

SEED, EPOCHS = 0, 40                     # follows_b recipe
MAX_EPOCHS, PATIENCE, STEPS = 200, 20, 16  # signals_study recipe
CACHE_DIR = Path("results/cache")
RULE = "=" * 74

VARIANTS = {
    "base": dict(source="skirmish", target="skirmish", signals="current",
                 title="ГЛАВНАЯ ГИПОТЕЗА: A по сигналам модели B называет причину, "
                       "на которую опирается сама B",
                 ref=("results/follows_b.json", "interpreter", None, "ALL", "DISAGREE")),
    "dynamic": dict(source="skirmish", target="skirmish", signals="both",
                    title="ДИНАМИЧЕСКИЕ СИГНАЛЫ: то же, но A читает ещё и динамику B "
                          "вдоль пути к нейтральному входу",
                    ref=("results/signals_study.json", "+both", None, "ALL", "DISAGREE")),
    "transfer": dict(source="clinic", target="skirmish", signals="current",
                     title="ПЕРЕНОС: A обучен только на медицинских моделях и читает "
                           "тактические, которых не видел ни разу",
                     ref=("results/transfer_b.json", "interpreter", None,
                          "clinic->skirmish|ALL", "clinic->skirmish|DISAGREE")),
    "lm": dict(source="entail", target="entail", signals="current",
               title="ЯЗЫКОВАЯ МОДЕЛЬ: B — маленький трансформер, который решает "
                     "логическую задачу по предложению",
               ref=("results/statement_lm.json", "interpreter", "cause", "ALL", "DISAGREE")),
}
OUTPUT = {"skirmish": "вероятность победы", "entail": "вероятность ответа «yes»"}


# ------------------------------------------------------------------ data ---
def load_packs(domain_name: str, signals: str, bundles=None):
    """(packs, population sha256, bundles) for a pinned population, or for given bundles."""
    d = get_domain(domain_name)
    sha = ""
    if bundles is None:
        bundles, _, sha = load_population(domain_name, 24)
    if signals == "current":
        return [prepare(b, d) for b in bundles], sha, bundles
    out = []
    for b in bundles:
        sp = signal_pack(b, d, STEPS)
        out.append(arm_view(sp, "+both") | {"fo": sp["fo"]})
    return out, sha, bundles


# ----------------------------------------------------------- interpreter ---
def _key(v: dict, shas: tuple[str, str], u_max: int, dims) -> str:
    """Everything the weights depend on, including the code that defines and trains
    the model, so a cache can never outlive an edit to either."""
    fns = (Pointer, train_pointer, tensors, fit, signal_pack, arm_view, path_signals)
    code = "".join(inspect.getsource(o) for o in fns)
    payload = {"variant": {k: v[k] for k in ("source", "target", "signals")}, "populations": shas,
               "u_max": u_max, "dims": list(dims), "seed": SEED, "epochs": EPOCHS,
               "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "steps": STEPS, "margin": MARGIN,
               "code": hashlib.sha256(code.encode()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def get_interpreter(name: str, v: dict, src, src_sha, tgt_sha, u_max, retrain: bool):
    spec = get_domain(v["source"]).spec
    tr = tensors(src[:12], u_max, spec, MARGIN)
    dims = (tr[0].shape[-1], tr[2].shape[-1], tr[3].shape[-1])
    key = _key(v, (src_sha, tgt_sha), u_max, dims)
    path = CACHE_DIR / f"demo-{name}-{key}.pt"
    if path.exists() and not retrain:
        blob = torch.load(path, weights_only=False)
        m = Pointer(*dims)
        m.load_state_dict(blob["state_dict"])
        print(f"интерпретатор загружен из кэша: обучен {blob['trained_at']} за "
              f"{blob['train_seconds']:.0f} с, ключ {key}")
        return m.eval()
    print(f"обучаю интерпретатор на 12 моделях {v['source']} (от минуты до нескольких)...", flush=True)
    t0 = time.time()
    if v["signals"] == "current":
        m = train_pointer(tr, spec, seed=SEED, epochs=EPOCHS)
    else:
        man = load_manifest("results/volume_manifest.json")
        val_b = torch.load(man["files"]["val_models"]["path"], weights_only=False)
        val, _, _ = load_packs(v["source"], v["signals"], val_b)
        m, _ = fit(tr, tensors(val, u_max, spec, MARGIN), spec, SEED, MAX_EPOCHS, PATIENCE)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "trained_at": datetime.now().isoformat(" ", "seconds"),
                "train_seconds": time.time() - t0, "key": key, "variant": name}, path)
    print(f"обучен за {time.time() - t0:.0f} с и сохранён, ключ {key}")
    return m.eval()


@torch.no_grad()
def scores_of(m, pack, u_max, spec) -> np.ndarray:
    """(scored states, input positions): A's score for every input of B."""
    return m(*tensors([pack], u_max, spec, MARGIN)[:4]).numpy()


def pick(packs, index: int, disagree: bool) -> tuple[int, int] | None:
    """(model, position among scored states) of the index-th held-out state in the
    chosen list. Uses labels only, never A's answer."""
    seen = 0
    for i in range(12, len(packs)):
        p = packs[i]
        for j, s in enumerate(np.nonzero(p["margin"] > MARGIN)[0]):
            if disagree and p["y"][s] == p["world"][s]:
                continue
            if seen == index:
                return i, j
            seen += 1
    return None


# --------------------------------------------------------------- summary ---
def reference(v: dict) -> tuple[float, float, str]:
    path, method, metric, all_cell, dis_cell = v["ref"]
    rows = json.loads(Path(path).read_text())["rows"]
    get = lambda cell: next(r["accuracy"] for r in rows if r["method"] == method
                            and r["cell"] == cell and r.get("metric") == metric)
    return get(all_cell), get(dis_cell), path


def summary(m, v, src, tgt, u_max) -> None:
    src_spec, spec = get_domain(v["source"]).spec, get_domain(v["target"]).spec
    ys = np.concatenate([p["y"][p["margin"] > MARGIN] for p in src[:12]])
    maj = min(int(np.bincount(ys, minlength=src_spec.n_factors).argmax()), spec.n_factors - 1)
    rng = np.random.default_rng(0)
    acc: dict[tuple[str, str], list[float]] = {}
    for p in tgt[12:]:
        k = p["margin"] > MARGIN
        y, dis = p["y"][k], p["world"][k] != p["y"][k]
        # Two aggregation rules over the same first-order term, named apart. The
        # signed sum is the linearisation of the very intervention the label is
        # measured by, so it is the strong one wherever a cause has several
        # carriers; showing only the weak one made the demo disagree with the
        # paper's baseline row under one name. Both come from `scripts/floors.py`.
        car = carriers_in(spec, p["fo"].shape[1])
        fo = p["fo"][k]
        arms = {"интерпретатор A": recover_name(scores_of(m, p, u_max, spec), spec),
                "указатель, сумма со знаком": np.stack(
                    [np.abs(fo[:, c].sum(1)) for c in car], 1).argmax(1),
                "указатель, средний модуль": recover_name(np.abs(fo), spec),
                "константа": np.full(len(y), maj),
                "случайный выбор": rng.integers(0, spec.n_factors, len(y)),
                "идеальный предсказатель мира": p["world"][k]}
        for nm, pred in arms.items():
            acc.setdefault((nm, "all"), []).append(float((pred == y).mean()))
            if dis.any():
                acc.setdefault((nm, "dis"), []).append(float((pred[dis] == y[dis]).mean()))
    print(f"   {'':30s}{'все состояния':>15s}{'B ≠ мир':>10s}")
    for nm in ("интерпретатор A", "указатель, сумма со знаком", "указатель, средний модуль",
               "константа", "случайный выбор", "идеальный предсказатель мира"):
        print(f"   {nm:30s}{np.mean(acc[(nm, 'all')]):>15.3f}{np.mean(acc[(nm, 'dis')]):>10.3f}")
    ra, rd, path = reference(v)
    print(f"\n   случайный уровень: {1 / spec.n_factors:.3f}.  Демо учит A на одном seed'е;")
    print(f"   подтверждённый результат на трёх seed'ах ({path}): {ra:.3f} и {rd:.3f}.")
    print("   Одно состояние — иллюстрация; доказывает эта таблица.")


# ------------------------------------------------------------------ main ---
def main(variant: str = "base", disagree: bool = False, state: int = 0, retrain: bool = False) -> int:
    torch.set_num_threads(4)
    if variant not in VARIANTS:
        print(f"неизвестный вариант {variant!r}; есть: {', '.join(VARIANTS)}"); return 2
    v = VARIANTS[variant]
    d = get_domain(v["target"]); spec = d.spec
    tgt, tgt_sha, bundles = load_packs(v["target"], v["signals"])
    src, src_sha = ((tgt, tgt_sha) if v["source"] == v["target"]
                    else load_packs(v["source"], v["signals"])[:2])
    u_max = max(p["unit_x"].shape[1] for p in src[:12] + tgt[12:])
    A = get_interpreter(variant, v, src, src_sha, tgt_sha, u_max, retrain)

    hit = pick(tgt, state, disagree)
    if hit is None:
        print("подходящего состояния не нашлось"); return 1
    i, j = hit
    b, p = bundles[i], tgt[i]
    B = b.trained.model
    s = int(np.nonzero(p["margin"] > MARGIN)[0][j])
    row, sd = b.focal_raw[s], float(b.gt.probe.logit_std)
    names = [f.name for f in spec.factors]

    print("\n" + v["title"])
    print(f"модель B №{i} — её интерпретатор A никогда не читал.")
    which = ("где причина, на которую опирается B, не совпадает с причиной, которая решает "
             "исход\nпо законам мира" if disagree else "из всех оцениваемых состояний")
    print(f"правило выбора: {state + 1}-е по порядку {which}. Выбрано по меткам,\n"
          "до того как смотреть на ответ A.")

    print("\n" + RULE + "\n1. СТИМУЛ — человек это видит, интерпретатор A не видит никогда\n" + RULE)
    if spec.kind == "tabular":
        print("   " + ", ".join(f"{n}={val:g}" for n, val in zip(spec.input_names, row)))
    else:
        print("   «" + d.to_sentence(d.to_model_input(row[None])[0]) + "»")

    print("\n" + RULE + "\n2. МОДЕЛЬ B — что она выдаёт\n" + RULE)
    with torch.no_grad():
        lg = float(B(_as_input(d.to_model_input(row[None])))[0].numpy()[0])
    prob = lambda z: 1 / (1 + np.exp(-z))
    print(f"   {OUTPUT[v['target']]} {prob(lg):.3f}")

    print("\n" + RULE + "\n3. ЧТО ПОЛУЧАЕТ A — только сигналы модели B\n" + RULE)
    print(f"   по каждому внутреннему нейрону B: {p['unit_x'].shape[2]} чисел ({p['unit_x'].shape[1]} нейронов)")
    print(f"   по каждому входу B: {p['pos_x'].shape[2]} чисел, общий контекст: {p['global_x'].shape[1]} чисел")
    if v["signals"] == "current":
        print("   всё снято с B в этом состоянии прямым и обратным проходом, без единого")
        print("   вмешательства; ни одного имени признака и никакого списка причин на выбор.")
    else:
        print("   снимок в этом состоянии плюс динамика: B прогоняется на 17 промежуточных входах")
        print("   на пути к нейтральному; ни одна причина не убирается по отдельности;")
        print("   ни одного имени признака и никакого списка причин на выбор.")
    if v["source"] != v["target"]:
        print(f"   A обучался только на моделях {v['source']} и не видел ни одной модели {v['target']}.")

    print("\n" + RULE + "\n4. ОТВЕТ A — на какую причину опирается B\n" + RULE)
    sc = scores_of(A, p, u_max, spec)[j]
    cause = int(recover_name(sc[None], spec)[0])
    car = carriers_in(spec, len(sc))
    print(f"   B опирается на: {', '.join(spec.input_names[c] for c in car[cause])}")
    print(f"   это {names[cause]} — {spec.factors[cause].description}")
    per = sorted(((float(np.mean(sc[c])), names[k]) for k, c in enumerate(car)), reverse=True)[:3]
    print("   баллы A по причинам: " + ", ".join(f"{n} {val:+.2f}" for val, n in per))

    print("\n" + RULE + "\n5. ПРОВЕРКА НА B — измерено, не предсказано\n" + RULE)
    print("   каждую причину по очереди убираем из входа B и смотрим, насколько сдвинулся выход")
    print("   (в единицах разброса выхода B):")
    shift = {k: float(executed(B, d, row[None], np.array([k]), sd)[0]) for k in range(spec.n_factors)}
    order = sorted(shift, key=lambda k: -abs(shift[k]))
    actual, world = order[0], int(p["world"][s])
    for k in order:
        tags = [t for t, on in (("← назвал A", k == cause), ("← на это опирается B", k == actual),
                                ("← решает по законам мира", k == world)) if on]
        print(f"     {names[k]:28s}{shift[k]:+7.2f}   {'  '.join(tags)}")
    print(f"\n   причина, названная A: {'✓ верно' if cause == actual else '✗ неверно'}"
          + ("" if cause == actual else f" — B опирается на {names[actual]}"))
    print(f"   если убрать {names[cause]}, {OUTPUT[v['target']]} по B: {prob(lg):.3f} → "
          f"{prob(lg + shift[cause] * sd):.3f}   (измерено на B)")

    print("\n" + RULE + "\nИТОГ по всем 12 моделям, которых A никогда не читал: причина названа верно\n" + RULE)
    summary(A, v, src, tgt, u_max)
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    opt = lambda flag, default: argv[argv.index(flag) + 1] if flag in argv else default
    sys.exit(main(variant=opt("--variant", "base"), disagree="--disagree" in argv,
                  state=int(opt("--state", 0)), retrain="--retrain" in argv))

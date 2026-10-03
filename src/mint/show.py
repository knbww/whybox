"""Presenting the results: `show`, `verify`, `audit`, `explain`.

Built for showing the work live. Nothing here needs a network or a package beyond
the project's own: colour is plain ANSI and switches off when the output is not a
terminal or `NO_COLOR` is set.

`verify` re-derives every statistic with its **own** implementation of the exact
sign-flip test and of Holm's step-down correction, independent of
`mint.eval.stats`, which produced the stored numbers. A check that reused the code
under test would prove nothing.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RESULTS = [
    ("1", "The interpreter reads the target model, not the world",
     "results/follows_b.json", "docs/PREREG_FOLLOWS_B.md"),
    ("2", "It states a whole claim, every clause executed — tabular model",
     "results/statement.json", "docs/PREREG_STATEMENT.md"),
    ("3", "The same, on a language model",
     "results/statement_lm.json", "docs/PREREG_STATEMENT.md"),
    ("4", "The reading transfers across ontologies",
     "results/transfer_b.json", "docs/PREREG_TRANSFER_B.md"),
]

# ---------------------------------------------------------------- terminal ---
_COLOUR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOUR else s


bold = lambda s: _c("1", s)
dim = lambda s: _c("2", s)
green = lambda s: _c("32", s)
red = lambda s: _c("31", s)
yellow = lambda s: _c("33", s)
cyan = lambda s: _c("36", s)


def _vis(s: str) -> int:
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def table(head: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> str:
    cols = len(head)
    w = [max(_vis(r[i]) for r in [head, *rows]) for i in range(cols)]
    pad = lambda s, i: (" " * (w[i] - _vis(s)) + s) if i in right else (s + " " * (w[i] - _vis(s)))
    line = lambda r: "  " + "   ".join(pad(r[i], i) for i in range(cols))
    rule = "  " + "   ".join("─" * w[i] for i in range(cols))
    return "\n".join([bold(line(head)), dim(rule), *map(line, rows)])


def rule(title: str) -> str:
    return "\n" + bold(cyan("━" * 78)) + "\n" + bold(title) + "\n" + bold(cyan("━" * 78))


def gate_str(g: str) -> str:
    return {"CONFIRMED": green(bold("CONFIRMED")), "FAIL": red(bold("FAIL")),
            "VOID": red(bold("VOID"))}.get(g, yellow(bold(g)))


# ------------------------------------------------- independent statistics ---
def exact_sign_flip(diffs: list[float]) -> float:
    """Two-sided exact paired sign-flip p over every one of the 2^n sign patterns."""
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    hits = sum(abs(sum(s * d for s, d in zip(signs, diffs)) / n) >= obs - 1e-12
               for signs in itertools.product((1, -1), repeat=n))
    return hits / 2 ** n


def holm(pvals: list[float]) -> list[float]:
    """Holm's step-down adjustment, with monotonicity enforced."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# -------------------------------------------------------------- reading ---
def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _row(d: dict, method: str, cell: str, metric: str | None) -> dict:
    """Rows of the pointer runs carry no `metric` (their comparisons say `accuracy`);
    rows of the stated-claim runs carry one per clause and must match it."""
    for r in d["rows"]:
        if r["method"] == method and r["cell"] == cell and r.get("metric", metric) == metric:
            return r
    raise KeyError((method, cell, metric))


def _per_model(d: dict, method: str, cell: str, metric: str | None) -> list[float]:
    r = _row(d, method, cell, metric)
    return [m["accuracy"] for m in r["per_model"]]


# ---------------------------------------------------------------- show ---
def _prov(d: dict) -> str:
    p = d["provenance"]
    dirty = red("dirty") if p.get("dirty") else green("clean")
    sha = p.get("cache_sha256") or next(
        (v["sha256"] for v in p.get("caches", {}).values()), None)
    if sha:
        extra = f"   population sha256 {sha[:12]}"
    elif p.get("cache"):
        extra = f"   population {Path(p['cache']).name} (path recorded, hash not)"
    else:
        extra = "   population not recorded in this file — pinned in scripts/follows_b.py::CACHE"
    return dim(f"  commit {p['commit'][:7]} (") + dirty + dim(")" + extra)


def cmd_show() -> int:
    print(bold("\nmint — causal interpretability from internal model representations"))
    print(dim("Can one AI model read another's internal signals and say why it decided?"))
    for num, title, path, prereg in RESULTS:
        d = _load(path)
        print(rule(f"{num}. {title}"))
        print(f"  gate {gate_str(d['gate'])}    protocol {dim(prereg + ' (git history, commit ' + d['provenance']['commit'][:7] + ')')}")
        print(_prov(d))
        comps = d["comparisons"]
        rows = []
        for c in comps:
            sig = (green("+") if c["delta"] > 0 else red("−")) if c["reject"] else dim("·")
            label = f"{c.get('metric', 'accuracy')} vs {c['reference']}"
            rows.append([label, c["cell"], f"{c['a']:.4f}", f"{c['b']:.4f}",
                         f"{c['delta']:+.4f}", f"{c['p_adj']:.4f}", sig])
        print(table(["comparison", "subset", "interp.", "ref.", "Δ", "p adj", ""],
                    rows, right={2, 3, 4, 5}))
    print(rule("Qualification that travels with every number above"))
    print("  The 12 held-out target models are " + bold("four independent seeds") +
          " at three decoy\n  strengths. Every primary effect is positive in all four seed groups; the\n"
          "  p-values assume twelve independent units and overstate the design's resolution.\n"
          + dim("  results/frozen/ERRATA.md §11"))
    print()
    return 0


# -------------------------------------------------------------- verify ---
def cmd_verify(path: str) -> int:
    d = _load(path)
    comps = d["comparisons"]
    print(rule(f"verify  {path}"))
    print(dim("  every statistic recomputed from the stored per-model values, with an\n"
              "  implementation of the sign-flip test and of Holm independent of the one\n"
              "  that produced them\n"))
    # A result whose units are already independent (it says so in provenance) is not
    # four seeds at three decoy strengths, and regrouping it by threes would print a
    # false seed-level p for every comparison.
    prov = d.get("provenance", {})
    held = prov.get("held_initialisations")
    independent = bool(prov.get("independent_units")) or (
        held is not None and len(held) >= 12 and len(set(held)) == len(held))
    pv, deltas, seed_p, seed_pos = [], [], [], []
    for c in comps:
        a = _per_model(d, "interpreter", c["cell"], c.get("metric"))
        b = _per_model(d, c["reference"], c["cell"], c.get("metric"))
        diffs = [x - y for x, y in zip(a, b)]
        deltas.append(sum(diffs) / len(diffs))
        pv.append(exact_sign_flip(diffs))
        if len(diffs) == 12 and not independent:  # 4 seeds x 3 decoy strengths
            groups = [sum(diffs[i:i + 3]) / 3 for i in range(0, 12, 3)]
            seed_p.append(exact_sign_flip(groups))
            seed_pos.append(sum(g > 0 for g in groups))
        else:
            seed_p.append(None); seed_pos.append(None)
    adj = holm(pv)

    rows, bad = [], 0
    for c, dl, p, pa, sp, spos in zip(comps, deltas, pv, adj, seed_p, seed_pos):
        ok = abs(dl - c["delta"]) < 1e-9 and abs(pa - c["p_adj"]) < 1e-9
        bad += not ok
        rows.append([f"{c.get('metric', 'accuracy')} vs {c['reference']}", c["cell"],
                     f"{dl:+.4f}", f"{p:.5f}", f"{pa:.5f}",
                     green("✓") if ok else red("✗ stored " + f"{c['p_adj']:.5f}"),
                     f"{spos}/4" if spos is not None else "–",
                     f"{sp:.3f}" if sp is not None else "–"])
    print(table(["comparison", "subset", "Δ", "p", "p adj (Holm)", "matches stored",
                 "seeds +", "p (n=4)"], rows, right={2, 3, 4, 6, 7}))
    n = len(comps)
    print(f"\n  {n} comparisons, {bold(green(str(n - bad)) if not bad else red(str(n - bad)))} "
          f"reproduce exactly.   smallest attainable p: n=12 → {2 / 4096:.6f},  n=4 → 0.125")
    if independent:
        print(dim(f"  units are independent ({prov.get('unit', 'initialisation')}); "
                  "no seed grouping applies"))
    else:
        print(dim("  'seeds +' — how many of the four independent seed groups move the same way"))
    return 1 if bad else 0


# --------------------------------------------------------------- audit ---
def cmd_audit() -> int:
    print(rule("audit"))
    failures = 0

    print(bold("\n  1. test suite"))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1] if r.stdout.strip() else ""
    print(f"     {green('✓') if r.returncode == 0 else red('✗')} {tail}   (exit {r.returncode})")
    failures += r.returncode != 0

    print(bold("\n  2. every statistic, recomputed independently"))
    for num, title, path, _ in RESULTS:
        with open(os.devnull, "w") as null:
            saved, sys.stdout = sys.stdout, null
            try:
                code = cmd_verify(path)
            finally:
                sys.stdout = saved
        print(f"     {green('✓') if code == 0 else red('✗')} {path}")
        failures += code != 0

    print(bold("\n  3. every number in the front-page documents, traced to its source"))
    known: dict[str, str] = {}

    def add(v: float, src: str) -> None:
        for dp in (2, 3, 4):
            known.setdefault(f"{v:.{dp}f}", src)

    for _, _, path, _ in RESULTS:
        d = _load(path)
        for r_ in d["rows"] + d.get("sensitivity", []):
            add(r_["accuracy"], path)
        for c in d["comparisons"]:
            for k in ("a", "b", "p", "p_adj"):
                add(c[k], path)
            add(abs(c["delta"]), path)
    # Quantities that are not measurements but follow from the design, and the few
    # numbers established by a separate recorded calculation rather than a run.
    for v, src in ((1 / 7, "chance, 7 causes"), (1 / 5, "chance, 5 causes"),
                   (1 / 4, "chance, 4 causes"), (6 * 2 / 4096, "Holm floor, family of 6"),
                   (2 / 4096, "sign-flip floor, n=12"), (0.125, "sign-flip floor, n=4"),
                   (0.8708, "ERRATA §12"), (0.9016, "ERRATA §12"), (0.7371, "ERRATA §12"),
                   (0.4999, "ERRATA §12"),
                   (0.10, "margin, protocol"), (0.15, "margin, protocol"),
                   (0.75, "magnitude cut, protocol")):
        add(v, src)
    hyb = _load("results/frozen/hybrid_confirm.json")   # cited only as withdrawn
    for c in hyb["comparisons"]:
        add(abs(c["delta"]), "hybrid_confirm (withdrawn)")
    num = re.compile(r"(?<![\d.])0\.\d{2,4}(?!\d)")
    for doc in ("README.md", "README.ru.md"):
        if not (ROOT / doc).exists():
            print(f"     {red('✗')} {doc:24s} missing"); failures += 1; continue
        nums = set(num.findall((ROOT / doc).read_text()))
        missing = sorted(n for n in nums if n not in known)
        derived = sorted({known[n] for n in nums if n in known and not known[n].startswith("results/")})
        print(f"     {green('✓') if not missing else red('✗')} {doc:24s} {len(nums):3d} numbers"
              + (red(f"   untraced: {', '.join(missing)}") if missing else "")
              + (dim(f"   also from: {'; '.join(derived)}") if derived and not missing else ""))
        failures += bool(missing)

    print(bold("\n  4. gates"))
    for num, title, path, _ in RESULTS:
        print(f"     {gate_str(_load(path)['gate'])}  {title}")

    verdict = green(bold("all checks pass")) if not failures else red(bold(f"{failures} failed"))
    print(f"\n  {verdict}\n")
    return 1 if failures else 0


def cmd_explain(retrain: bool = False, state: int = 0, variant: str = "base",
                disagree: bool = False) -> int:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from scripts.demo_explain import main
    return main(variant=variant, disagree=disagree, state=state, retrain=retrain)

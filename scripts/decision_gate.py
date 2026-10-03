#!/usr/bin/env python
"""Apply the frozen decision gate to a run, mechanically.

    python scripts/decision_gate.py results/rerun_core.json

The criteria are those written in docs/PREREG.md before the run.  They are
implemented here rather than read off a table by eye, so that applying them is
not a judgement call and cannot drift toward the result one hoped for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PRIMARY = "executed.sufficiency@3"
CONTROL = "first_order"
METHOD = "A_full"
ALPHA = 0.05


def main(path: str) -> int:
    res = json.loads(Path(path).read_text())
    prov = res.get("provenance", {})
    cells = res.get("cells", [])
    source = set(res.get("source_cells", []))
    comps = res.get("comparisons", [])

    print(f"run        : {Path(path).name}")
    print(f"commit     : {prov.get('commit','?')[:12]}"
          f"{'  (DIRTY TREE)' if prov.get('dirty') else ''}")
    print(f"condition  : {prov.get('condition','?')}   "
          f"target seeds {prov.get('target_seeds')}   "
          f"interpreter seeds {prov.get('interpreter_seeds')}")
    if not comps:
        print("\nno declared comparisons in this file — the gate cannot be applied")
        return 2

    key = [c for c in comps if c["metric"] == PRIMARY
           and c["method"] == METHOD and c["reference"] == CONTROL]
    transfer = [c for c in key if c["cell"] not in source]
    wins = [c for c in transfer if c["reject"] and c["delta"] > 0]
    losses = [c for c in key if c["reject"] and c["delta"] < 0]

    print(f"\nprimary comparison: {METHOD} vs {CONTROL} on {PRIMARY}")
    print(f"{'cell':28s}{'delta':>9s}{'p':>9s}{'p_adj':>9s}   verdict")
    print("-" * 68)
    for c in key:
        tag = "source" if c["cell"] in source else ""
        v = "wins" if (c["reject"] and c["delta"] > 0) else \
            "LOSES" if (c["reject"] and c["delta"] < 0) else "n.s."
        print(f"{c['cell']:28s}{c['delta']:+9.3f}{c['p']:9.3f}{c['p_adj']:9.3f}   {v} {tag}")

    print()
    if losses:
        verdict = ("FAIL — the analytic control beats the learned interpreter "
                   "where the difference is resolvable")
    elif wins:
        verdict = (f"PASS — beats the analytic control in {len(wins)} of "
                   f"{len(transfer)} transfer cells, loses nowhere")
    else:
        verdict = ("AMBIGUOUS — no resolvable difference in either direction; "
                   "the next step is more target models, not a new metric")
    print(f"GATE: {verdict}")
    print("\nper docs/PREREG.md: a FAIL keeps first_order as the lower causal layer "
          "and moves\nthe scientific question up to causal state -> logical explanation. "
          "It is not a\nreason to discard the project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results/rerun_core.json"))

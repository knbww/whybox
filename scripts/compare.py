#!/usr/bin/env python
"""Head-to-head view of one metric across methods and domains.

    python scripts/compare.py results/pilot.json executed.sufficiency@3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(path: str, metric: str = "executed.sufficiency@3") -> int:
    res = json.loads(Path(path).read_text())
    rows = res["rows"]
    domains = res.get("cells") or sorted({r["cell"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    src = set(res.get("source_cells", []))

    w = max(len(m) for m in methods) + 2
    cw = max(24, max(len(d) for d in domains) + 8)
    head = "".rjust(w) + "".join(f"{d + (' (src)' if d in src else ''):>{cw}}" for d in domains)
    print(f"\n{metric}\n{head}\n" + "-" * len(head))
    def key(m):
        vals = [r[metric] for r in rows
                if r["method"] == m and metric in r and r["cell"] not in src]
        return -np.nanmean(vals) if vals else 1e9
    for m in sorted(methods, key=key):
        line = m.rjust(w)
        for d in domains:
            r = next((r for r in rows if r["method"] == m and r["cell"] == d), None)
            if r is None or metric not in r:
                line += "".rjust(cw)
            else:
                line += f"{r[metric]:>{cw - 6}.3f}±{r.get(metric + '.ci', float('nan')):.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], *sys.argv[2:3]))

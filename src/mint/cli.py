"""Command line entry point.

    python -m mint.cli show               what is established: every gate, every table
    python -m mint.cli explain            the whole chain on one live state, in words
    python -m mint.cli verify <result>    recompute every statistic independently
    python -m mint.cli audit              tests + statistics + numbers in the docs
    python -m mint.cli run --config ...   the original transfer experiment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .experiments.transfer import ExperimentConfig, report, run


def _load(path: str | None) -> ExperimentConfig:
    if not path:
        return ExperimentConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for k in ("source_domains", "target_domains", "seeds", "spurious",
              "interpreter_seeds", "ceiling_cells", "source_cells"):
        if k in raw:
            raw[k] = tuple(raw[k])
    raw["config_path"] = path
    return ExperimentConfig(**raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mint")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the transfer experiment")
    r.add_parser = None
    r.add_argument("--config", default=None)
    r.add_argument("--refresh", action="store_true", help="retrain target models, ignore cache")
    r.add_argument("--no-executed", action="store_true", help="skip executed sufficiency checks")
    s = sub.add_parser("report", help="print the table for a finished run")
    s.add_argument("path")
    sub.add_parser("show", help="what is established: every gate, every table")
    e = sub.add_parser("explain", help="the whole chain on one live state, in words")
    e.add_argument("--retrain", action="store_true",
                   help="train the interpreter afresh instead of loading the saved one")
    e.add_argument("--variant", default="base", choices=("base", "dynamic", "transfer", "lm"),
                   help="which established result to show (default: the main hypothesis)")
    e.add_argument("--disagree", action="store_true",
                   help="pick states where B's cause differs from the world's")
    e.add_argument("--state", type=int, default=0, help="show the K-th state of the chosen list")
    v = sub.add_parser("verify", help="recompute every statistic of a result independently")
    v.add_argument("path")
    sub.add_parser("audit", help="tests, statistics, and every number in the docs")
    args = ap.parse_args(argv)

    if args.cmd in ("show", "explain", "verify", "audit"):
        from . import show
        return {"show": show.cmd_show, "audit": show.cmd_audit,
                "explain": lambda: show.cmd_explain(args.retrain, args.state, args.variant, args.disagree),
                "verify": lambda: show.cmd_verify(args.path)}[args.cmd]()
    if args.cmd == "run":
        cfg = _load(args.config)
        res = run(cfg, refresh=args.refresh, run_executed=not args.no_executed)
        print(report(res))
    else:
        print(report(json.loads(Path(args.path).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

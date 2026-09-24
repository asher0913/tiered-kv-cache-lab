"""``kvtier run | compare | plan | report``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import experiments
from .simulator import SimConfig, simulate
from .specs import MODELS, dram, hbm, nvme
from .workload import WorkloadSpec


def _tiers(args) -> list:
    tiers = []
    if args.hbm_gib > 0:
        tiers.append(hbm(args.hbm_gib))
    if args.dram_gib > 0:
        tiers.append(dram(args.dram_gib, args.pcie_gb_s))
    if args.nvme_tib > 0:
        tiers.append(nvme(args.nvme_tib))
    return tiers


def _run(args) -> int:
    workload = WorkloadSpec(sessions_per_s=args.rate, duration_s=args.duration, seed=args.seed)
    result = simulate(SimConfig(_tiers(args), workload, model=MODELS[args.model], policy=args.policy))
    print(json.dumps(result, indent=2))
    return 0


def _compare(args) -> int:
    report = experiments.tier_comparison(rates=tuple(args.rates))
    for rate, block in report.items():
        print(f"\n{rate} sessions/s — {block['workload']}")
        print("| setup | reuse | util | p50 s | p95 s | p99 s |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in block["results"]:
            print(
                f"| {r['label']} | {r['reuse_fraction']:.3f} | {r['utilization']:.2f} | "
                f"{r['ttft_p50_s']:.2f} | {r['ttft_p95_s']:.2f} | {r['ttft_p99_s']:.2f} |"
            )
    return 0


def _plan(args) -> int:
    for row in experiments.capacity_plan(args.slo, duration_s=args.duration):
        print(json.dumps(row))
    return 0


def _report(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {
        "assumptions": experiments.assumptions,
        "tier_comparison": experiments.tier_comparison,
        "eviction_policies": experiments.eviction_policies,
        "hbm_capacity_sweep": experiments.hbm_capacity_sweep,
        "pcie_sensitivity": experiments.pcie_sensitivity,
        "model_sensitivity": experiments.model_sensitivity,
        "capacity_plan": experiments.capacity_plan,
    }
    for name, run in runs.items():
        if args.only and name not in args.only:
            continue
        print(f"running {name} …", flush=True)
        (out / f"{name}.json").write_text(json.dumps(run(), indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvtier", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate one configuration")
    run.add_argument("--rate", type=float, default=1.0, help="new chat sessions per second")
    run.add_argument("--duration", type=float, default=1200.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--hbm-gib", type=float, default=16)
    run.add_argument("--dram-gib", type=float, default=128)
    run.add_argument("--pcie-gb-s", type=float, default=25)
    run.add_argument("--nvme-tib", type=float, default=0)
    run.add_argument("--model", choices=sorted(MODELS), default="llama-3-8b")
    run.add_argument("--policy", choices=("lru", "leaf-first"), default="leaf-first")
    run.set_defaults(func=_run)

    compare = sub.add_parser("compare", help="no cache vs HBM vs HBM+DRAM vs three tiers")
    compare.add_argument("--rates", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    compare.set_defaults(func=_compare)

    plan = sub.add_parser("plan", help="max sustainable load under a p95 TTFT SLO")
    plan.add_argument("--slo", type=float, default=2.0, help="p95 TTFT target in seconds")
    plan.add_argument("--duration", type=float, default=900.0)
    plan.set_defaults(func=_plan)

    report = sub.add_parser("report", help="run every experiment and write JSON results")
    report.add_argument("--out", default="results")
    report.add_argument("--only", nargs="*")
    report.set_defaults(func=_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

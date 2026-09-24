"""The experiments reported in the README. Each returns plain JSON-able data."""

from __future__ import annotations

from dataclasses import replace

from .simulator import SimConfig, max_sustainable_rate, simulate
from .specs import MODELS, ComputeSpec, dram, hbm, nvme
from .workload import WorkloadSpec, generate, summarize

TIER_SETUPS = {
    "no cache": lambda: [],
    "HBM 16 GiB": lambda: [hbm()],
    "HBM + DRAM 128 GiB": lambda: [hbm(), dram()],
    "HBM + DRAM + NVMe 1 TiB": lambda: [hbm(), dram(), nvme()],
}


def _row(result: dict) -> dict:
    keep = (
        "label", "policy", "reuse_fraction", "reuse_by_tier", "utilization",
        "ttft_p50_s", "ttft_p95_s", "ttft_p99_s", "prefill_tokens_computed",
    )  # fmt: skip
    row = {k: result[k] for k in keep}
    if "cache" in result:
        row["orphaned_blocks"] = result["cache"]["orphaned_blocks"]
    return row


def tier_comparison(rates=(0.5, 1.0, 2.0), duration_s: float = 1200.0, policy: str = "leaf-first") -> dict:
    out = {}
    for rate in rates:
        workload = WorkloadSpec(sessions_per_s=rate, duration_s=duration_s)
        requests = generate(workload)
        out[str(rate)] = {
            "workload": summarize(requests, workload.block_tokens),
            "results": [
                _row(simulate(SimConfig(build(), workload, policy=policy, label=name), requests))
                for name, build in TIER_SETUPS.items()
            ],
        }
    return out


def eviction_policies(rate: float = 1.0, capacities_gib=(4, 8, 16, 32)) -> list[dict]:
    """LRU versus leaf-first on an HBM-only cache, where evictions are final."""
    workload = WorkloadSpec(sessions_per_s=rate)
    requests = generate(workload)
    rows = []
    for gib in capacities_gib:
        for policy in ("lru", "leaf-first"):
            result = simulate(SimConfig([hbm(gib)], workload, policy=policy), requests)
            rows.append({"hbm_gib": gib, **_row(result)})
    return rows


def hbm_capacity_sweep(rate: float = 1.0, capacities_gib=(2, 4, 8, 16, 32, 64, 128)) -> list[dict]:
    workload = WorkloadSpec(sessions_per_s=rate)
    requests = generate(workload)
    rows = []
    for gib in capacities_gib:
        result = simulate(SimConfig([hbm(gib)], workload, policy="leaf-first"), requests)
        rows.append({"hbm_gib": gib, **_row(result)})
    return rows


def pcie_sensitivity(rate: float = 2.0, bandwidths_gb_s=(8, 16, 25, 50)) -> list[dict]:
    """How much the DRAM tier's value depends on host-to-device bandwidth."""
    workload = WorkloadSpec(sessions_per_s=rate)
    requests = generate(workload)
    rows = []
    for bw in bandwidths_gb_s:
        result = simulate(SimConfig([hbm(), dram(gb_per_s=bw)], workload, policy="leaf-first"), requests)
        rows.append({"dram_gb_per_s": bw, **_row(result), "dram_transfer_s": result["transfer_seconds"]["dram"]})
    return rows


def model_sensitivity(rate: float = 1.0) -> list[dict]:
    """Same HBM budget, different KV footprint per token (compute held fixed to isolate memory)."""
    workload = WorkloadSpec(sessions_per_s=rate)
    requests = generate(workload)
    rows = []
    for name, model in MODELS.items():
        for label, build in (("HBM 16 GiB", lambda: [hbm()]), ("HBM + DRAM 128 GiB", lambda: [hbm(), dram()])):
            result = simulate(SimConfig(build(), workload, model=model, policy="leaf-first", label=label), requests)
            rows.append({"model": name, "kv_kib_per_token": model.kv_bytes_per_token // 1024, **_row(result)})
    return rows


def capacity_plan(slo_p95_ttft_s: float = 2.0, duration_s: float = 900.0) -> list[dict]:
    """Highest session arrival rate that meets the p95 TTFT SLO, per tier setup."""
    rows = []
    for name, build in TIER_SETUPS.items():

        def make(rate, build=build, name=name):
            workload = WorkloadSpec(sessions_per_s=rate, duration_s=duration_s)
            return SimConfig(build(), workload, policy="leaf-first", label=name)

        rate, result = max_sustainable_rate(make, slo_p95_ttft_s)
        rows.append(
            {
                "setup": name,
                "max_sessions_per_s": round(rate, 3),
                "requests_per_s": round(result["requests"] / duration_s, 3) if result else 0.0,
                "ttft_p95_s_at_max": round(result.get("ttft_p95_s", 0.0), 3),
                "reuse_fraction_at_max": round(result.get("reuse_fraction", 0.0), 3),
                "utilization_at_max": round(result.get("utilization", 0.0), 3),
            }
        )
    return rows


def assumptions() -> dict:
    return {
        "model": "llama-3-8b unless stated (32 layers, 8 KV heads, head_dim 128, bf16 → 128 KiB/token)",
        "block_tokens": ComputeSpec().block_tokens,
        "prefill_tokens_per_s": ComputeSpec().prefill_tokens_per_s,
        "tiers": {
            "hbm": {"capacity_gib": 16, "load_cost": "none (already resident)"},
            "dram": {"capacity_gib": 128, "gb_per_s": 25, "latency_us": 10},
            "nvme": {"capacity_tib": 1, "gb_per_s": 6, "latency_us": 100},
        },
        "workload": replace(WorkloadSpec(), sessions_per_s=0).describe(),
    }

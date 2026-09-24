"""Discrete-event simulation of a prefill instance with a tiered prefix cache.

The serving model is a disaggregated prefill instance: requests are prefilled
one at a time in arrival order, and decode runs elsewhere. For each request:

1. find the longest cached prefix (blocks may sit in HBM, DRAM or NVMe);
2. copy the lower-tier blocks into HBM (bandwidth + latency of that tier);
3. prefill the remaining tokens at the engine's prefill throughput;
4. make the prompt and, once decoded, the reply resident in HBM.

Time to first token is queueing delay plus steps 2 and 3. Transfers are not
overlapped with compute, a conservative choice.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from .cache import TieredKVCache, chain_keys
from .specs import MODELS, ComputeSpec, ModelSpec, TierSpec
from .workload import Request, WorkloadSpec, generate


@dataclass
class SimConfig:
    tiers: Sequence[TierSpec]
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    model: ModelSpec = MODELS["llama-3-8b"]
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    policy: str = "lru"
    label: str = ""


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


def simulate(config: SimConfig, requests: list[Request] | None = None) -> dict:
    block_tokens = config.compute.block_tokens
    if config.workload.block_tokens != block_tokens:
        raise ValueError("workload and compute block sizes differ")
    block_bytes = config.model.kv_bytes_per_token * block_tokens
    requests = requests if requests is not None else generate(config.workload)
    cache = TieredKVCache(config.tiers, block_bytes, config.policy) if config.tiers else None

    free_at = 0.0
    busy = 0.0
    ttft, waits = [], []
    prompt_blocks = computed_blocks = 0
    reused = {t.name: 0 for t in config.tiers}
    transfer_s = {t.name: 0.0 for t in config.tiers}

    for request in requests:
        keys = chain_keys(request.prompt, block_tokens)
        prompt_blocks += len(keys)
        load = 0.0
        matched: list[int] = []
        if cache is not None:
            matched = cache.access(keys)
            per_tier: dict[int, int] = {}
            for where in matched:
                per_tier[where] = per_tier.get(where, 0) + 1
            for where, count in per_tier.items():
                spec = config.tiers[where]
                reused[spec.name] += count
                seconds = spec.transfer_seconds(count * block_bytes) if where > 0 else 0.0
                transfer_s[spec.name] += seconds
                load += seconds
        compute_blocks = len(keys) - len(matched)
        computed_blocks += compute_blocks
        service = load + compute_blocks * block_tokens / config.compute.prefill_tokens_per_s
        start = max(request.arrival, free_at)
        free_at = start + service
        busy += service
        waits.append(start - request.arrival)
        ttft.append(free_at - request.arrival)
        if cache is not None:
            reply = chain_keys([request.output], block_tokens, parent=keys[-1])
            cache.access(keys + reply, record=False)  # the decoded reply joins the cached history

    makespan = max(free_at, requests[-1].arrival) - requests[0].arrival if requests else 0.0
    total_tokens = prompt_blocks * block_tokens
    result = {
        "label": config.label or "+".join(t.name for t in config.tiers) or "no cache",
        "policy": config.policy,
        "requests": len(requests),
        "prompt_tokens": total_tokens,
        "reuse_fraction": 1 - computed_blocks / prompt_blocks if prompt_blocks else 0.0,
        "reuse_by_tier": {name: n / prompt_blocks for name, n in reused.items()} if prompt_blocks else {},
        "prefill_tokens_computed": computed_blocks * block_tokens,
        "transfer_seconds": {k: round(v, 3) for k, v in transfer_s.items()},
        "utilization": busy / makespan if makespan else 0.0,
        "ttft_mean_s": statistics.fmean(ttft) if ttft else 0.0,
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p95_s": percentile(ttft, 95),
        "ttft_p99_s": percentile(ttft, 99),
        "max_queue_wait_s": max(waits) if waits else 0.0,
    }
    if cache is not None:
        result["cache"] = {
            "resident_blocks": cache.resident_blocks(),
            "demotions": cache.stats.demotions,
            "promotions": cache.stats.promotions,
            "dropped_blocks": cache.stats.dropped_blocks,
            "orphaned_blocks": cache.orphaned_blocks(),
        }
    return result


def max_sustainable_rate(
    make_config,
    slo_p95_ttft_s: float,
    *,
    low: float = 0.05,
    high: float = 8.0,
    iterations: int = 9,
) -> tuple[float, dict]:
    """Largest session arrival rate whose p95 TTFT meets the SLO (bisection).

    ``make_config(rate)`` must build a ``SimConfig`` for that arrival rate.
    Assumes p95 TTFT is non-decreasing in load, which holds for this FIFO model
    up to sampling noise from the seeded workload.
    """
    best_rate, best_result = 0.0, {}
    if simulate(make_config(low))["ttft_p95_s"] > slo_p95_ttft_s:
        return 0.0, {}
    for _ in range(iterations):
        mid = (low + high) / 2
        result = simulate(make_config(mid))
        if result["ttft_p95_s"] <= slo_p95_ttft_s:
            best_rate, best_result, low = mid, result, mid
        else:
            high = mid
    return best_rate, best_result

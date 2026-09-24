import pytest

from kvtier.cli import main
from kvtier.simulator import SimConfig, max_sustainable_rate, simulate
from kvtier.specs import ComputeSpec, dram, hbm
from kvtier.workload import Request, WorkloadSpec, generate

SMALL = WorkloadSpec(sessions_per_s=0.5, duration_s=300, seed=3)


def test_workload_is_deterministic_and_turns_extend_history():
    a, b = generate(SMALL), generate(SMALL)
    assert a == b
    assert all(x.arrival <= y.arrival for x, y in zip(a, a[1:], strict=False))
    by_session: dict[int, list[Request]] = {}
    for r in a:
        by_session.setdefault(r.session, []).append(r)
    for turns in by_session.values():
        turns.sort(key=lambda r: r.turn)
        for prev, nxt in zip(turns, turns[1:], strict=False):
            assert nxt.prompt[: len(prev.prompt) + 1] == (*prev.prompt, prev.output)


def test_no_cache_recomputes_everything():
    result = simulate(SimConfig([], SMALL))
    assert result["reuse_fraction"] == 0
    assert result["prefill_tokens_computed"] == result["prompt_tokens"]


def test_more_hbm_never_reduces_reuse():
    requests = generate(SMALL)
    reuse = [simulate(SimConfig([hbm(g)], SMALL), requests)["reuse_fraction"] for g in (1, 2, 4, 8)]
    assert reuse == sorted(reuse)
    assert reuse[-1] > reuse[0]


def test_adding_dram_never_reduces_reuse():
    requests = generate(SMALL)
    alone = simulate(SimConfig([hbm(2)], SMALL), requests)
    tiered = simulate(SimConfig([hbm(2), dram(64)], SMALL), requests)
    assert tiered["reuse_fraction"] >= alone["reuse_fraction"]
    # A block in HBM can only count as a hit if its whole prefix is cached; the DRAM tier
    # keeps more parents alive, so HBM hits can only go up.
    assert tiered["reuse_by_tier"]["hbm"] >= alone["reuse_by_tier"]["hbm"]


def test_ttft_without_contention_is_pure_prefill_time():
    workload = WorkloadSpec(block_tokens=16)
    request = Request(arrival=10.0, session=0, turn=0, prompt=((1, 625),), output=(2, 1))
    result = simulate(SimConfig([], workload, compute=ComputeSpec(prefill_tokens_per_s=10_000)), [request])
    assert result["ttft_p50_s"] == pytest.approx(625 * 16 / 10_000)


def test_simultaneous_requests_queue():
    workload = WorkloadSpec(block_tokens=16)
    requests = [Request(0.0, i, 0, ((i + 10, 100),), (i + 100, 1)) for i in range(2)]
    result = simulate(SimConfig([], workload, compute=ComputeSpec(prefill_tokens_per_s=1600)), requests)
    assert result["ttft_p50_s"] in (pytest.approx(1.0), pytest.approx(2.0))
    assert result["max_queue_wait_s"] == pytest.approx(1.0)


def test_block_size_mismatch_is_rejected():
    with pytest.raises(ValueError):
        simulate(SimConfig([], WorkloadSpec(block_tokens=32)))


def test_capacity_planner_orders_setups():
    def make(tiers):
        return lambda rate: SimConfig(tiers(), WorkloadSpec(sessions_per_s=rate, duration_s=240, seed=1))

    none_rate, _ = max_sustainable_rate(make(lambda: []), 2.0, iterations=5)
    tiered_rate, result = max_sustainable_rate(make(lambda: [hbm(), dram()]), 2.0, iterations=5)
    assert tiered_rate > none_rate > 0
    assert result["ttft_p95_s"] <= 2.0


def test_cli_run(capsys):
    assert main(["run", "--rate", "0.3", "--duration", "120", "--nvme-tib", "0.5"]) == 0
    assert '"reuse_fraction"' in capsys.readouterr().out

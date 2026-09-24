"""Model and hardware descriptions that turn tokens into bytes and bytes into seconds.

All figures here are *assumptions* for a simulator, chosen to be in the right
order of magnitude for a single modern datacenter GPU. They are parameters,
not measurements, and every experiment records the values it used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

GiB = 1024**3
TiB = 1024**4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16 / bf16

    @property
    def kv_bytes_per_token(self) -> int:
        """Keys and values for every layer: 2 × layers × kv_heads × head_dim × dtype."""
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes


# Grouped-query attention configurations as published in the model cards.
MODELS = {
    "llama-3-8b": ModelSpec("llama-3-8b", layers=32, kv_heads=8, head_dim=128),
    "llama-3-70b": ModelSpec("llama-3-70b", layers=80, kv_heads=8, head_dim=128),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", layers=28, kv_heads=4, head_dim=128),
}


@dataclass(frozen=True)
class TierSpec:
    """One storage tier. ``bandwidth`` is the effective rate into GPU memory."""

    name: str
    capacity_bytes: int
    bandwidth_bytes_per_s: float
    latency_s: float = 0.0

    def transfer_seconds(self, nbytes: int) -> float:
        if nbytes <= 0:
            return 0.0
        return self.latency_s + nbytes / self.bandwidth_bytes_per_s


def hbm(capacity_gib: float = 16) -> TierSpec:
    # Already resident: reading cached KV from HBM costs nothing extra on the prefill path.
    return TierSpec("hbm", int(capacity_gib * GiB), float("inf"), 0.0)


def dram(capacity_gib: float = 128, gb_per_s: float = 25) -> TierSpec:
    # Pinned host memory over PCIe 4.0 x16 (~25 GB/s effective).
    return TierSpec("dram", int(capacity_gib * GiB), gb_per_s * 1e9, 10e-6)


def nvme(capacity_tib: float = 1, gb_per_s: float = 6) -> TierSpec:
    # A single PCIe 4.0 NVMe drive, sequential reads.
    return TierSpec("nvme", int(capacity_tib * TiB), gb_per_s * 1e9, 100e-6)


@dataclass(frozen=True)
class ComputeSpec:
    """Prefill throughput of the serving engine (tokens per second)."""

    prefill_tokens_per_s: float = 10_000.0
    block_tokens: int = 16  # KV block size, as in paged attention


def describe(*items) -> list[dict]:
    return [asdict(item) for item in items]

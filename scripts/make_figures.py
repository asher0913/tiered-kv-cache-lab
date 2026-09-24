"""Redraw docs/*.png from results/*.json (run `kvtier report` first)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def capacity_plan() -> None:
    rows = load("capacity_plan")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = [r["setup"] for r in rows]
    values = [r["requests_per_s"] for r in rows]
    bars = ax.barh(labels, values, color=["#9e9e9e", "#6baed6", "#3182bd", "#08519c"])
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{value:.1f} req/s", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("max request rate meeting p95 TTFT ≤ 2 s")
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title("Sustainable load on one prefill instance (Llama-3-8B)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "capacity_plan.png", dpi=150)
    plt.close(fig)


def hbm_sweep() -> None:
    rows = load("hbm_capacity_sweep")
    gib = [r["hbm_gib"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.5))
    left.plot(gib, [100 * r["reuse_fraction"] for r in rows], marker="o")
    left.set(
        xscale="log",
        xlabel="HBM prefix-cache capacity (GiB)",
        ylabel="prompt tokens reused %",
        title="Reuse vs HBM capacity (1 session/s)",
    )
    right.plot(gib, [r["ttft_p95_s"] for r in rows], marker="o", color="#d62728")
    right.axhline(2.0, color="grey", ls="--", lw=1, label="2 s SLO")
    right.set(
        xscale="log",
        yscale="log",
        xlabel="HBM prefix-cache capacity (GiB)",
        ylabel="p95 TTFT (s)",
        title="Tail latency vs HBM capacity",
    )
    right.legend(frameon=False)
    for ax in (left, right):
        ax.set_xticks(gib, [str(g) for g in gib])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "hbm_capacity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    capacity_plan()
    hbm_sweep()

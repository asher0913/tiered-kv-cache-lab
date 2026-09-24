"""Seeded multi-turn chat workload with shared system prompts.

Why this shape: in chat serving most prompt tokens are *repeated*. Every turn
resends the whole conversation so far, and many conversations start from the
same few system prompts. That repetition is what a prefix cache monetises,
and conversations that pause for a minute between turns are what push
blocks out of GPU memory and into lower tiers.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

Segment = tuple[int, int]  # (segment_id, n_blocks)


@dataclass(frozen=True)
class WorkloadSpec:
    sessions_per_s: float = 1.0
    duration_s: float = 1200.0
    system_prompts: int = 24
    zipf_s: float = 1.1
    system_tokens: tuple[int, int] = (512, 4096)
    mean_turns: float = 4.0
    max_turns: int = 12
    user_tokens: tuple[int, int] = (32, 512)
    output_tokens: tuple[int, int] = (64, 768)
    think_time_mean_s: float = 45.0
    decode_s_per_token: float = 0.02
    block_tokens: int = 16
    seed: int = 0

    def describe(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Request:
    arrival: float
    session: int
    turn: int
    prompt: tuple[Segment, ...]  # full prompt: system + history + new user message
    output: Segment  # the assistant reply; its KV joins the history for the next turn

    @property
    def prompt_blocks(self) -> int:
        return sum(n for _, n in self.prompt)


def _blocks(tokens: int, block_tokens: int) -> int:
    return max(1, math.ceil(tokens / block_tokens))


def generate(spec: WorkloadSpec) -> list[Request]:
    """Poisson session arrivals; geometric turn counts; Zipf-popular system prompts."""
    rng = random.Random(spec.seed)
    bt = spec.block_tokens
    weights = [1 / (rank + 1) ** spec.zipf_s for rank in range(spec.system_prompts)]
    system = [(-(i + 1), _blocks(rng.randint(*spec.system_tokens), bt)) for i in range(spec.system_prompts)]
    next_segment = 1
    requests: list[Request] = []
    t, session = 0.0, 0
    while True:
        t += rng.expovariate(spec.sessions_per_s)
        if t >= spec.duration_s:
            break
        turns = 1
        while turns < spec.max_turns and rng.random() > 1 / spec.mean_turns:
            turns += 1
        history: list[Segment] = [rng.choices(system, weights)[0]]
        arrival = t
        for turn in range(turns):
            user = (next_segment, _blocks(rng.randint(*spec.user_tokens), bt))
            output_tokens = rng.randint(*spec.output_tokens)
            reply = (next_segment + 1, _blocks(output_tokens, bt))
            next_segment += 2
            prompt = (*history, user)
            requests.append(Request(arrival, session, turn, prompt, reply))
            history = [*prompt, reply]
            arrival += output_tokens * spec.decode_s_per_token + rng.expovariate(1 / spec.think_time_mean_s)
        session += 1
    requests.sort(key=lambda r: (r.arrival, r.session))
    return [r for r in requests if r.arrival < spec.duration_s]


def summarize(requests: list[Request], block_tokens: int) -> dict:
    prompt_tokens = sum(r.prompt_blocks for r in requests) * block_tokens
    sessions = len({r.session for r in requests})
    return {
        "requests": len(requests),
        "sessions": sessions,
        "mean_prompt_tokens": round(prompt_tokens / max(len(requests), 1)),
        "total_prompt_tokens": prompt_tokens,
    }

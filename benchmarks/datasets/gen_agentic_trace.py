"""Generate synthetic agentic trace dataset (M3.7).

Each session has a shared system prompt (same tokens across all turns) followed
by 100 turns of user messages. This simulates an agentic loop where the
system prompt is long and re-used, exercising KV-cache prefix reuse.

Usage:
    python benchmarks/datasets/gen_agentic_trace.py \
        --sessions 500 --turns 100 --system-prompt-tokens 2048
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import tiktoken


def _make_token_ids(enc: tiktoken.Encoding, n: int, seed: int) -> list[int]:
    """Generate n pseudo-random but reproducible token IDs within the safe range."""
    rng = random.Random(seed)
    # cl100k_base has 100,277 tokens but special tokens (≥100,257) can't be decoded.
    # Use only the regular token range to avoid KeyError on decode.
    safe_max = min(enc.n_vocab, 100_256)
    return [rng.randint(0, safe_max - 1) for _ in range(n)]


def generate_trace(
    sessions: int = 500,
    turns: int = 100,
    system_prompt_tokens: int = 2048,
    output_path: str | Path = "",
    seed: int = 42,
) -> Path:
    """Generate agentic trace JSONL and return the output path."""
    enc = tiktoken.get_encoding("cl100k_base")

    # One fixed system-prompt token sequence shared across ALL sessions
    rng = random.Random(seed)
    # Stay within decodable token range (special tokens ≥ 100,257 can't be decoded)
    safe_max = min(enc.n_vocab, 100_256)
    system_token_ids: list[int] = [rng.randint(0, safe_max - 1) for _ in range(system_prompt_tokens)]
    system_text = enc.decode(system_token_ids)

    if not output_path:
        output_path = Path(__file__).parent / "agentic_trace.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for session_idx in range(sessions):
            for turn_idx in range(turns):
                # User message: short, unique per turn
                user_token_ids = _make_token_ids(
                    enc,
                    n=rng.randint(16, 64),
                    seed=seed + session_idx * 10000 + turn_idx,
                )
                user_text = enc.decode(user_token_ids)

                record = {
                    "session_id": f"session-{session_idx:04d}",
                    "turn_id": turn_idx,
                    "system_token_ids": system_token_ids,
                    "user_token_ids": user_token_ids,
                    "system_text": system_text,
                    "user_text": user_text,
                    # Combined token sequence: system + user (what the model sees)
                    "token_ids": system_token_ids + user_token_ids,
                }
                fout.write(json.dumps(record) + "\n")
                total_records += 1

    print(f"Generated {total_records} records ({sessions} sessions × {turns} turns)")
    print(f"System prompt: {system_prompt_tokens} tokens (shared across all sessions)")
    print(f"Output: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate agentic trace dataset")
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--system-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "agentic_trace.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_trace(
        sessions=args.sessions,
        turns=args.turns,
        system_prompt_tokens=args.system_prompt_tokens,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert sammshen/lmcache-agentic-traces HuggingFace dataset to mooncake_trace JSONL.

Each row in the HF dataset has cumulative OpenAI-format messages. This script
converts them to mooncake_trace entries grouped by session_id.

Usage:
    python convert_lmcache_to_mooncake.py \
        --output /workspace/results/lmcache_traces.jsonl

    # Or with a local copy of the dataset:
    python convert_lmcache_to_mooncake.py \
        --input /path/to/train.jsonl \
        --output /workspace/results/lmcache_traces.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson


def download_dataset(cache_dir: Path):
    """Download the dataset from HuggingFace and return the dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: 'datasets' package required for HF download. "
            "Install with: uv add datasets\n"
            "Or download manually and use --input.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Downloading sammshen/lmcache-agentic-traces from HuggingFace...")
    ds = load_dataset(
        "sammshen/lmcache-agentic-traces", split="train", cache_dir=str(cache_dir)
    )
    return ds


def convert_row(row: dict) -> dict:
    """Convert one HF dataset row to mooncake_trace format."""
    result = {
        "session_id": row["session_id"],
        "messages": row["input"],
        "output_length": row["output_length"],
    }
    # Include pre_gap as delay (ms) for trace replay timing.
    # This is the real tool-execution / user-thinking time between
    # the previous response completing and this request being sent.
    pre_gap = row.get("pre_gap", 0)
    if pre_gap and pre_gap > 0:
        result["delay"] = int(pre_gap * 1000)  # seconds -> ms
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert lmcache-agentic-traces to mooncake_trace JSONL"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to local train.jsonl (downloads from HF if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
        help="HF cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        print(f"Loading from {args.input}...")
        rows = []
        with open(args.input, "rb") as f:
            for line in f:
                if line.strip():
                    rows.append(orjson.loads(line))
    else:
        rows = download_dataset(args.cache_dir)

    count = 0
    sessions = set()
    with open(args.output, "wb") as out:
        for row in rows:
            entry = convert_row(row)
            out.write(orjson.dumps(entry))
            out.write(b"\n")
            sessions.add(entry["session_id"])
            count += 1

    print(f"Wrote {count} entries ({len(sessions)} sessions) to {args.output}")


if __name__ == "__main__":
    main()

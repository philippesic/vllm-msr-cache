"""Small synthetic agent-session load contract, independent of model/runtime.

Tool durations and abandonment belong only to the load generator. The policy
gets completed prompt prefixes and a fixed estimate, never the next arrival.
Each agent is closed-loop (model completion, then tool wait, then next turn).
Background arrivals are open-loop. Arm latency can therefore affect subsequent
agent arrival times; compare the same users/process, not identical wall clocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SCHEMA = "synthetic-agent-load-trial-v1"
TASKS = (
    ("api", "Review a service API change and identify missing validation."),
    ("tests", "Diagnose a flaky test from test reports and propose a small fix."),
    ("search", "Find relevant project notes and reconcile conflicting details."),
    ("logs", "Investigate a service incident using successive log excerpts."),
    ("config", "Review configuration changes for a deployment."),
    ("docs", "Check documentation against source excerpts and summarize gaps."),
)


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_trial(seed: int = 4109) -> dict:
    rng = random.Random(seed)
    sessions = []
    for index, (family, task) in enumerate(TASKS):
        # Session identity starts the document to avoid an artificial shared
        # system-prefix cache hit dominating all six independent contexts.
        records = []
        for record in range(90):
            records.append(
                f"{family} record {record}: component worker_{index}_{record % 9} "
                f"accepts task IDs, validates revision {record % 7}, and records "
                f"status before releasing resources. Observed count "
                f"{rng.randrange(10, 900)}; verify against the next tool report."
            )
        turns = []
        for turn in range(6):
            turns.append({
                "turn": turn,
                "user_instruction": (
                    f"Investigation step {turn + 1}: {task} "
                    "Use only the supplied records. Give a concise finding "
                    "and one next action."
                ),
                "tool_result_for_this_turn": (
                    f"Tool report for {family}, step {turn + 1}: "
                    f"checked worker_{index}_{turn % 9}; "
                    f"{rng.randrange(2, 30)} checks passed, "
                    f"{rng.randrange(0, 3)} findings need review. "
                    "No file changes were made."
                ),
            })
        sessions.append({
            "session_id": f"agent-{index}",
            "task_family": family,
            "initial_arrival_ms": index * 180,
            "context_text": f"Session agent-{index}.\n" + "\n".join(records),
            "target_initial_context_tokens": 2048 + (index % 3) * 256,
            "max_output_tokens": 64,
            "turns": turns,
            "load_generator_private": {
                "tool_wait_ms_after_turn": [
                    rng.choice((75, 180, 500, 1200, 2400)) for _ in turns
                ],
                # After the third completion the orchestrator still believes
                # a tool is pending; its eventual cancellation is not hinted.
                "abandon_after_turn": 2 if index == 5 else None,
            },
        })
    result = {
        "schema": SCHEMA,
        "seed": seed,
        "provenance": "generated synthetic workload; not a production trace",
        "schedule": "closed_loop_agents_plus_open_loop_background",
        "policy_contract": {
            "declared_wait_estimate_ms": 250,
            "tool_pending_hint_refresh_ms": 250,
            "refresh_limit_ms": 2000,
            "future_tool_durations_visible": False,
            "future_abandonment_visible": False,
            "actual_next_prompt_visible": False,
        },
        "sessions": sessions,
        "background": [
            {
                "request_id": f"background-{i}",
                "arrival_ms": 1200 + 800 * i,
                "target_prompt_tokens": 768,
                "max_output_tokens": 96,
                "prompt": (
                    f"Independent request {i}. Explain how to review service "
                    "logs, validate a hypothesis, and report uncertainty. "
                    "Give concrete examples."
                ),
            }
            for i in range(12)
        ],
    }
    result["trace_digest"] = canonical_digest(result)
    return result


def policy_observation(trial: dict, session_id: str, completed_turn: int) -> dict:
    """Allowlisted past-only view; prompt token IDs are supplied by the driver.

    This is not the connector's HTTP hint: it states what its producer may
    inspect. The producer must hash only the prefix already processed by the
    model, and use live monotonic clocks for publication/availability.
    """
    session = next(s for s in trial["sessions"] if s["session_id"] == session_id)
    if not 0 <= completed_turn < len(session["turns"]):
        raise ValueError("completed_turn outside the recorded session")
    return {
        "session_id": session_id,
        "completed_turn": completed_turn,
        "past_context": session["context_text"],
        "completed_turn_inputs": session["turns"][: completed_turn + 1],
        "declared_wait_estimate_ms": trial["policy_contract"][
            "declared_wait_estimate_ms"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4109)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trial = build_trial(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trial, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sessions": len(trial["sessions"]),
                      "background_requests": len(trial["background"])}))


if __name__ == "__main__":
    main()

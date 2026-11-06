"""CPU-only audit of native observations; no policy-performance inference.

Usage: python -m benchmarks.research_checks.hybrid_qualification TRACE.jsonl
The candidate overlap is optimistic: CPU readiness and bandwidth are unobserved.
An async load's issue-to-receipt time includes engine polling and scheduling; it
is not PCIe copy time or necessarily exposed critical-path delay.
"""

import argparse
import json
import math
from pathlib import Path


def analyze(events, block_tokens=16):
    if type(block_tokens) is not int or block_tokens <= 0:
        raise ValueError("block_tokens must be a positive integer")
    if not events or events[0].get("schema") != "kv-qualification-v1":
        raise ValueError("Missing supported trace header")
    previous = -1
    arrivals, pending, episodes, steps, admissions = {}, {}, [], [], {}
    errors = []
    for e in events:
        ns = e.get("ns")
        if type(ns) is not int or ns < previous:
            raise ValueError("Trace timestamps must be ordered integer nanoseconds")
        previous = ns
        event, rid = e["event"], e.get("request_id")
        if event == "arrival":
            if rid in arrivals:
                errors.append(f"duplicate arrival: {rid}")
            arrivals[rid] = ns
        elif event == "load_issued":
            if rid in pending:
                errors.append(f"overlapping loads: {rid}")
            if rid not in arrivals:
                errors.append(f"load without arrival: {rid}")
            pending[rid] = e
        elif event == "load_ready":
            issue = pending.pop(rid, None)
            if issue is None:
                errors.append(f"receipt without issue: {rid}")
                continue
            if type(issue.get("external_tokens")) is not int or issue["external_tokens"] <= 0:
                raise ValueError("A load must contain positive external tokens")
            needed = math.ceil(issue["external_tokens"] / block_tokens)
            # A later reload cannot use an earlier pre-admission window.
            previous_admissions = [t for t in admissions.get(rid, []) if t < issue["ns"]]
            window_start = max([arrivals.get(rid, issue["ns"])] + previous_admissions)
            candidates = [
                s for s in steps
                if window_start <= s["ns"] < issue["ns"]
                and any(r == rid and status == "WAITING" for r, status in s["waiting"])
                and s["scheduled_tokens"] > 0 and s["reclaimable"] >= needed
            ]
            lookahead = (issue["ns"] - candidates[0]["ns"]) if candidates else 0
            residence = ns - issue["ns"]
            episodes.append(dict(
                request_id=rid, external_tokens=issue["external_tokens"],
                issue_ns=issue["ns"], ready_ns=ns,
                async_residence_ms=residence / 1e6,
                arrival_to_issue_ms=(issue["ns"] - arrivals.get(rid, issue["ns"])) / 1e6,
                optimistic_lookahead_ms=lookahead / 1e6,
                optimistic_direct_saving_ms=min(residence, lookahead) / 1e6,
                candidate_samples=len(candidates),
            ))
        elif event == "admitted":
            admissions.setdefault(rid, []).append(ns)
        elif event == "step":
            steps.append(e)
    for episode in episodes:
        later = [t for t in admissions.get(episode["request_id"], []) if t >= episode["ready_ns"]]
        episode["ready_to_admission_ms"] = (min(later) - episode["ready_ns"]) / 1e6 if later else None
    return dict(
        schema="hybrid-opportunity-audit-v1", arrivals=len(arrivals),
        completed_loads=len(episodes), incomplete_loads=sorted(pending),
        errors=errors, truncated_steps=sum(bool(s["waiting_truncated"]) for s in steps),
        step_count=len(steps),
        total_async_residence_ms=sum(x["async_residence_ms"] for x in episodes),
        optimistic_direct_saving_ms=sum(x["optimistic_direct_saving_ms"] for x in episodes),
        loads_with_candidate_overlap=sum(x["candidate_samples"] > 0 for x in episodes),
        episodes=episodes,
        qualification="inconclusive",
        limitations=[
            "No CPU-ready-prefix or transfer-bandwidth observation; candidate overlap is optimistic.",
            "Reclaimable blocks may contain valuable idle histories; victim costs are not zero.",
            "Async residence includes scheduler/polling delay; exposed stall needs a causal intervention.",
            "Summed per-request savings are not a throughput or makespan bound.",
            "This trace cannot qualify MSR predictive information or exact-cache correctness.",
        ],
    )


def audit_conversation_source(data):
    """Check label observability before fitting; never invent abandonment labels."""
    messages = [m for r in data["rows"] for m in r["conversation"]]
    users = [m for m in messages if m["role"] == "user"]
    assistants = [m for m in messages if m["role"] == "assistant"]
    return dict(
        schema="hybrid-source-observability-v1", sessions=len(data["rows"]),
        user_messages=len(users), user_timestamp_count=sum(bool(m.get("timestamp")) for m in users),
        assistant_messages=len(assistants), assistant_timestamp_count=sum(bool(m.get("timestamp")) for m in assistants),
        observed_session_end_count=sum(bool(r.get("observed_session_end")) for r in data["rows"]),
        reusable_timing_labels=False,
        reason="Timestamp presence alone does not establish request arrivals or observation horizons. Verify source semantics; do not label final records as abandonment.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--block-tokens", type=int, default=16)
    args = p.parse_args()
    events = [json.loads(line) for line in args.trace.read_text().splitlines()]
    print(json.dumps(analyze(events, args.block_tokens), indent=2))


if __name__ == "__main__":
    main()

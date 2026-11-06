"""Bounded observability audits, never a source of fabricated reuse labels.

Parquet is optional and imported only by the CLI. Read nested traces one row
group at a time: whole-table conversion fails on some published shards.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Naive timestamp has no established clock")
    return parsed


def messages(value):
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not parsed:
        return None
    if not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in parsed):
        return None
    # finish_reason describes generation, not the cached message text.
    return [{k: v for k, v in m.items() if k != "finish_reason"} for m in parsed]


def audit_span_rows(rows):
    counts = Counter()
    cohorts = Counter()
    gaps = []
    for row in rows:
        counts["sessions"] += 1
        cohorts[str((row.get("benchmark"), row.get("harness")))] += 1
        spans = row.get("spans") or []
        counts["spans"] += len(spans)
        counts["sessions_without_observation_end"] += not bool(row.get("observed_session_end"))
        streams = {}
        for span in spans:
            attrs = span.get("attributes") or {}
            if attrs.get("gen_ai.operation.name") != "chat":
                counts["nonchat_spans"] += 1
                continue
            try:
                start, end = timestamp(span["start_time"]), timestamp(span["end_time"])
                if end < start:
                    raise ValueError("Negative call duration")
                inp = messages(attrs.get("gen_ai.input.messages"))
                output = messages(attrs.get("gen_ai.output.messages"))
            except (KeyError, TypeError, ValueError):
                counts["invalid_spans"] += 1
                continue
            counts["missing_input"] += inp is None
            counts["missing_output"] += output is None
            success = (span.get("status") or {}).get("code") == 1
            counts["failed_calls"] += (span.get("status") or {}).get("code") == 2
            counts["unconfirmed_success"] += not success
            counts["zero_duration_calls"] += start == end
            counts["spans_with_events"] += bool(span.get("events") not in (None, "", "[]", "null"))
            identity = attrs.get("gen_ai.conversation.id")
            counts["missing_conversation_id"] += not bool(identity)
            # Missing IDs cannot establish that adjacent calls are one conversation.
            if not identity:
                continue
            streams.setdefault(identity, []).append((start, end, inp, output, attrs, success))
        for stream in streams.values():
            stream.sort(key=lambda s: s[0])
            for previous, following in zip(stream, stream[1:]):
                counts["adjacent_same_conversation_calls"] += 1
                gap = (following[0] - previous[1]).total_seconds()
                gaps.append(gap)
                counts["overlapping_calls"] += gap < 0
                inp, output, nxt = previous[2], previous[3], following[2]
                prefix = bool(inp and nxt and nxt[:len(inp)] == inp)
                counts["input_message_prefix_preserved"] += prefix
                completed = bool(prefix and output and nxt[:len(inp) + len(output)] == inp + output)
                counts["completed_message_prefix_preserved"] += completed
                a, b = previous[4], following[4]
                same_config = all(a.get(k) == b.get(k) for k in (
                    "gen_ai.request.model", "gen_ai.system_instructions", "gen_ai.tool.definitions"))
                counts["config_changes"] += not same_config
                counts["structurally_usable_sequential_pairs"] += (
                    completed and same_config and gap >= 0 and previous[5] and following[5]
                    and previous[1] > previous[0] and following[1] > following[0]
                )
    return dict(
        schema="hybrid-span-source-audit-v1", counts=dict(counts), cohorts=dict(cohorts),
        gap_range_s=[min(gaps), max(gaps)] if gaps else None,
        qualification="not_established",
        limitations=[
            "Message-prefix equality is not tokenizer/cache-key equality.",
            "Span clocks and source filtering must be verified independently.",
            "Missing output is unavailable, not an empty generated response.",
            "Last records are censored; they are not abandonment labels.",
            "No MSR features or held-out cache-value evidence are established here.",
        ],
    )


def audit_transcripts(paths):
    counts, by_cohort = Counter(), {}
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        counts["files"] += 1
        cohort = Path(path).name.split("_claude_code_")[0]
        by_cohort.setdefault(cohort, Counter())
        local = by_cohort[cohort]
        assistants = [r for r in rows if r.get("type") == "assistant"]
        counts["files_without_assistant"] += not bool(assistants)
        counts["records"] += len(rows)
        local["files"] += 1
        for i, row in enumerate(rows):
            if row.get("type") != "assistant":
                continue
            counts["assistant_records"] += 1
            local["assistant_records"] += 1
            following = rows[i + 1] if i + 1 < len(rows) else None
            if following is None:
                counts["censored_terminal_assistant"] += 1
                continue
            if following.get("type") != "user":
                counts["next_record_not_user"] += 1
                continue
            try:
                gap = (timestamp(following["timestamp"]) - timestamp(row["timestamp"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                counts["invalid_return_timestamps"] += 1
                continue
            counts["adjacent_assistant_user"] += 1
            counts["nonnegative_assistant_user_gap"] += gap >= 0
            local["adjacent_assistant_user"] += 1
            # Match a causally linked tool result, without reading any future text
            # as a predictor or calling every user record an HTTP arrival.
            output = (row.get("message") or {}).get("content") or []
            result = (following.get("message") or {}).get("content") or []
            calls = {p.get("id") for p in output if isinstance(p, dict) and p.get("type") == "tool_use"}
            replies = {p.get("tool_use_id") for p in result if isinstance(p, dict) and p.get("type") == "tool_result"}
            counts["linked_tool_result_pairs"] += bool((calls - {None}) & (replies - {None}))
        counts["files_with_explicit_system_message"] += any((r.get("message") or {}).get("role") == "system" for r in rows)
    return dict(
        schema="hybrid-transcript-source-audit-v1", counts=dict(counts),
        cohorts={k: dict(v) for k, v in by_cohort.items()}, qualification="not_established",
        limitations=[
            "Transcript timestamps alone do not establish API arrival or response-completion semantics.",
            "Tool-result availability is not necessarily the next model request time.",
            "Full model inputs, system instructions and tool schemas are not established by transcript content.",
            "Terminal records remain censored; no abandonment labels were generated.",
        ],
    )


def parquet_rows(paths):
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            yield from parquet.read_row_group(group, use_threads=False).to_pylist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["parquet", "transcript"])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Exclusive output: never overwrite an earlier audit.
    with args.output.open("x") as stream:
        result = audit_span_rows(parquet_rows(args.paths)) if args.kind == "parquet" else audit_transcripts(args.paths)
        result["inputs"] = []
        for path in args.paths:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            result["inputs"].append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()

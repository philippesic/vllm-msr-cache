"""Verify executed cadence from raw causal source and acknowledgement receipts."""

import collections
import hashlib
import json


def verify_cadence(raw, arm):
    hints = raw.get("hints")
    requests = raw.get("requests")
    issues = []
    if not isinstance(hints, list) or not isinstance(requests, list):
        return {"qualified": False, "issues": ["missing_rows"]}
    byindex = {q["index"]: q for q in requests}
    groups = collections.defaultdict(list)
    for h in hints:
        hint = h.get("hint", {})
        key = (
            hint.get("session_id"),
            hint.get("owner_id"),
            h.get("source_completed_at_ns"),
        )
        groups[key].append(h)
    receipts = []
    target = {
        "once_positive": "first_positive_cpu_recency_ack",
        "once_complete": "first_complete_cpu_recency_ack",
    }.get(arm)
    for key, seq in groups.items():
        seq.sort(key=lambda h: h["started_ns"])
        positive = complete = qualifying = stops = 0
        try:
            index = int(key[1].rsplit("sample-", 1)[1])
            previous = byindex[index - 1]
            current = byindex[index]
        except (ValueError, KeyError, AttributeError):
            issues.append("missing_causal_request")
            continue
        tokens = list(previous["payload"]["prompt"]) + previous["output_token_ids"][:-1]
        tokens = tokens[: len(tokens) // 16 * 16]
        for ordinal, h in enumerate(seq):
            hint = h["hint"]
            ack = h.get("response", {})
            ready = ack.get("ready_cpu_prefix_keys")
            valid = (
                h.get("http_status") == 200
                and ack.get("status") == "no_action"
                and ack.get("reason") == "cpu_prefix_recency_refreshed"
                and type(ready) is int
                and ready > 0
            )
            isfull = valid and ready == len(tokens) // 16
            positive += int(valid)
            complete += int(isfull)
            matches = (
                valid
                if arm == "once_positive"
                else isfull
                if arm == "once_complete"
                else False
            )
            qualifying += int(matches)
            expected = hashlib.sha256(
                json.dumps(
                    {"namespace": raw["namespace"], "token_ids": tokens},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if (
                key[0] != previous["session_id"]
                or key[0] != current["session_id"]
                or key[2] != previous["request_end_ns"]
                or hint.get("prefix_boundary_tokens") != len(tokens)
                or hint.get("observed_prefix_id") != expected
            ):
                issues.append("source_identity_or_completed_transcript_mismatch")
            if (
                not previous["request_end_ns"]
                <= h["started_ns"]
                <= current["request_start_ns"]
            ):
                issues.append("hint_started_outside_observed_pending_interval")
            marker = h.get("producer_policy_stop")
            if marker:
                stops += 1
                if marker != target or not matches or ordinal != len(seq) - 1:
                    issues.append("invalid_stop_or_post_after_stop")
            if target and matches and marker != target:
                issues.append("qualifying_ack_not_stopped")
        if target and (qualifying > 1 or stops > 1):
            issues.append("repeated_qualifying_ack_or_stop")
        receipts.append(
            {
                "session_id": key[0],
                "owner_id": key[1],
                "source_completed_at_ns": key[2],
                "hint_count": len(seq),
                "positive_acknowledgements": positive,
                "complete_acknowledgements": complete,
                "qualifying_acknowledgements": qualifying,
                "stop_markers": stops,
                "final_marker": seq[-1].get("producer_policy_stop"),
            }
        )
    if arm == "native" and hints:
        issues.append("native_emitted_hints")
    if target and not any(x["stop_markers"] for x in receipts):
        issues.append("once_policy_never_observed_stopping")
    return {
        "qualified": not issues,
        "issues": issues,
        "arm": arm,
        "hint_rows": len(hints),
        "wait_groups": len(groups),
        "stop_markers": sum(x["stop_markers"] for x in receipts),
        "positive_acknowledgements": sum(
            x["positive_acknowledgements"] for x in receipts
        ),
        "complete_acknowledgements": sum(
            x["complete_acknowledgements"] for x in receipts
        ),
        "groups": receipts,
        "scope": "Once-complete may have multiple positive partial acknowledgements before its single complete stop. Hints verified against actual prior completed transcript and before next request submission; private future wait schedule is not used for policy validation.",
    }

"""Explicit conversion of retained public sessions; does not execute legacy code."""

import argparse
import hashlib
import json

from .contracts import validate_corpus, write_json


def convert(data, source_hash, cohort=None):
    if data.get("schema") != "quantized-realworld-sessions-v1":
        raise ValueError("Expected quantized-realworld-sessions-v1")
    sessions = []
    seen = set()
    for s in data["sessions"]:
        if cohort and s.get("cohort") != cohort:
            continue
        if s["id"] in seen:
            raise ValueError("Duplicate source session identity")
        seen.add(s["id"])
        sessions.append(
            dict(
                id=s["id"],
                source_id=s.get("source_id"),
                turns=[t["user_text"] for t in s["turns"]],
                wait_s=[0.2 if i == 0 else 1.0 for i in range(len(s["turns"]) - 1)],
                source_metadata=s.get("metadata", {}),
            )
        )
    return validate_corpus(
        dict(
            schema="semantic-serving-corpus-v1",
            synthetic=False,
            provenance=f"Retained public content from quantized-realworld-sessions-v1; source SHA256 {source_hash}. Scripted arrivals and waits; recorded tools are not executed. Prior truncation, if any, is inherited. No real-user deployment.",
            source_sha256=source_hash,
            source_provenance=data.get("provenance"),
            sessions=sessions,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=__import__("pathlib").Path)
    p.add_argument("output", type=__import__("pathlib").Path)
    p.add_argument("--cohort", choices=["chat", "agent"])
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Refusing to overwrite corpus")
    raw = a.source.read_bytes()
    write_json(
        a.output, convert(json.loads(raw), hashlib.sha256(raw).hexdigest(), a.cohort)
    )
    print(a.output)


if __name__ == "__main__":
    main()

"""Reuse pilot feature captures only for byte-identical token inputs and code."""

import argparse
import hashlib
import json
from pathlib import Path


def inputs(path):
    manifest = json.loads((path / "manifest.json").read_text())
    raw = (path / "examples.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["examples_sha256"]:
        raise ValueError("Inputs changed")
    examples = [json.loads(line) for line in raw.splitlines()]
    if len({e["session_id"] for e in examples}) != len(examples):
        raise ValueError("Duplicate session")
    return manifest, examples


def features(path, expected):
    receipt = json.loads((path / "capture.json").read_text())
    if receipt["status"] != "complete" or receipt["manifest"] != expected:
        raise ValueError("Capture receipt mismatch")
    rows = [json.loads(line) for line in (path / "features.jsonl").read_text().splitlines()]
    if len({r["session_id"] for r in rows}) != len(rows) or len(rows) != receipt["count"]:
        raise ValueError("Duplicate or incomplete capture")
    return receipt, {r["session_id"]: r for r in rows}


def reusable(full_manifest, full, prior_manifest, prior, prior_features):
    for key in ["model", "capture_source_sha256"]:
        if full_manifest[key] != prior_manifest[key]:
            raise ValueError("Feature code/model changed")
    old = {e["session_id"]: e for e in prior}
    if old.keys() != prior_features.keys():
        raise ValueError("Prior feature population mismatch")
    reused, remaining = {}, []
    for example in full:
        sid = example["session_id"]
        if sid in old and example["tokens"] == old[sid]["tokens"]:
            reused[sid] = dict(prior_features[sid], split=example["split"])
        else:
            remaining.append(example)
    return reused, remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["remaining", "combine"])
    parser.add_argument("--full-inputs", required=True, type=Path)
    parser.add_argument("--prior-inputs", required=True, type=Path)
    parser.add_argument("--prior-capture", required=True, type=Path)
    parser.add_argument("--remaining-inputs", type=Path)
    parser.add_argument("--remaining-capture", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    full_manifest, full = inputs(args.full_inputs)
    prior_manifest, prior = inputs(args.prior_inputs)
    prior_receipt, prior_features = features(args.prior_capture, prior_manifest)
    reused, remaining = reusable(full_manifest, full, prior_manifest, prior, prior_features)
    args.output.mkdir(exist_ok=False)
    if args.action == "remaining":
        raw = "".join(json.dumps(e) + "\n" for e in remaining).encode()
        (args.output / "examples.jsonl").write_bytes(raw)
        manifest = dict(full_manifest, examples_sha256=hashlib.sha256(raw).hexdigest(),
                        parent_examples_sha256=full_manifest["examples_sha256"],
                        counts=dict(full_manifest["counts"], selected=len(remaining)), reused_examples=len(reused))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        remaining_manifest, remaining_examples = inputs(args.remaining_inputs)
        if remaining_examples != remaining:
            raise ValueError("Remaining input population changed")
        receipt, new = features(args.remaining_capture, remaining_manifest)
        if new.keys() != {e["session_id"] for e in remaining}:
            raise ValueError("Remaining feature population mismatch")
        all_features = reused | new
        with (args.output / "features.jsonl").open("x") as stream:
            for example in full:
                stream.write(json.dumps(all_features[example["session_id"]]) + "\n")
        receipt = dict(receipt, manifest=full_manifest, count=len(full),
                       combined_from=[str(args.prior_capture), str(args.remaining_capture)],
                       reused_examples=len(reused), provenance="Exact token inputs, model and capture source matched before reuse")
        (args.output / "capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(reused=len(reused), remaining=len(remaining), full=len(full))))


if __name__ == "__main__":
    main()

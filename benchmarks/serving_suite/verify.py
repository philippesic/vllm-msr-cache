"""Offline integrity and data-completeness verification; no GPU or model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .contracts import SCHEMA, write_json
from .report import paired_comparison, summarize_requests


def seal(directory):
    directory = Path(directory)
    files = {}
    for p in sorted(directory.rglob("*")):
        # The runtime owns and removes this ephemeral import stage. Its source
        # and wheel identities are retained separately; sealing its symlink
        # targets races shutdown and does not preserve executed bytes.
        if 'runtime-staging' in p.relative_to(directory).parts:
            continue
        if p.is_file() and p.name != "artifacts.json":
            with p.open("rb") as f:
                files[str(p.relative_to(directory))] = hashlib.file_digest(
                    f, "sha256"
                ).hexdigest()
    write_json(directory / "artifacts.json", {"schema": SCHEMA, "sha256": files})


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "artifacts.json").read_text())
    for name, expected in manifest["sha256"].items():
        p = directory / name
        if not p.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Artifact escapes result directory")
        if not p.is_file():
            raise ValueError(f"Missing artifact: {name}")
        with p.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Changed artifact: {name}")
    result = json.loads((directory / "result.json").read_text())
    if result["schema"] != SCHEMA:
        raise ValueError("Unsupported result schema")
    count = 0
    for case in result["cases"]:
        if result.get('plan',{}).get('tier') == 'agent' and case.get('agent'):
            if case.get('published_runtime'):
                from .published import diagnose_external as diagnose
            else:
                from .agent_report import diagnose
            effective_plan = {**result['plan'], **case.get('qualification_plan_overrides', {})}
            observed = diagnose(directory/case['name'], case, effective_plan, write=False)
            if observed != case['agent']:
                raise ValueError(f"Wrong agent diagnostics: {case['name']}")
            if json.loads((directory/case['name']/'agent.json').read_text()) != observed:
                raise ValueError(f"Wrong agent.json: {case['name']}")
        if result.get('plan', {}).get('campaign_adapter') and case.get('campaign'):
            from .campaign_workload import diagnose
            observed = diagnose(directory / case['name'], case, result['plan'], write=False)
            if case['campaign']['schema'] == 'serving-campaign-case-observation-v1':
                # The first development adapter did not expose class/phase or
                # bridge aggregates. Recompute every field that it did retain.
                observed = {key: observed[key] for key in case['campaign']}
                observed['schema'] = 'serving-campaign-case-observation-v1'
            if observed != case['campaign'] or load_json_campaign_observation(directory / case['name']) != observed:
                raise ValueError('Campaign workflow observation mismatch')
        rows = [
            json.loads(line)
            for line in (directory / case["name"] / "requests.jsonl")
            .read_text()
            .splitlines()
        ]
        recomputed = summarize_requests(rows, case["load_duration_s"])
        keys = [
            (r.get("wave"), r.get("source_id"), r.get("turn"))
            for r in rows
            if not r.get("cancel_probe")
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate request identity")
        if case["evaluation"]["status"] == "passed" and case.get("source_ids"):
            expected = {
                (w, s, t)
                for w in range(result["plan"]["waves"])
                for s in case["source_ids"]
                for t in range(result["plan"]["turns"])
            }
            if result["plan"].get("campaign_adapter"):
                from .campaign_workload import request_keys
                corpus = json.loads((directory / "corpus.json").read_text())
                expected = {tuple(key) for w in range(result["plan"]["waves"]) for key in request_keys(corpus["sessions"], w)}
            if set(keys) != expected:
                raise ValueError("Incomplete planned request identities")
        for key in (
            "completed",
            "failed",
            "output_tokens",
            "finish_reasons",
            "prompt_tokens",
            "content_checks",
            "content_checks_failed",
        ):
            if recomputed[key] != case["summary"][key]:
                raise ValueError(f"Wrong request aggregate: {case['name']}/{key}")
        for key in ("output_tokens_per_s", "ttft_mean_s", "ttft_p95_s", "e2e_p95_s",
                    "completed_requests_per_s", "client_queue_p95_s",
                    "end_user_ttft_mean_s", "end_user_ttft_p95_s"):
            if key not in case['summary']:
                continue  # Historical records predate the additional clocks.
            a, b = recomputed[key], case["summary"][key]
            if (a is None) != (b is None) or (
                a is not None and not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
            ):
                raise ValueError(f"Wrong timing aggregate: {case['name']}/{key}")
        if case["evaluation"]["status"] == "passed":
            if (
                recomputed["completed"] != case["expected_requests"]
                or recomputed["failed"]
            ):
                raise ValueError("Passed case has incomplete requests")
            if not case["cleanup"]["verified"]:
                raise ValueError("Passed case has unverified cleanup")
            for name in (
                "metrics-before.prom",
                "metrics-after.prom",
                "resources.jsonl",
            ):
                if not (directory / case["name"] / name).is_file():
                    raise ValueError(f"Missing required measurement: {name}")
        count += len(rows)
    # Recompute new whole-run primary estimates as well as individual clocks.
    # Historical records lack these fields and keep their original contract.
    if 'geometric_workflow_throughput_ratio' in (result.get('comparison') or {}):
        cases=result['cases']
        if result['plan'].get('published_runtimes'):
            from .published import comparisons
            expected_comparisons = comparisons(cases)
            if result.get('comparisons') != expected_comparisons:
                raise ValueError('Wrong five-way primary comparisons')
            expected = expected_comparisons['wait_vs_arc']
        else:
            expected=paired_comparison([c for c in cases if c.get('variant')!='semantic'])
        if result['comparison'] != expected:
            raise ValueError('Wrong paired primary comparison')
        if result['plan'].get('semantic_config') and not result['plan'].get('published_runtimes'):
            comparisons=dict(wait_vs_arc=expected,
                semantic_vs_arc=paired_comparison([c for c in cases if c.get('variant') in ('arc','semantic')]),
                semantic_vs_wait=paired_comparison([{**c,'arm':'arc' if c['variant']=='wait' else 'candidate'}
                    for c in cases if c.get('variant') in ('wait','semantic')]))
            if result.get('comparisons') != comparisons:
                raise ValueError('Wrong three-way primary comparisons')
    return {
        "verified_files": len(manifest["sha256"]),
        "cases": len(result["cases"]),
        "request_records": count,
        "measurement_status": result["status"],
        "scope": "Artifact integrity and aggregate consistency; not a new GPU/quality qualification.",
    }


def load_json_campaign_observation(directory):
    return json.loads((directory / "campaign-observation.json").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.directory), indent=2))


if __name__ == "__main__":
    main()

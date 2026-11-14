from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
import shutil
import sqlite3
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from tools.config import check_environment
from tools.process import run_owned
from tools.runtime_verify import snapshot, write_record

PARTITIONS = ("calibration", "pilot", "confirmation")
MODEL_REVISIONS = {
    "D8": "4da05a8edb55c6046cce958586c33b61da07bb79",
    "HM": "d3e78b50c727b1f34802f440ca84b25781625352",
    "HS": "7c781be42ce8d29d0975b857f6d0c0b2558d9b63",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def repository(task: str) -> str:
    match = re.match(r"^(.+__.+)\.([0-9a-f]{8,40})\.", task)
    if match is None:
        raise ValueError(f"unmapped repository identity: {task}")
    return match[1].replace("__", "/", 1)


def trajectory_order(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return row["sort_key"], row["trajectory"], row["shard"], row["row"]


def partition(repo: str) -> str:
    bucket = int(text_digest(repo), 16) % 10
    return PARTITIONS[0 if bucket < 2 else 1 if bucket < 4 else 2]


def canonical_messages(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for message in raw:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported_message_role")
        content = message.get("content", "")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("non_text_content")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("non_text_content")
        item: dict[str, Any] = {"role": role, "content": content}
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("non_object_tool_arguments")
                calls.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": function["name"], "arguments": arguments},
                    }
                )
            item["tool_calls"] = calls
        if role == "tool":
            ids = message.get("tool_call_ids", [])
            if len(ids) != 1:
                raise ValueError("ambiguous_tool_result_identity")
            item["tool_call_id"] = ids[0]
        messages.append(item)
    return messages


def lcp(left: list[int], right: list[int]) -> int:
    return next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


def episode(messages: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    ending = "natural"
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if len(turns) == 8:
            ending = "turn_limit"
            break
        prompt = tokenizer.encode(
            tokenizer.apply_chat_template(
                messages[:index],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            add_special_tokens=False,
        )
        complete = tokenizer.encode(
            tokenizer.apply_chat_template(
                messages[: index + 1],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            ),
            add_special_tokens=False,
        )
        output_lcp = lcp(prompt, complete)
        recorded_output = max(1, len(complete) - output_lcp)
        requested = min(256, recorded_output)
        if len(prompt) + requested > 6144:
            ending = "context_limit"
            break
        reuse = max((lcp(turn["prompt_tokens"], prompt) for turn in turns), default=0)
        previous = max((len(turn["prompt_tokens"]) for turn in turns), default=0)
        turns.append(
            {
                "message_index": index,
                "prompt_tokens": prompt,
                "recorded_output_tokens": recorded_output,
                "requested_output_tokens": requested,
                "output_template_prefix_changed_tokens": len(prompt) - output_lcp,
                "previous_request_lcp_tokens": reuse,
                "previous_request_lost_suffix_tokens": previous - reuse,
                "post_response_eligible": any(
                    call["function"]["name"] != "submit"
                    for call in message.get("tool_calls", [])
                ),
            }
        )
    if len(turns) < 2:
        return {"excluded": "fewer_than_two_turns", "ending": ending}
    peak = max(
        len(turn["prompt_tokens"]) + turn["requested_output_tokens"] for turn in turns
    )
    return {
        "turns": turns,
        "ending": ending,
        "stratum": "0-2048"
        if peak <= 2048
        else "2049-4096"
        if peak <= 4096
        else "4097-6144",
    }


def models_from_args(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        directory = Path(path).absolute()
        if (
            name not in MODEL_REVISIONS
            or name in result
            or directory.name != MODEL_REVISIONS[name]
        ):
            raise ValueError("unknown, duplicate or unpinned W1 tokenizer")
        result[name] = directory
    if not result:
        raise ValueError("at least one tokenizer required")
    return result


def admit_native(dataset: Path, models: list[str], out: Path) -> None:
    parquet = importlib.import_module("pyarrow.parquet")
    transformers = importlib.import_module("transformers")
    root = Path(__file__).resolve().parents[2]
    source = json.loads((root / "benchmarks/pr-evidence/w1-source.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    for name, expected in source["files"].items():
        if digest(dataset / name) != expected:
            raise ValueError(f"dataset source drift: {name}")
    model_paths = models_from_args(models)
    tokenizers = {
        name: transformers.AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        for name, path in model_paths.items()
    }
    model_identity = {
        name: {
            "revision": path.name,
            "tokenizer_class": type(tokenizers[name]).__name__,
            "files": {
                item.name: digest(item)
                for item in sorted(path.iterdir())
                if item.suffix in {".json", ".jinja", ".txt"}
            },
        }
        for name, path in model_paths.items()
    }
    selected: dict[str, dict[str, Any]] = {}
    source_repos: dict[str, Counter[str]] = {}
    source_rows = 0
    for name in sorted(source["files"]):
        offset = 0
        for batch in parquet.ParquetFile(dataset / name).iter_batches(
            columns=["instance_id", "traj_id", "resolved"]
        ):
            for row in batch.to_pylist():
                task = row["instance_id"]
                repo = repository(task)
                source_repos.setdefault(repo, Counter())[str(row["resolved"])] += 1
                candidate = {
                    "task": task,
                    "trajectory": row["traj_id"],
                    "repository": repo,
                    "partition": partition(repo),
                    "shard": name,
                    "row": offset,
                    "resolved": row["resolved"],
                    "sort_key": text_digest(row["traj_id"]),
                }
                previous = selected.get(task)
                if previous is None or trajectory_order(candidate) < trajectory_order(
                    previous
                ):
                    selected[task] = candidate
                offset += 1
                source_rows += 1
    write_record(out / "deduplicated.json", selected)
    connection = sqlite3.connect(out / "episodes.sqlite")
    connection.execute(
        "CREATE TABLE episodes (model TEXT, task TEXT, partition TEXT, sort_key TEXT, metadata TEXT, payload BLOB, PRIMARY KEY(model,task))"
    )
    exclusions: dict[str, Counter[str]] = {name: Counter() for name in model_paths}
    message_fields: Counter[str] = Counter()
    processed = 0
    try:
        for name in sorted(source["files"]):
            offset = 0
            for batch in parquet.ParquetFile(dataset / name).iter_batches(
                batch_size=8, columns=["instance_id", "messages"]
            ):
                for row in batch.to_pylist():
                    chosen = selected[row["instance_id"]]
                    row_number = offset
                    offset += 1
                    if (chosen["shard"], chosen["row"]) != (name, row_number):
                        continue
                    raw = json.loads(row["messages"])
                    message_fields.update(key for message in raw for key in message)
                    try:
                        messages = canonical_messages(raw)
                    except (KeyError, TypeError, ValueError) as error:
                        for model in model_paths:
                            exclusions[model][
                                f"{chosen['partition']}:canonical:{error}"
                            ] += 1
                        continue
                    for model, tokenizer in tokenizers.items():
                        try:
                            value = episode(messages, tokenizer)
                        except (KeyError, TypeError, ValueError) as error:
                            exclusions[model][
                                f"{chosen['partition']}:render:{type(error).__name__}:{error}"
                            ] += 1
                            continue
                        if "excluded" in value:
                            exclusions[model][
                                f"{chosen['partition']}:{value['excluded']}:{value['ending']}"
                            ] += 1
                            continue
                        payload = json.dumps(value, separators=(",", ":")).encode()
                        metadata = chosen | {
                            "messages_sha256": text_digest(row["messages"]),
                            "episode_sha256": hashlib.sha256(payload).hexdigest(),
                            "turns": len(value["turns"]),
                            "ending": value["ending"],
                            "stratum": value["stratum"],
                            "prompt_tokens": [
                                len(turn["prompt_tokens"]) for turn in value["turns"]
                            ],
                            "requested_output_tokens": [
                                turn["requested_output_tokens"]
                                for turn in value["turns"]
                            ],
                            "capped_outputs": sum(
                                turn["recorded_output_tokens"] > 256
                                for turn in value["turns"]
                            ),
                        }
                        connection.execute(
                            "INSERT INTO episodes VALUES (?,?,?,?,?,?)",
                            (
                                model,
                                chosen["task"],
                                chosen["partition"],
                                chosen["sort_key"],
                                json.dumps(metadata, sort_keys=True),
                                zlib.compress(payload),
                            ),
                        )
                    processed += 1
                    if processed % 200 == 0:
                        connection.commit()
                        print(
                            f"W1 processed {processed}/{len(selected)} distinct tasks",
                            flush=True,
                        )
            connection.commit()
        admitted: dict[str, dict[str, Any]] = {}
        for model in model_paths:
            admitted[model] = {}
            for part in PARTITIONS:
                rows = connection.execute(
                    "SELECT metadata FROM episodes WHERE model=? AND partition=? ORDER BY sort_key,task LIMIT 2048",
                    (model, part),
                ).fetchall()
                entries = [json.loads(row[0]) for row in rows]
                eligible = connection.execute(
                    "SELECT count(*) FROM episodes WHERE model=? AND partition=?",
                    (model, part),
                ).fetchone()[0]
                admitted[model][part] = {
                    "eligible_tasks": eligible,
                    "admitted_tasks": len(entries),
                    "repositories": len({entry["repository"] for entry in entries}),
                    "stratum_counts": dict(
                        Counter(entry["stratum"] for entry in entries)
                    ),
                    "entries": entries,
                }
    finally:
        connection.close()
    write_record(
        out / "manifest.json",
        {
            "schema": "pr-w1-manifest-v1",
            "label": "semi-synthetic transcript replay",
            "source": source,
            "source_rows": source_rows,
            "source_repositories": source_repos,
            "deduplicated_tasks": len(selected),
            "partition_tasks": dict(
                Counter(row["partition"] for row in selected.values())
            ),
            "repository_partition": {
                repo: partition(repo) for repo in sorted(source_repos)
            },
            "message_fields": message_fields,
            "models": model_identity,
            "admitted": admitted,
            "exclusions": exclusions,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("transformers", "tokenizers", "pyarrow")
            },
            "recipe": {
                "deduplication": "minimum (SHA256 trajectory ID, trajectory ID, shard, row) per instance_id",
                "repository": "instance_id prefix before .hexcommit.; first __ becomes /",
                "partition": "integer SHA256(repository) mod 10: 0-1 calibration, 2-3 pilot, 4-9 confirmation",
                "sampling": "first 2048 eligible distinct tasks per model/partition sorted by SHA256 trajectory ID then task; no success filter",
                "messages": "text parts concatenated; original content and tool calls/results preserved; duplicate thought/action and trace-only metadata omitted; JSON tool arguments decoded",
                "tool_schema": "none supplied by source, none invented; terminal function name exactly submit",
                "template": "pinned native tokenizer chat template; enable_thinking=False; add_generation_prompt=True for requests; no additional special tokens",
                "output_length": "completed-transcript tokens after token LCP with request rendering, minimum 1; cap 256; mismatch at generation prefix recorded",
                "episodes": "chronological first 2-8 assistant turns, stop at first prompt+requested-output >6144; no middle removal",
                "reuse": "maximum exact token LCP with earlier request prompts; group alignment deferred to admitted native layout",
                "strata": "peak prompt+requested-output <=2048, <=4096, <=6144; inclusion-share weights",
                "timing": "source has no usable tool gaps; declared seeded lognormal median4s sigma0.8 clamp300s; causal TTL12s clamp60s",
            },
            "output_sha256": {
                name: digest(out / name)
                for name in ("deduplicated.json", "episodes.sqlite")
            },
        },
    )


def admit(
    runtime_python: Path, dataset: Path, models: list[str], out: Path
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    identity = snapshot(root, source)
    write_record(
        out / "admission.json",
        {
            "schema": "pr-w1-admission-v1",
            "source_sha256": identity,
            "models": models,
            "dataset": str(dataset.absolute()),
            "gpu_exposure_seconds": 0,
        },
    )
    captured = source / "artifacts/w1"
    command = [
        str(runtime_python.absolute()),
        "-m",
        "tools.benchmark",
        "pr-w1-native",
        "--dataset",
        str(dataset.absolute()),
        "--out",
        str(captured),
    ]
    for model in models:
        command.extend(("--model", model))
    started = time.monotonic()
    with (out / "execution.log").open("x") as log:
        code = run_owned(command, check_environment(source), source, 3600, log)
    if captured.exists():
        shutil.copytree(captured, out / "data")
    write_record(
        out / "terminal.json",
        {
            "schema": "pr-w1-terminal-v1",
            "exit_code": code,
            "wall_seconds": time.monotonic() - started,
            "execution_sha256": digest(out / "execution.log"),
            "manifest_sha256": digest(out / "data/manifest.json")
            if (out / "data/manifest.json").exists()
            else None,
        },
    )
    return verify(out)


def verify(out: Path) -> dict[str, Any]:
    admission = json.loads((out / "admission.json").read_text())
    terminal = json.loads((out / "terminal.json").read_text())
    if (
        admission["schema"] != "pr-w1-admission-v1"
        or terminal["schema"] != "pr-w1-terminal-v1"
        or terminal["exit_code"] != 0
        or admission["gpu_exposure_seconds"] != 0
    ):
        raise ValueError("unqualified W1 admission attempt")
    for path, expected in admission["source_sha256"].items():
        if (
            Path(path).is_absolute()
            or ".." in Path(path).parts
            or digest(out / "source" / path) != expected
        ):
            raise ValueError("W1 source snapshot drift")
    if (
        digest(out / "execution.log") != terminal["execution_sha256"]
        or digest(out / "data/manifest.json") != terminal["manifest_sha256"]
    ):
        raise ValueError("W1 receipt drift")
    manifest = json.loads((out / "data/manifest.json").read_text())
    if manifest["schema"] != "pr-w1-manifest-v1":
        raise ValueError("unknown W1 manifest")
    for name, expected in manifest["output_sha256"].items():
        if Path(name).name != name or digest(out / "data" / name) != expected:
            raise ValueError("W1 data drift")
    connection = sqlite3.connect(
        f"file:{out / 'data/episodes.sqlite'}?mode=ro", uri=True
    )
    try:
        for model, parts in manifest["admitted"].items():
            for part, group in parts.items():
                expected = connection.execute(
                    "SELECT task FROM episodes WHERE model=? AND partition=? ORDER BY sort_key,task LIMIT 2048",
                    (model, part),
                ).fetchall()
                entries = group["entries"]
                if [entry["task"] for entry in entries] != [
                    row[0] for row in expected
                ] or len(entries) != group["admitted_tasks"]:
                    raise ValueError("W1 selection drift")
                for entry in entries:
                    data = connection.execute(
                        "SELECT payload FROM episodes WHERE model=? AND task=?",
                        (model, entry["task"]),
                    ).fetchone()[0]
                    payload = zlib.decompress(data)
                    if hashlib.sha256(payload).hexdigest() != entry["episode_sha256"]:
                        raise ValueError("W1 episode drift")
                    value = json.loads(payload)
                    if not 2 <= len(value["turns"]) <= 8:
                        raise ValueError("W1 turn constraint violated")
                    for turn in value["turns"]:
                        if (
                            len(turn["prompt_tokens"]) + turn["requested_output_tokens"]
                            > 6144
                            or not 1 <= turn["requested_output_tokens"] <= 256
                        ):
                            raise ValueError("W1 context constraint violated")
    finally:
        connection.close()
    return {
        "schema": "pr-w1-verification-v1",
        "manifest_sha256": terminal["manifest_sha256"],
        "source_rows": manifest["source_rows"],
        "deduplicated_tasks": manifest["deduplicated_tasks"],
        "admitted_tasks": {
            model: {part: value["admitted_tasks"] for part, value in parts.items()}
            for model, parts in manifest["admitted"].items()
        },
    }


def freeze(admission_path: Path, out: Path) -> dict[str, Any]:
    verification = verify(admission_path)
    manifest = json.loads((admission_path / "data/manifest.json").read_text())
    selected = json.loads((admission_path / "data/deduplicated.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(
        f"file:{admission_path / 'data/episodes.sqlite'}?mode=ro", uri=True
    )
    exclusions: dict[str, list[dict[str, str]]] = {}
    summaries: dict[str, Any] = {}
    try:
        for model, parts in manifest["admitted"].items():
            eligible = {
                row[0]
                for row in connection.execute(
                    "SELECT task FROM episodes WHERE model=?", (model,)
                )
            }
            exclusions[model] = []
            for task in sorted(set(selected) - eligible):
                part = selected[task]["partition"]
                reasons = [
                    key
                    for key in manifest["exclusions"][model]
                    if key.startswith(part + ":")
                ]
                if len(reasons) != 1:
                    raise ValueError(
                        "per-task exclusion reason cannot be reconstructed uniquely"
                    )
                exclusions[model].append(
                    {"task": task, "partition": part, "reason": reasons[0]}
                )
            if len(exclusions[model]) != sum(manifest["exclusions"][model].values()):
                raise ValueError("exclusion accounting does not reconcile")
            summaries[model] = {}
            for part, group in parts.items():
                entries = group["entries"]
                turn_counts = Counter(str(entry["turns"]) for entry in entries)
                endings = Counter(entry["ending"] for entry in entries)
                turns = sum(entry["turns"] for entry in entries)
                summaries[model][part] = {
                    "eligible_tasks": group["eligible_tasks"],
                    "admitted_tasks": len(entries),
                    "repositories": group["repositories"],
                    "stratum_counts": group["stratum_counts"],
                    "stratum_weight_denominator": len(entries),
                    "turn_count_distribution": turn_counts,
                    "endings": endings,
                    "turns": turns,
                    "capped_outputs": sum(entry["capped_outputs"] for entry in entries),
                    "resolved_tasks": sum(entry["resolved"] for entry in entries),
                    "selection_sha256": text_digest(
                        json.dumps(
                            [entry["trajectory"] for entry in entries],
                            separators=(",", ":"),
                        )
                    ),
                }
    finally:
        connection.close()
    write_record(out / "exclusions.json", exclusions)
    frozen = {
        "schema": "pr-w1-freeze-v1",
        "manifest_sha256": verification["manifest_sha256"],
        "admission_receipt_sha256": digest(admission_path / "admission.json"),
        "source": manifest["source"],
        "recipe": manifest["recipe"],
        "models": manifest["models"],
        "dependencies": manifest["dependencies"],
        "source_rows": manifest["source_rows"],
        "source_repositories": len(manifest["source_repositories"]),
        "source_resolved_rows": sum(
            row.get("True", 0) for row in manifest["source_repositories"].values()
        ),
        "deduplicated_tasks": manifest["deduplicated_tasks"],
        "partition_tasks": manifest["partition_tasks"],
        "source_repositories_per_partition": dict(
            Counter(manifest["repository_partition"].values())
        ),
        "repository_mapping_sha256": text_digest(
            json.dumps(
                manifest["repository_partition"], sort_keys=True, separators=(",", ":")
            )
        ),
        "admitted": summaries,
        "exclusion_counts": manifest["exclusions"],
        "exclusions_sha256": digest(out / "exclusions.json"),
        "data_sha256": manifest["output_sha256"],
        "scope": "content admission only; confirmation partition remains unexposed to performance development; hardware model admission remains pending",
    }
    write_record(out / "freeze.json", frozen)
    return {
        "schema": "pr-w1-freeze-summary-v1",
        "manifest_sha256": verification["manifest_sha256"],
        "freeze_sha256": digest(out / "freeze.json"),
        "admitted": summaries,
    }

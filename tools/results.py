from __future__ import annotations

import hashlib
import io
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

REPOSITORIES = {"legacy", "current"}
VERDICTS = {"supported", "negative", "mixed", "unresolved", "functional"}
FIELDS = {
    "schema",
    "title",
    "date",
    "origin",
    "verdict",
    "summary",
    "scope",
    "limitations",
    "decision",
    "sources",
    "chart",
    "correction",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_result(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    if set(data) != FIELDS or data["schema"] != 1:
        raise ValueError("require result schema 1 and its exact fields")
    for key in ("title", "date", "summary", "scope", "limitations", "decision"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"missing result {key}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["date"]):
        raise ValueError("date must be YYYY-MM-DD")
    if (
        data["origin"] not in {"current", "historical"}
        or data["verdict"] not in VERDICTS
        or not isinstance(data["correction"], str)
    ):
        raise ValueError("invalid origin, verdict or correction")
    if not isinstance(data["sources"], list) or not data["sources"]:
        raise ValueError("result needs sources")
    for source in data["sources"]:
        if (
            set(source) != {"repository", "revision", "path", "sha256"}
            or source["repository"] not in REPOSITORIES
        ):
            raise ValueError("invalid source fields")
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]) or not re.fullmatch(
            r"[0-9a-f]{64}", source["sha256"]
        ):
            raise ValueError("source needs full revision and SHA-256")
        if Path(source["path"]).is_absolute() or ".." in Path(source["path"]).parts:
            raise ValueError("source path must be repository-relative")
    chart = data["chart"]
    if chart is not None:
        if (
            set(chart) != {"metric", "caption", "baseline", "points"}
            or not chart["metric"]
            or not chart["caption"]
            or not chart["points"]
        ):
            raise ValueError("chart requires metric, caption, baseline and points")
        for point in chart["points"]:
            if (
                set(point) != {"label", "estimate", "lower", "upper"}
                or not point["label"]
            ):
                raise ValueError("invalid chart point")
            numbers = [point["estimate"]]
            if (point["lower"] is None) != (point["upper"] is None):
                raise ValueError("both interval endpoints are required")
            if point["lower"] is not None:
                numbers += [point["lower"], point["upper"]]
            if chart["baseline"] is not None:
                numbers += [chart["baseline"]]
            if any(
                type(n) not in {float, int} or not math.isfinite(n) for n in numbers
            ):
                raise ValueError("chart numbers must be finite")
            if (
                point["lower"] is not None
                and not point["lower"] <= point["estimate"] <= point["upper"]
            ):
                raise ValueError("interval must contain estimate")
    return data


def figure(chart: dict[str, Any]) -> bytes:
    from matplotlib import rc_context
    from matplotlib.figure import Figure

    with rc_context(
        {
            "svg.hashsalt": "result-chart-v1",
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
        }
    ):
        fig = Figure(
            figsize=(9, max(3, len(chart["points"]) * 0.42 + 1.4)), layout="constrained"
        )
        ax = fig.subplots()
        for row, point in enumerate(chart["points"]):
            ax.plot(point["estimate"], row, "o", color="#176b87", markersize=6)
            if point["lower"] is not None:
                ax.plot(
                    [point["lower"], point["upper"]],
                    [row, row],
                    color="#176b87",
                    linewidth=2,
                )
        if chart["baseline"] is not None:
            ax.axvline(chart["baseline"], color="#666666", linestyle="--", linewidth=1)
        ax.set_yticks(
            range(len(chart["points"])), [p["label"] for p in chart["points"]]
        )
        ax.invert_yaxis()
        ax.set_xlabel(chart["metric"])
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="svg",
            metadata={"Date": None, "Description": chart["caption"]},
        )
        return buffer.getvalue()


def page(data: dict[str, Any]) -> str:
    text = f"# {data['title']}\n\n{data['date']} · {data['origin']} · {data['verdict']}\n\n{data['summary']}\n"
    if data["chart"] is not None:
        text += (
            f"\n![{data['chart']['metric']}](chart.svg)\n\n{data['chart']['caption']}\n"
        )
    for heading, key in (
        ("Scope", "scope"),
        ("Limits", "limitations"),
        ("Decision", "decision"),
    ):
        text += f"\n## {heading}\n\n{data[key]}\n"
    if data["correction"]:
        text += f"\n## Correction\n\n{data['correction']}\n"
    text += "\n## Sources\n\n"
    for source in data["sources"]:
        if source["repository"] == "legacy":
            target = f"https://github.com/philippesic/vllm-msr-cache/blob/{source['revision']}/{source['path']}"
            text += f"- [{source['path']}]({target})\n"
        else:
            text += (
                f"- `{source['repository']}:{source['revision']}:{source['path']}`\n"
            )
    return (
        text + "\nExact source hashes and figure inputs: [result.json](result.json).\n"
    )


def index(records: list[tuple[Path, dict[str, Any]]]) -> str:
    text = "# Findings\n\nShort conclusion/link index; edit result records, then regenerate. Full evidence\nand limitations live in the linked results. Historical results describe a\ndifferent implementation and do not qualify this code. See the [result profile](docs/results.md).\n"
    for origin, title in (
        ("current", "Current code"),
        ("historical", "Earlier research"),
    ):
        text += f"\n## {title}\n\n"
        for path, data in records:
            if data["origin"] == origin:
                text += f"- **{data['verdict']}** — [{data['title']}](results/{path.parent.name}/result.md)\n"
    return text


def records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (p, read_result(p)) for p in sorted((root / "results").glob("*/result.json"))
    ]


def render(root: Path) -> None:
    items = records(root)
    for path, data in items:
        outputs = {"result.md": page(data).encode()}
        if data["chart"] is not None:
            outputs["chart.svg"] = figure(data["chart"])
        else:
            (path.parent / "chart.svg").unlink(missing_ok=True)
        for name, payload in outputs.items():
            (path.parent / name).write_bytes(payload)
        receipt = {
            "input_sha256": digest(path.read_bytes()),
            "outputs": {n: digest(b) for n, b in outputs.items()},
        }
        (path.parent / "rendered.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
    (root / "FINDINGS.md").write_text(index(items))


def check(root: Path) -> list[str]:
    if not (root / "results").exists():
        return []
    errors = []
    try:
        items = records(root)
        if not items:
            raise ValueError("results directory has no results")
        for path, data in items:
            for source in data["sources"]:
                archive = json.loads(
                    (
                        root / "results/sources" / (source["sha256"] + ".json")
                    ).read_text()
                )
                if (
                    archive["source"] != source
                    or digest(archive["content"].encode()) != source["sha256"]
                ):
                    raise ValueError(f"source archive mismatch: {source['path']}")
            receipt = json.loads((path.parent / "rendered.json").read_text())
            names = (
                {"result.md", "chart.svg"}
                if data["chart"] is not None
                else {"result.md"}
            )
            if (
                receipt["input_sha256"] != digest(path.read_bytes())
                or set(receipt["outputs"]) != names
            ):
                raise ValueError(f"stale rendering: {path.parent.name}")
            if (path.parent / "result.md").read_text() != page(data):
                raise ValueError(f"edited generated page: {path.parent.name}")
            for name, expected in receipt["outputs"].items():
                if digest((path.parent / name).read_bytes()) != expected:
                    raise ValueError(f"changed output: {path.parent.name}/{name}")
        if (root / "FINDINGS.md").read_text() != index(items):
            raise ValueError("FINDINGS index is stale")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(
            f"results.invalid: {error}; fix result.json then run tools/check.py --render-results"
        )
    return errors


def verify_sources(root: Path, repository: Path) -> None:
    checked: set[tuple[str, str]] = set()
    missing: set[tuple[str, str]] = set()
    for _path, data in records(root):
        for source in data["sources"]:
            identity = (source["revision"], source["path"])
            if identity in checked or identity in missing:
                continue
            command = [
                "git",
                "-C",
                str(repository),
                "show",
                f"{identity[0]}:{identity[1]}",
            ]
            result = subprocess.run(command, capture_output=True, check=False)
            if result.returncode:
                missing.add(identity)
            elif digest(result.stdout) != source["sha256"]:
                raise ValueError(f"source hash mismatch: {identity}")
            else:
                checked.add(identity)
    print(
        f"Sources verified: {len(checked)}; unavailable in this repository: {len(missing)}"
    )
    if not checked:
        raise ValueError("no sources verified; provide the correct Git archive")

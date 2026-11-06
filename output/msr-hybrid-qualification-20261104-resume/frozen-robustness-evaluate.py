"""Evaluate the frozen returning-only feature pilot, never qualify a live policy."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def pending_tool_metadata(manifest, examples):
    """Recover only already-issued tool names as a stronger cheap baseline."""
    import pyarrow.parquet as pq
    from benchmarks.research_checks.hybrid_feature_pilot import completed_example

    selected = {r["session_id"]: r for r in examples}
    names = {}
    previous, turn = None, 0
    for source in manifest["sources"]:
        path = Path(source["path"])
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source["sha256"]:
                raise ValueError("Original source changed")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1, use_threads=False):
            row = batch.to_pylist()[0]
            if previous is None or row["session_id"] != previous["session_id"]:
                turn = 0
            else:
                turn += 1
                sid = row["session_id"]
                if sid in selected and turn == selected[sid]["cheap"][0]:
                    example = completed_example(previous, row, turn)
                    if example is None or example["cheap"] != selected[sid]["cheap"][:-1]:
                        raise ValueError("Selected causal prefix could not be reconstructed")
                    # Prefix ends at the response, before next tool/user input.
                    issued = example["messages"][-1].get("tool_calls", [])
                    names[sid] = "+".join(sorted((c.get("function") or {}).get("name") or "unknown_tool" for c in issued)) or "text_only"
            previous = row
    if names.keys() != selected.keys():
        raise ValueError("Incomplete cheap tool metadata")
    return names


def fit_logistic(x, y, regularization):
    weights = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], regularization / len(y))
    penalty[0] = 0

    def loss(w):
        z = x @ w
        return np.mean(np.logaddexp(0, z) - y * z) + .5 * np.sum(penalty * w * w)

    for _ in range(50):
        z = np.clip(x @ weights, -30, 30)
        probability = 1 / (1 + np.exp(-z))
        gradient = x.T @ (probability - y) / len(y) + penalty * weights
        hessian = (x.T * (probability * (1 - probability))) @ x / len(y) + np.diag(penalty + 1e-8)
        step = np.linalg.solve(hessian, gradient)
        scale = 1.
        before = loss(weights)
        while scale > 1e-6 and loss(weights - scale * step) > before:
            scale *= .5
        weights -= scale * step
        if np.max(np.abs(scale * step)) < 1e-6:
            break
    return weights


def probability(x, weights):
    return 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))


def interval_fields(bounds, alpha):
    result = dict(paired_interval=bounds, confidence=1 - alpha)
    if abs(alpha - .05) < 1e-12:
        result["paired_95_interval"] = bounds
    return result


def paired_interval(improvement, groups, seed=63104, alpha=.05):
    unique = sorted(set(groups))
    values = np.array([np.mean([v for v, g in zip(improvement, groups) if g == group]) for group in unique])
    if len(values) < 2:
        return dict(tasks=len(values), mean=None, **interval_fields(None, alpha))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    return dict(tasks=len(values), mean=float(values.mean()), **interval_fields(np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist(), alpha))


def retained_value(prediction, labels, tokens):
    budget = .25 * sum(tokens)
    value = used = 0.
    for i in sorted(range(len(tokens)), key=lambda i: (-prediction[i], i)):
        if used + tokens[i] <= budget:
            used += tokens[i]
            value += tokens[i] * labels[i]
    return dict(useful_tokens=float(value), retained_tokens=float(used), budget_tokens=float(budget))


def retention_interval(baseline, candidate, labels, tokens, groups, alpha=.05):
    """Paired task bootstrap of the explicitly hypothetical capacity exercise."""
    unique = sorted(set(groups))
    positions = {group: [i for i, value in enumerate(groups) if value == group] for group in unique}
    rng = np.random.default_rng(63104)
    changes = []
    for _ in range(10000):
        selected = [i for group in rng.choice(unique, len(unique)) for i in positions[group]]
        lengths = [tokens[i] for i in selected]
        outcomes = [labels[i] for i in selected]
        base = retained_value([baseline[i] for i in selected], outcomes, lengths)
        new = retained_value([candidate[i] for i in selected], outcomes, lengths)
        changes.append((new["useful_tokens"] - base["useful_tokens"]) / base["budget_tokens"])
    return dict(metric="Useful-token difference divided by common hypothetical token budget",
                **interval_fields(np.quantile(changes, [alpha / 2, 1 - alpha / 2]).tolist(), alpha),
                scope="Synthetic simultaneous pause, equal per-token costs, no shared-prefix deduplication; not observed cache competition")


def cheap_design(numeric, categories, train, interactions=False):
    """Build metadata controls using a vocabulary learned on training data only."""
    numeric = np.log1p(numeric)
    vocabulary = sorted({(column, category[column]) for category, keep in zip(categories, train)
                         if keep for column in range(len(category))})
    categorical = np.array([[category[column] == value for column, value in vocabulary]
                            for category in categories], dtype=float)
    parts = [numeric, numeric ** 2, categorical]
    if interactions:
        parts.extend(numeric[:, i:i + 1] * numeric[:, i + 1:] for i in range(numeric.shape[1] - 1))
        parts.extend(numeric[:, i:i + 1] * categorical for i in range(numeric.shape[1]))
        combinations = sorted({tuple(category) for category, keep in zip(categories, train) if keep})
        parts.append(np.array([[tuple(category) == value for value in combinations] for category in categories]))
    return np.column_stack(parts)


def evaluate(inputs, capture, output, *, horizon=1.0, family_size=1, cheap_interactions=False):
    if horizon <= 0 or family_size < 1:
        raise ValueError("Positive horizon and family size required")
    manifest = json.loads((inputs / "manifest.json").read_text())
    raw = (inputs / "examples.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["examples_sha256"]:
        raise ValueError("Prepared inputs changed")
    receipt = json.loads((capture / "capture.json").read_text())
    if receipt["status"] != "complete" or receipt["manifest"] != manifest:
        raise ValueError("Capture must be complete and bound to these inputs")
    rows = [json.loads(line) for line in raw.splitlines()]
    features = {r["session_id"]: r for r in map(json.loads, (capture / "features.jsonl").read_text().splitlines())}
    if len(features) != len(rows):
        raise ValueError("Feature population mismatch")
    train = np.array([r["split"] == "train" for r in rows])
    validation = np.array([r["split"] == "validation" for r in rows])
    test = np.array([r["split"] == "test" for r in rows])
    if not all(mask.sum() for mask in [train, validation, test]):
        raise ValueError("Every split needs examples")
    tools = pending_tool_metadata(manifest, rows)
    categories = [(r["session_id"].split("__")[0], r["source_model"], tools[r["session_id"]]) for r in rows]
    cheap = cheap_design(np.array([r["cheap"] for r in rows], dtype=float), categories, train, cheap_interactions)
    y = np.array([int(r["next_gap_s"] <= horizon) for r in rows], dtype=float)
    result = dict(schema="hybrid-predictive-pilot-result-v1", qualification="not_established",
                  counts=dict(Counter(r["split"] for r in rows)), examples_sha256=manifest["examples_sha256"],
                  limitations=manifest["limitations"], models={}, cheap_tool_metadata="Already-issued tool name, train-only categorical vocabulary; no future tool result")
    result.update(horizon_s=horizon, interval_family_size=family_size,
                  interval_confidence=1 - .05 / family_size,
                  analysis="Original primary horizon" if horizon == 1.0 and family_size == 1 else "Exploratory deadline sensitivity; original gate unchanged")
    result["cheap_design"] = "Numeric pair products, numeric/category interactions and joint categorical state" if cheap_interactions else "Log numeric main effects and squares; categorical main effects"
    predictions = {}
    for name, feature in [("cheap", None), ("scalar", "scalar"), ("distribution", "distribution"),
                          ("representation", "representation")]:
        x = cheap if feature is None else np.column_stack([cheap, np.array([features[r["session_id"]][feature] for r in rows])])
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite features")
        mean, std = x[train].mean(0), x[train].std(0)
        x = np.column_stack([np.ones(len(x)), (x - mean) / np.maximum(std, 1e-6)])
        candidates = []
        for ridge in [.01, .1, 1., 10., 100.]:
            weights = fit_logistic(x[train], y[train], ridge)
            p = np.clip(probability(x[validation], weights), 1e-12, 1 - 1e-12)
            loss = -np.mean(y[validation] * np.log(p) + (1 - y[validation]) * np.log(1 - p))
            candidates.append((loss, ridge, weights))
        _, ridge, weights = min(candidates, key=lambda value: value[:2])
        p = probability(x[test], weights)
        predictions[name] = p
        result["models"][name] = dict(ridge=ridge, test_brier=float(np.mean((p - y[test]) ** 2)),
                                           validation_logloss=float(min(candidates, key=lambda value: value[:2])[0]),
                                           hypothetical_retention=retained_value(p, y[test], [len(r["tokens"]) for r, keep in zip(rows, test) if keep]))
    groups = [r["group"] for r, keep in zip(rows, test) if keep]
    for name in ["scalar", "distribution", "representation"]:
        improvement = (predictions["cheap"] - y[test]) ** 2 - (predictions[name] - y[test]) ** 2
        result["models"][name]["task_paired_brier_improvement"] = paired_interval(improvement, groups, alpha=.05 / family_size)
        result["models"][name]["hypothetical_retention_interval"] = retention_interval(
            predictions["cheap"], predictions[name], y[test],
            [len(r["tokens"]) for r, keep in zip(rows, test) if keep], groups, alpha=.05 / family_size)
    result["test_positive_fraction"] = float(y[test].mean())
    result["capture_seconds"] = sum(r["capture_s"] for r in features.values())
    result["capture_timing_scope"] = (
        "Forward, hooks and CPU score summaries; not incremental serving overhead"
        if all("forward_and_hook_s" in r for r in features.values()) else
        "Legacy timer: forward and hooks only; final CPU summaries excluded. Not incremental serving overhead"
    )
    result["capture_peak_allocated_bytes"] = max(r["peak_allocated_bytes"] for r in features.values())
    result["evaluation_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    with output.with_suffix(".predictions.jsonl").open("x") as stream:
        for i, row in enumerate(r for r, keep in zip(rows, test) if keep):
            stream.write(json.dumps(dict(session_id=row["session_id"], group=row["group"], target=int(y[test][i]),
                                         predictions={name: float(values[i]) for name, values in predictions.items()})) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--family-size", type=int, default=1)
    parser.add_argument("--cheap-interactions", action="store_true")
    args = parser.parse_args()
    evaluate(args.inputs, args.capture, args.output, horizon=args.horizon, family_size=args.family_size,
             cheap_interactions=args.cheap_interactions)


if __name__ == "__main__":
    main()

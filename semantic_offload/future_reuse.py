# SPDX-License-Identifier: Apache-2.0
"""Leakage-safe future-reuse labels for offline semantic-policy fitting.

The serving preservation event for request *t* describes the decision made for
that request; its hit/miss outcome is not a feature label for that same
decision. This module only labels keys from later authoritative request
windows and refuses to cross a caller-supplied train boundary. The resulting
labels can be fed to ``SemanticOffloadingManager.update_future_reuse_labels``
after an offline held-out gate. Train/test membership is carried by a separate,
sealed split metadata artifact because the serving v1 event schema has no
train/test split field.
"""

import csv
import hashlib
import hmac
import io
import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from benchmarks.csv_utils import CSV_FIELD_SIZE_LIMIT  # noqa: F401
from semantic_offload.decision_trace import validate_decision_feature_records

KeyT = TypeVar("KeyT")
_EXTERNAL_ID_RE = re.compile(r"^sc\|([^|]+)\|([^|]+)\|([^|]+)\|sample-(\d+)$")
_INTERNAL_ID_RE = re.compile(
    r"^cmpl-(sc\|[^|]+\|[^|]+\|[^|]+\|sample-\d+)-\d+-([0-9a-f]{8})$"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SEAL_KEY = secrets.token_bytes(32)


@dataclass(frozen=True, slots=True)
class FutureReuseLabelBatch:
    """Immutable, provenance-bearing labels from one validated train window."""

    _entries: tuple[tuple[object, bool], ...] = field(init=False, repr=False)
    _train_window_id: str = field(init=False, repr=False)
    _test_window_id: str = field(init=False, repr=False)
    _train_request_ids: tuple[str, ...] = field(init=False, repr=False)
    _test_request_ids: tuple[str, ...] = field(init=False, repr=False)
    _source_id: str = field(init=False, repr=False)
    _config_id: str = field(init=False, repr=False)
    _future_window: int = field(init=False, repr=False)
    _split_metadata_sha256: str = field(init=False, repr=False)
    _seal: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raise TypeError("future-reuse batches can only be built from validated events")

    @property
    def entries(self) -> tuple[tuple[object, bool], ...]:
        return self._entries

    @property
    def train_window_id(self) -> str:
        return self._train_window_id

    @property
    def test_window_id(self) -> str:
        return self._test_window_id

    @property
    def train_request_ids(self) -> tuple[str, ...]:
        return self._train_request_ids

    @property
    def test_request_ids(self) -> tuple[str, ...]:
        return self._test_request_ids

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def config_id(self) -> str:
        return self._config_id

    @property
    def future_window(self) -> int:
        return self._future_window

    @property
    def split_metadata_sha256(self) -> str:
        return self._split_metadata_sha256

    def _validate_seal(self) -> None:
        """Validate contents and the private seal before a consumer mutates state."""
        if not isinstance(self._seal, str):
            raise TypeError("future-reuse batch has no valid seal")
        if (
            not isinstance(self.train_window_id, str)
            or not isinstance(self.test_window_id, str)
            or not self.train_window_id
            or not self.test_window_id
            or self.train_window_id == self.test_window_id
        ):
            raise ValueError("future-reuse windows must be distinct and non-empty")
        if (
            not isinstance(self.source_id, str)
            or not isinstance(self.config_id, str)
            or not self.source_id
            or not self.config_id
        ):
            raise ValueError("future-reuse source/config IDs must be non-empty strings")
        if not self.train_request_ids or not self.test_request_ids:
            raise ValueError("future-reuse batch requires train and test requests")
        if len(set(self.train_request_ids)) != len(self.train_request_ids):
            raise ValueError("duplicate train request in future-reuse batch")
        if len(set(self.test_request_ids)) != len(self.test_request_ids):
            raise ValueError("duplicate test request in future-reuse batch")
        if set(self.train_request_ids) & set(self.test_request_ids):
            raise ValueError("train/test request provenance overlaps")
        seen = set()
        serialized_keys = []
        for key, label in self.entries:
            if not isinstance(label, bool):
                raise TypeError("future-reuse labels must be boolean")
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError(
                    "future-reuse keys must be hashable OffloadKeys"
                ) from exc
            if key in seen:
                raise ValueError("duplicate key in future-reuse label batch")
            seen.add(key)
            serialized_keys.append(json.dumps(_canonical_entry(key), sort_keys=True))
        if serialized_keys != sorted(serialized_keys):
            raise ValueError("future-reuse entries are not in canonical order")
        if not isinstance(self.future_window, int) or self.future_window <= 0:
            raise ValueError("future window must be positive")
        if (
            not isinstance(self.split_metadata_sha256, str)
            or _SHA256_RE.fullmatch(self.split_metadata_sha256) is None
        ):
            raise ValueError("future-reuse split metadata seal identity is invalid")

        payload = _sealed_payload(self)
        expected = hmac.new(
            _SEAL_KEY,
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(self._seal, expected):
            raise ValueError("future-reuse batch provenance seal mismatch")

    @property
    def application_id(self) -> str:
        self._validate_seal()
        return self._seal

    @property
    def window_identity(self) -> tuple[str, str]:
        return self.train_window_id, self.test_window_id


def _token(value: object) -> str | None:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None


def _empty(value: object) -> bool:
    return value is None or str(value).strip() == ""


def _canonical_entry(key: object) -> dict[str, str]:
    """Serialize only keys with an unambiguous, canonical representation."""
    if isinstance(key, bytes):
        return {"bytes": key.hex()}
    token = _token(key)
    if token is None:
        raise TypeError("future-reuse entries require canonical JSON keys")
    return {"json": token}


def _sealed_payload(batch: FutureReuseLabelBatch) -> dict[str, object]:
    return {
        "source_id": batch.source_id,
        "config_id": batch.config_id,
        "train_window_id": batch.train_window_id,
        "test_window_id": batch.test_window_id,
        "train_request_ids": batch.train_request_ids,
        "test_request_ids": batch.test_request_ids,
        "future_window": batch.future_window,
        "split_metadata_sha256": batch.split_metadata_sha256,
        "entries": [(_canonical_entry(key), label) for key, label in batch.entries],
    }


_MANIFEST_DIGEST_FIELDS = (
    "artifact_manifest_sha256",
    "manifest_sha256",
    "sha256",
)
_ROW_DIGEST_FIELDS = ("row_sha256", "row_digest", "artifact_row_sha256")
_CONFIG_DIGEST_FIELDS = ("config_digest", "config_sha256", "config_id")
_CSV_DIGEST_FIELDS = ("csv_sha256", "results_csv_sha256")
_SPLIT_MANIFEST_DIGEST_FIELDS = (
    "artifact_manifest_sha256",
    "manifest_sha256",
    "variant_manifest_sha256",
)
_SPLIT_CSV_DIGEST_FIELDS = ("csv_sha256", "results_csv_sha256")
_SPLIT_ROW_DIGEST_FIELDS = ("row_sha256", "row_digest", "artifact_row_sha256")
_SPLIT_SELF_DIGEST_FIELDS = ("split_metadata_sha256", "metadata_sha256")
_REQUIRED_VARIANT_MANIFEST_FIELDS = (
    "variant",
    "command",
    "policies",
    "workloads",
    "request-rates",
    "seeds",
    "extra-config",
    "semantic_revision_start",
    "semantic_revision_end",
    "semantic_state_start",
    "semantic_state_end",
    "vllm_revision_start",
    "vllm_revision_end",
    "vllm_state_start",
    "vllm_state_end",
)
_SOURCE_MANIFEST_FIELDS = (
    "semantic_revision_start",
    "semantic_revision_end",
    "semantic_state_start",
    "semantic_state_end",
    "vllm_revision_start",
    "vllm_revision_end",
    "vllm_state_start",
    "vllm_state_end",
)


def _canonical_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "future-reuse artifact metadata is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _declared_digest(
    metadata: Mapping[str, object], fields: Sequence[str], *, label: str
) -> str | None:
    values = [metadata[field] for field in fields if metadata.get(field) is not None]
    if not values:
        return None
    if any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in values
    ):
        raise ValueError(f"future-reuse {label} declaration is not lowercase SHA-256")
    if len(set(values)) != 1:
        raise ValueError(f"future-reuse {label} declarations conflict")
    return values[0]


def _read_split_metadata(
    split_metadata: Mapping[str, object] | str | Path,
) -> tuple[Mapping[str, object], str]:
    """Read split metadata and derive an immutable identity for its source."""
    if isinstance(split_metadata, (str, Path)):
        path = Path(split_metadata)
        raw = path.read_bytes()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("future-reuse split metadata is not valid JSON") from exc
        digest = hashlib.sha256(raw).hexdigest()
    elif isinstance(split_metadata, Mapping):
        parsed = split_metadata
        digest = _canonical_digest(
            {
                key: value
                for key, value in split_metadata.items()
                if key not in _SPLIT_SELF_DIGEST_FIELDS
            }
        )
    else:
        raise TypeError(
            "future-reuse split metadata must be a mapping or an artifact path"
        )
    if not isinstance(parsed, Mapping):
        raise TypeError("future-reuse split metadata must be an object")
    declared = _declared_digest(
        parsed, _SPLIT_SELF_DIGEST_FIELDS, label="split metadata"
    )
    if declared is not None and declared != digest:
        raise ValueError("future-reuse split metadata digest does not match its bytes")
    return parsed, digest


def _validated_manifest(
    manifest: Mapping[str, object] | str | Path,
) -> tuple[Mapping[str, object], str]:
    """Load the manifest and derive its identity from the artifact itself."""
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        raw = manifest_path.read_bytes()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = _parse_variant_manifest(raw)
        if not isinstance(parsed, Mapping):
            raise TypeError("future-reuse artifact manifest must be an object")
        actual_digest = hashlib.sha256(raw).hexdigest()
    elif isinstance(manifest, Mapping):
        parsed = manifest
        digest_material = {
            key: value
            for key, value in manifest.items()
            if key not in _MANIFEST_DIGEST_FIELDS
        }
        actual_digest = _canonical_digest(digest_material)
    else:
        raise TypeError("future-reuse artifact manifest must be a mapping or path")
    declared = _declared_digest(parsed, _MANIFEST_DIGEST_FIELDS, label="manifest")
    if declared is not None and declared != actual_digest:
        raise ValueError("future-reuse manifest digest does not match its contents")
    return parsed, actual_digest


def _parse_variant_manifest(raw: bytes) -> dict[str, object]:
    """Parse the runner's byte-preserved ``variant_manifest.txt`` format."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("future-reuse variant manifest is not UTF-8") from exc
    values: dict[str, object] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or "=" not in line:
            raise ValueError(
                f"future-reuse variant manifest line {line_number} is malformed"
            )
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(
                f"future-reuse variant manifest line {line_number} has a duplicate key"
            )
        values[key] = value
    missing = [key for key in _REQUIRED_VARIANT_MANIFEST_FIELDS if key not in values]
    if missing:
        raise ValueError(
            "future-reuse variant manifest is missing canonical fields: "
            + ", ".join(missing)
        )
    for key in _SOURCE_MANIFEST_FIELDS:
        if not isinstance(values[key], str) or not values[key]:
            raise ValueError(f"future-reuse variant manifest has empty {key}")
    for prefix in (
        "semantic_revision",
        "semantic_state",
        "vllm_revision",
        "vllm_state",
    ):
        if values[f"{prefix}_start"] != values[f"{prefix}_end"]:
            raise ValueError(
                f"future-reuse variant manifest has source drift in {prefix}"
            )
    try:
        config = json.loads(str(values["extra-config"]))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "future-reuse variant manifest extra-config is not valid JSON"
        ) from exc
    if not isinstance(config, Mapping):
        raise TypeError("future-reuse variant manifest extra-config must be an object")
    values["config"] = config
    return values


def _validate_variant_manifest_for_artifact(
    manifest: Mapping[str, object], *, manifest_path: Path, variant: str | None
) -> tuple[str, str]:
    """Validate selector/source fields before any CSV row is trusted."""
    missing = [key for key in _REQUIRED_VARIANT_MANIFEST_FIELDS if key not in manifest]
    if missing:
        raise ValueError(
            "future-reuse variant manifest is missing canonical fields: "
            + ", ".join(missing)
        )
    manifest_variant = manifest.get("variant")
    if not isinstance(manifest_variant, str) or not manifest_variant:
        raise ValueError("future-reuse variant manifest has no variant")
    if variant is not None and manifest_variant != variant:
        raise ValueError("future-reuse variant selector does not match manifest")
    if manifest_path.parent.name != manifest_variant:
        raise ValueError("future-reuse variant manifest path is not bound to variant")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("future-reuse variant manifest has no parsed extra-config")
    for prefix in (
        "semantic_revision",
        "semantic_state",
        "vllm_revision",
        "vllm_state",
    ):
        start = manifest.get(f"{prefix}_start")
        end = manifest.get(f"{prefix}_end")
        if (
            not isinstance(start, str)
            or not isinstance(end, str)
            or not start
            or not end
        ):
            raise ValueError(f"future-reuse variant manifest has invalid {prefix}")
        if start != end:
            raise ValueError(
                f"future-reuse variant manifest has source drift in {prefix}"
            )
    config_digest = _canonical_digest(config)
    declared = _declared_digest(
        manifest, _CONFIG_DIGEST_FIELDS, label="manifest config"
    )
    if declared is not None and declared != config_digest:
        raise ValueError("future-reuse variant manifest config digest conflicts")
    source_token = _canonical_digest(
        {key: manifest[key] for key in _SOURCE_MANIFEST_FIELDS}
    )
    return config_digest, source_token


def _read_results_csv(path: Path) -> tuple[str, list[dict[str, str]]]:
    """Read the exact canonical CSV bytes and reject malformed rows."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("future-reuse results CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or any(
        field is None or field == "" for field in reader.fieldnames
    ):
        raise ValueError("future-reuse results CSV has no canonical header")
    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"future-reuse results CSV row {row_number} is malformed")
        rows.append(dict(row))
    if not rows:
        raise ValueError("future-reuse results CSV has no rows")
    return hashlib.sha256(raw).hexdigest(), rows


def _selector_matches(row: Mapping[str, str], selectors: Mapping[str, object]) -> bool:
    for field_name, expected in selectors.items():
        if expected is None:
            continue
        actual = row.get(field_name)
        if field_name in {"seed", "sample_index", "request_rate"}:
            if str(actual) != str(expected):
                return False
        elif actual != expected:
            return False
    return True


def _canonical_artifact_row(
    row: Mapping[str, str],
    *,
    manifest: Mapping[str, object],
    artifact_root: Path,
    manifest_digest: str,
    csv_digest: str,
    source_token: str,
) -> dict[str, object]:
    """Add only file-derived provenance to a CSV row before building labels."""
    result: dict[str, object] = dict(row)
    required = (
        "policy",
        "seed",
        "workload",
        "run_id",
        "cell_id",
        "sample_index",
        "request_id_prefix",
        "request_ids_json",
        "preservation_request_ids_json",
        "preservation_events_json",
        "preservation_event_count",
    )
    config = manifest["config"]
    trace_enabled = config.get("decision_feature_trace") is True and str(
        result.get("policy", "")
    ).startswith("semantic-")
    if trace_enabled:
        required += (
            "decision_feature_trace_json",
            "decision_feature_trace_count",
            "decision_feature_trace_join_status",
        )
    missing = [field for field in required if _empty(result.get(field))]
    if missing:
        raise ValueError(
            "future-reuse canonical CSV row is missing fields: " + ", ".join(missing)
        )
    if _empty(result.get("error")) is False:
        raise ValueError("future-reuse canonical CSV row contains an error")
    if result.get("preservation_join_status") != "complete":
        raise ValueError("future-reuse canonical CSV row is not a complete join")
    if trace_enabled:
        if result.get("decision_feature_trace_join_status") != "complete":
            raise ValueError("future-reuse CSV decision feature trace is not complete")
        try:
            feature_records = json.loads(str(result["decision_feature_trace_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "future-reuse CSV decision feature trace is not valid JSON"
            ) from exc
        if not isinstance(feature_records, list):
            raise ValueError("future-reuse CSV decision feature trace is not an array")
        try:
            request_ids = json.loads(str(result["preservation_request_ids_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "future-reuse CSV preservation request IDs are invalid"
            ) from exc
        if not isinstance(request_ids, list) or not all(
            isinstance(request_id, str) for request_id in request_ids
        ):
            raise ValueError("future-reuse CSV preservation request IDs are invalid")
        validated_features = validate_decision_feature_records(
            feature_records,
            expected_request_ids=request_ids,
            run_id=result["run_id"],
            cell_id=result["cell_id"],
        )
        declared_count = result.get("decision_feature_trace_count")
        if str(declared_count) != str(len(validated_features)):
            raise ValueError("future-reuse CSV decision feature count conflicts")
    elif (
        result.get("decision_feature_trace_json") not in (None, "", "[]")
        or result.get("decision_feature_trace_count") not in (None, "", 0, "0")
        or result.get("decision_feature_trace_join_status")
        not in (None, "", "not_requested", "disabled")
    ):
        raise ValueError("future-reuse CSV contains an unbound decision feature trace")
    manifest_policies = {
        value for value in str(manifest["policies"]).split(",") if value
    }
    if result["policy"] not in manifest_policies:
        raise ValueError("future-reuse CSV policy is not bound to manifest")
    manifest_workloads = {
        value for value in str(manifest["workloads"]).split(",") if value
    }
    if result["workload"] not in manifest_workloads:
        raise ValueError("future-reuse CSV workload is not bound to manifest")
    manifest_seeds = {value for value in str(manifest["seeds"]).split(",") if value}
    if str(result["seed"]) not in manifest_seeds:
        raise ValueError("future-reuse CSV seed is not bound to manifest")
    manifest_rates = {
        value for value in str(manifest["request-rates"]).split(",") if value
    }
    if (
        result.get("request_rate") not in {None, ""}
        and str(result["request_rate"]) not in manifest_rates
    ):
        raise ValueError("future-reuse CSV request rate is not bound to manifest")
    declared_row = _declared_digest(result, _ROW_DIGEST_FIELDS, label="CSV row")
    actual_row = _canonical_digest(
        {key: value for key, value in result.items() if key not in _ROW_DIGEST_FIELDS}
    )
    if declared_row is not None and declared_row != actual_row:
        raise ValueError("future-reuse CSV row digest conflicts with its contents")
    declared_root = result.get("artifact_root") or result.get("root")
    if declared_root is not None and str(Path(str(declared_root)).resolve()) != str(
        artifact_root
    ):
        raise ValueError("future-reuse CSV artifact root conflicts with caller")
    for key in _SOURCE_MANIFEST_FIELDS:
        if key in result and result[key] != manifest[key]:
            raise ValueError(f"future-reuse CSV {key} conflicts with manifest")
    declared_source_manifest = result.get("source_manifest_sha256")
    if (
        declared_source_manifest is not None
        and declared_source_manifest != manifest_digest
    ):
        raise ValueError("future-reuse CSV source manifest digest conflicts")
    declared_source_identity = result.get("source_identity")
    if (
        declared_source_identity is not None
        and declared_source_identity != source_token
    ):
        raise ValueError("future-reuse CSV source identity conflicts")
    expected_config_digest = _canonical_digest(config)
    declared_config = _declared_digest(
        result, _CONFIG_DIGEST_FIELDS, label="CSV config"
    )
    if declared_config is not None and declared_config != expected_config_digest:
        raise ValueError("future-reuse CSV config digest conflicts with manifest")
    declared_csv = _declared_digest(result, _CSV_DIGEST_FIELDS, label="CSV")
    if declared_csv is not None and declared_csv != csv_digest:
        raise ValueError("future-reuse CSV digest conflicts with its contents")
    declared_manifest = _declared_digest(
        result, _MANIFEST_DIGEST_FIELDS, label="CSV manifest"
    )
    if declared_manifest is not None and declared_manifest != manifest_digest:
        raise ValueError("future-reuse CSV manifest digest conflicts with its contents")
    result.update(
        {
            "artifact_root": str(artifact_root),
            "manifest_sha256": manifest_digest,
            "csv_sha256": csv_digest,
            "source_manifest_sha256": manifest_digest,
            "source_identity": source_token,
            "config": manifest["config"],
            "config_digest": expected_config_digest,
            "_canonical_row_sha256": actual_row,
        }
    )
    return result


@dataclass(frozen=True, slots=True)
class FutureReuseArtifact:
    """A file-backed, validated future-reuse input selected from one CSV row."""

    root: Path
    manifest_path: Path
    csv_path: Path
    manifest: Mapping[str, object]
    row: Mapping[str, object]
    manifest_sha256: str
    csv_sha256: str
    source_identity: str


def load_future_reuse_artifact(
    artifact_root: str | Path,
    *,
    variant: str | None = None,
    policy: str | None = None,
    seed: int | str | None = None,
    workload: str | None = None,
    sub_workload: str | None = None,
    request_rate: str | float | None = None,
    sample_index: int | str | None = None,
    manifest_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> FutureReuseArtifact:
    """Load and bind the canonical variant manifest and results CSV.

    Callers select a row; they cannot supply a replacement manifest, CSV row,
    chronology, or source/config identity.  Ambiguous selectors fail closed.
    """
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ValueError("future-reuse artifact root must be absolute")
    root = root.resolve()
    if variant is not None:
        variant_dir = (root / variant).resolve()
    elif manifest_path is not None:
        variant_dir = Path(manifest_path).resolve().parent
    elif csv_path is not None:
        variant_dir = Path(csv_path).resolve().parent
    elif (root / "variant_manifest.txt").is_file():
        variant_dir = root
    else:
        candidates = sorted(root.glob("*/variant_manifest.txt"))
        if len(candidates) != 1:
            raise ValueError(
                "future-reuse artifact root must identify exactly one variant"
            )
        variant_dir = candidates[0].parent.resolve()
    if root != variant_dir and root not in variant_dir.parents:
        raise ValueError("future-reuse variant directory escapes artifact root")
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else variant_dir / "variant_manifest.txt"
    )
    results_file = (
        Path(csv_path) if csv_path is not None else variant_dir / "results.csv"
    )
    if not manifest_file.is_absolute():
        manifest_file = (root / manifest_file).resolve()
    else:
        manifest_file = manifest_file.resolve()
    if not results_file.is_absolute():
        results_file = (root / results_file).resolve()
    else:
        results_file = results_file.resolve()
    if (
        manifest_file.parent != variant_dir.resolve()
        or results_file.parent != variant_dir.resolve()
    ):
        raise ValueError(
            "future-reuse manifest and CSV must share the variant directory"
        )
    manifest, manifest_digest = _validated_manifest(manifest_file)
    manifest_variant = manifest.get("variant")
    selected_variant = variant or (
        manifest_variant if isinstance(manifest_variant, str) else None
    )
    config_digest, source_token = _validate_variant_manifest_for_artifact(
        manifest, manifest_path=manifest_file, variant=selected_variant
    )
    csv_digest, rows = _read_results_csv(results_file)
    selectors = {
        "policy": policy,
        "seed": seed,
        "workload": workload,
        "sub_workload": sub_workload,
        "request_rate": request_rate,
        "sample_index": sample_index,
    }
    selected = [row for row in rows if _selector_matches(row, selectors)]
    if len(selected) != 1:
        raise ValueError(
            "future-reuse CSV selector must identify exactly one row; "
            f"matched {len(selected)}"
        )
    row = _canonical_artifact_row(
        selected[0],
        manifest=manifest,
        artifact_root=root,
        manifest_digest=manifest_digest,
        csv_digest=csv_digest,
        source_token=source_token,
    )
    if row["config_digest"] != config_digest:
        raise ValueError("future-reuse CSV config is not bound to manifest")
    return FutureReuseArtifact(
        root=root,
        manifest_path=manifest_file,
        csv_path=results_file,
        manifest=manifest,
        row=row,
        manifest_sha256=manifest_digest,
        csv_sha256=csv_digest,
        source_identity=source_token,
    )


def _archived_row_payload(
    artifact_row: Mapping[str, object],
) -> tuple[tuple[str, ...], list[Mapping[str, object]]]:
    """Read chronology and events only from the canonical archived row fields."""
    if "ordered_request_ids" in artifact_row:
        raise ValueError(
            "future-reuse chronology cannot use ordered_request_ids "
            "or sequence ordinals"
        )

    def read_json_array(field: str):
        value = artifact_row.get(field)
        if not isinstance(value, str):
            raise TypeError(f"future-reuse archived row needs {field}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"future-reuse archived row {field} is not valid JSON"
            ) from exc
        if not isinstance(parsed, list):
            raise TypeError(f"future-reuse archived row {field} is not a JSON array")
        return parsed

    request_ids = read_json_array("preservation_request_ids_json")
    if not all(isinstance(request_id, str) for request_id in request_ids):
        raise ValueError("future-reuse archived request IDs are not a string array")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("future-reuse archived request IDs are not unique")
    event_rows = read_json_array("preservation_events_json")
    if not all(isinstance(event, Mapping) for event in event_rows):
        raise ValueError("future-reuse archived events are not an object array")
    if any("sequence_ordinal" in event for event in event_rows):
        raise ValueError(
            "future-reuse chronology cannot use ordered_request_ids "
            "or sequence ordinals"
        )
    event_ids = [event.get("external_request_id") for event in event_rows]
    if not all(isinstance(request_id, str) for request_id in event_ids):
        raise ValueError("future-reuse archived events lack request IDs")
    if event_ids != request_ids:
        raise ValueError(
            "future-reuse archived events do not match preservation request IDs"
        )
    event_count = artifact_row.get("preservation_event_count")
    if event_count is not None:
        if isinstance(event_count, bool):
            raise ValueError("future-reuse archived event count is invalid")
        try:
            parsed_count = int(event_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("future-reuse archived event count is invalid") from exc
        if parsed_count != len(event_rows):
            raise ValueError("future-reuse archived event count does not match events")
    legacy_ids = artifact_row.get("request_ids_json")
    if legacy_ids is not None:
        if not isinstance(legacy_ids, str):
            raise ValueError("future-reuse legacy request IDs are not JSON")
        try:
            if json.loads(legacy_ids) != request_ids:
                raise ValueError(
                    "future-reuse legacy request IDs conflict with canonical "
                    "request IDs"
                )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "future-reuse legacy request IDs are not valid JSON"
            ) from exc
    return tuple(request_ids), list(event_rows)


def _validated_split_metadata(
    split_metadata: Mapping[str, object] | str | Path,
    *,
    artifact_manifest: Mapping[str, object] | str | Path,
    artifact_row: Mapping[str, object],
    request_ids: Sequence[str],
    train_end: int,
    train_window_id: str,
    test_window_id: str,
) -> tuple[Mapping[str, object], str, dict[str, str]]:
    """Validate the independent split binding and derive request membership."""
    metadata, metadata_digest = _read_split_metadata(split_metadata)
    if metadata.get("schema") != "future-reuse-split-v1":
        raise ValueError("future-reuse split metadata has an unsupported schema")

    _, manifest_digest = _validated_manifest(artifact_manifest)
    declared_manifest = _declared_digest(
        metadata, _SPLIT_MANIFEST_DIGEST_FIELDS, label="split manifest"
    )
    if declared_manifest is None or declared_manifest != manifest_digest:
        raise ValueError("future-reuse split metadata is not bound to variant manifest")

    csv_digest = _declared_digest(artifact_row, _CSV_DIGEST_FIELDS, label="CSV")
    if csv_digest is None:
        raise ValueError("future-reuse artifact row has no canonical CSV digest")
    declared_csv = _declared_digest(
        metadata, _SPLIT_CSV_DIGEST_FIELDS, label="split CSV"
    )
    if declared_csv is None or declared_csv != csv_digest:
        raise ValueError("future-reuse split metadata is not bound to canonical CSV")

    canonical_row_digest = artifact_row.get("_canonical_row_sha256")
    if canonical_row_digest is None:
        canonical_row_digest = _canonical_digest(
            {
                key: value
                for key, value in artifact_row.items()
                if key not in _ROW_DIGEST_FIELDS
            }
        )
    if (
        not isinstance(canonical_row_digest, str)
        or _SHA256_RE.fullmatch(canonical_row_digest) is None
    ):
        raise ValueError("future-reuse artifact row has no canonical row digest")
    declared_row = _declared_digest(
        metadata, _SPLIT_ROW_DIGEST_FIELDS, label="split row"
    )
    if declared_row is None or declared_row != canonical_row_digest:
        raise ValueError(
            "future-reuse split metadata row digest is not bound to exact CSV row"
        )

    row_identity = metadata.get("row_identity")
    if row_identity is not None:
        if isinstance(row_identity, str):
            if row_identity != canonical_row_digest:
                raise ValueError(
                    "future-reuse split row identity conflicts with CSV row"
                )
        elif isinstance(row_identity, Mapping):
            for key, expected in row_identity.items():
                if key in _ROW_DIGEST_FIELDS:
                    if expected != canonical_row_digest:
                        raise ValueError(
                            "future-reuse split row identity conflicts with CSV row"
                        )
                elif key in artifact_row and str(artifact_row[key]) != str(expected):
                    raise ValueError(
                        "future-reuse split row identity conflicts with CSV row"
                    )
        else:
            raise TypeError(
                "future-reuse split row_identity must be a string or object"
            )

    persisted_ids = metadata.get("ordered_request_ids")
    if not isinstance(persisted_ids, list) or not all(
        isinstance(request_id, str) for request_id in persisted_ids
    ):
        raise ValueError("future-reuse split metadata needs ordered request IDs")
    if tuple(persisted_ids) != tuple(request_ids):
        raise ValueError(
            "future-reuse split ordered request IDs do not match canonical row"
        )
    if len(set(persisted_ids)) != len(persisted_ids):
        raise ValueError("future-reuse split ordered request IDs are not unique")

    metadata_train_end = metadata.get("train_end")
    if (
        isinstance(metadata_train_end, bool)
        or not isinstance(metadata_train_end, int)
        or metadata_train_end != train_end
    ):
        raise ValueError("future-reuse split train_end does not match evaluator")
    metadata_train = metadata.get("train_window_id")
    metadata_test = metadata.get("test_window_id")
    if metadata_train != train_window_id or metadata_test != test_window_id:
        raise ValueError("future-reuse split window IDs do not match evaluator")
    if (
        not isinstance(metadata_train, str)
        or not isinstance(metadata_test, str)
        or not metadata_train
        or not metadata_test
        or metadata_train == metadata_test
    ):
        raise ValueError("future-reuse split window IDs must be distinct and non-empty")

    expected_windows = [
        train_window_id if index < train_end else test_window_id
        for index in range(len(request_ids))
    ]
    declared_windows = metadata.get("window_ids")
    if declared_windows is not None and declared_windows != expected_windows:
        raise ValueError("future-reuse split window membership conflicts with boundary")
    membership = dict(zip(persisted_ids, expected_windows, strict=True))
    return metadata, metadata_digest, membership


def _validated_artifact_provenance(
    events: Sequence[Mapping[str, object]],
    *,
    artifact_manifest: Mapping[str, object] | str | Path,
    artifact_root: str | Path,
    artifact_row: Mapping[str, object],
    config_digest: str | None,
    source_id: str | None,
    config_id: str | None,
    chronological_request_ids: Sequence[str],
    split_metadata_sha256: str,
) -> tuple[str, str, str | None]:
    manifest, manifest_digest = _validated_manifest(artifact_manifest)
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ValueError("future-reuse artifact root must be absolute")
    root_id = str(root.resolve())
    for metadata, label in ((manifest, "manifest"), (artifact_row, "row")):
        for key in ("artifact_root", "root"):
            declared_root = metadata.get(key)
            declared_root_id = (
                str(Path(str(declared_root)).resolve())
                if declared_root is not None
                else None
            )
            if declared_root is not None and declared_root_id != root_id:
                raise ValueError(f"future-reuse artifact root does not match {label}")

    canonical_row_digest = artifact_row.get("_canonical_row_sha256")
    if canonical_row_digest is not None:
        if (
            not isinstance(canonical_row_digest, str)
            or _SHA256_RE.fullmatch(canonical_row_digest) is None
        ):
            raise ValueError("future-reuse canonical row digest is invalid")
        row_digest = canonical_row_digest
    else:
        row_digest = _canonical_digest(
            {
                key: value
                for key, value in artifact_row.items()
                if key not in _ROW_DIGEST_FIELDS
            }
        )
    declared_row_digest = _declared_digest(
        artifact_row, _ROW_DIGEST_FIELDS, label="row"
    )
    if declared_row_digest is not None and declared_row_digest != row_digest:
        raise ValueError("future-reuse row digest does not match its contents")
    declared_manifest_digest = artifact_row.get("manifest_sha256")
    if (
        declared_manifest_digest is not None
        and declared_manifest_digest != manifest_digest
    ):
        raise ValueError("future-reuse row is bound to a different manifest")
    csv_digest = _declared_digest(artifact_row, _CSV_DIGEST_FIELDS, label="CSV")

    run_cells = {(event.get("run_id"), event.get("cell_id")) for event in events}
    if len(run_cells) != 1:
        raise ValueError("future-reuse events must have one run/cell identity")
    run_id, cell_id = next(iter(run_cells))
    for metadata, label in ((artifact_row, "row"), (manifest, "manifest")):
        for key, expected in (("run_id", run_id), ("cell_id", cell_id)):
            declared = metadata.get(key)
            if declared is not None and declared != expected:
                raise ValueError(f"future-reuse {label} {key} does not match events")

    configs = [
        metadata["config"]
        for metadata in (manifest, artifact_row)
        if "config" in metadata
    ]
    if not configs:
        raise ValueError("future-reuse artifacts need an actual config object")
    config_token = json.dumps(
        configs[0], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if any(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        != config_token
        for config in configs[1:]
    ):
        raise ValueError("future-reuse manifest and row configs conflict")
    actual_config_digest = _canonical_digest(configs[0])
    for metadata, label in ((manifest, "manifest"), (artifact_row, "row")):
        declared = _declared_digest(
            metadata, _CONFIG_DIGEST_FIELDS, label=f"{label} config"
        )
        if declared is not None and declared != actual_config_digest:
            raise ValueError(
                f"future-reuse {label} config digest does not match its contents"
            )
    if config_digest is not None and config_digest != actual_config_digest:
        raise ValueError("future-reuse config digest is not derived from artifacts")
    if config_id is not None and config_id != actual_config_digest:
        raise ValueError("future-reuse config_id is not the validated config digest")

    derived_source_id = _canonical_digest(
        {
            "artifact_manifest_sha256": manifest_digest,
            "csv_sha256": csv_digest,
            "artifact_root": root_id,
            "artifact_row_sha256": row_digest,
            "run_id": run_id,
            "cell_id": cell_id,
            "config_digest": actual_config_digest,
            "chronological_request_ids": tuple(chronological_request_ids),
            "split_metadata_sha256": split_metadata_sha256,
        }
    )
    if source_id is not None and source_id != derived_source_id:
        raise ValueError(
            "future-reuse source_id is not derived from artifact provenance"
        )
    row_prefix = artifact_row.get("request_id_prefix")
    if row_prefix is not None and not isinstance(row_prefix, str):
        raise ValueError("future-reuse artifact row request_id_prefix is not a string")
    return derived_source_id, actual_config_digest, row_prefix


def _valid_event_identity(event: Mapping[str, object]) -> bool:
    if (
        event.get("schema") != "semantic-preservation-event-v1"
        or event.get("event_instance") != 1
        or event.get("expected_event_count") != 1
    ):
        return False
    external = event.get("external_request_id")
    request_id = event.get("request_id")
    match = _EXTERNAL_ID_RE.fullmatch(external) if isinstance(external, str) else None
    internal = (
        _INTERNAL_ID_RE.fullmatch(request_id) if isinstance(request_id, str) else None
    )
    if match is None or internal is None or internal.group(1) != external:
        return False
    if event.get("run_id") != match.group(1) or event.get("cell_id") != match.group(2):
        return False
    if event.get("sample_id") != external:
        return False
    return event.get("sample_index") == int(match.group(4))


def _authoritative(event: Mapping[str, object]) -> bool:
    status = str(event.get("terminal_status", "")).lower()
    if not bool(
        _valid_event_identity(event)
        and event.get("authoritative") is True
        and event.get("event_integrity") == "complete"
        and event.get("terminal_reason") == "completed"
        and (
            status in {"finished", "completed", "success"}
            or status.startswith("finished_")
        )
        and not any(word in status for word in ("aborted", "error", "ignored"))
    ):
        return False
    # Reuse the production mapper's complete identity/lookup/transfer/
    # admission/physical/TP/accounting validator. Future fitting must not
    # accept a weaker event contract than serving attribution.
    from benchmarks.preservation import _map_one_preservation_event

    detail, _ = _map_one_preservation_event(dict(event))
    return detail is not None


def _validated_lookup_queries(event: Mapping[str, object]) -> set[str]:
    """Return mapper-validated subordinate queried keys, never top-level totals."""
    from benchmarks.preservation import _lookup_set_tokens

    queries: set[str] = set()
    episodes = event.get("lookup_episodes")
    if not isinstance(episodes, list):
        raise TypeError("future-reuse lookup episodes are not a list")
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise TypeError("future-reuse lookup episode is not an object")
        groups = episode.get("groups")
        if not isinstance(groups, Mapping):
            raise TypeError("future-reuse lookup groups are not an object")
        for group in groups.values():
            if not isinstance(group, Mapping):
                raise TypeError("future-reuse lookup group is not an object")
            queried = _lookup_set_tokens(group.get("queried"))
            if queried is None:
                raise ValueError("future-reuse queried keys are malformed")
            queries.update(queried)
    return queries


def _default_decode(value: object):
    if not isinstance(value, str):
        raise TypeError("production OffloadKeys must be canonical hex strings")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("production OffloadKey is not canonical hex") from exc
    if decoded.hex() != value:
        raise ValueError("production OffloadKey must be lowercase contiguous hex")
    if len(decoded) < 4:
        raise ValueError("production OffloadKey is missing its group suffix")
    return decoded


def _decode_key(value: object, decoder: Callable[[object], KeyT] | None):
    decoded = (decoder or _default_decode)(value)
    try:
        hash(decoded)
    except TypeError as exc:
        raise TypeError(
            "future-reuse key decoder must return a hashable OffloadKey"
        ) from exc
    return decoded


def _event_outcomes(event: Mapping[str, object]) -> dict[str, bool]:
    if not _authoritative(event):
        return {}
    prior = set()
    for name in ("prior_stored_keys", "prior_evicted_keys"):
        values = event.get(name)
        if not isinstance(values, list):
            return {}
        for value in values:
            key = _token(value)
            if key is None:
                return {}
            prior.add(key)
    episodes = event.get("lookup_episodes")
    if not isinstance(episodes, list):
        return {}
    outcomes: dict[str, bool] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            return {}
        groups = episode.get("groups")
        if not isinstance(groups, Mapping):
            return {}
        for group in groups.values():
            if not isinstance(group, Mapping):
                return {}
            hits = group.get("hits")
            missing = group.get("missing")
            if not isinstance(hits, list) or not isinstance(missing, list):
                return {}
            for value in hits:
                key = _token(value)
                if key is None:
                    return {}
                previous = outcomes.get(key)
                if previous is False:
                    raise ValueError("contradictory future-reuse hit/missing evidence")
                outcomes[key] = True
            for value in missing:
                key = _token(value)
                if key is None:
                    return {}
                # Cold/no-eligible requests are not negative reuse labels.
                if key in prior:
                    previous = outcomes.get(key)
                    if previous is True:
                        raise ValueError(
                            "contradictory future-reuse hit/missing evidence"
                        )
                    outcomes[key] = False
    return outcomes


def _bind_archived_events(
    supplied_events: Sequence[Mapping[str, object]],
    archived_events: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Use the row's event order and reject a second caller-owned event stream."""
    if not isinstance(supplied_events, Sequence) or isinstance(
        supplied_events, (str, bytes)
    ):
        raise TypeError("future-reuse events must be a sequence")
    supplied_by_id: dict[str, Mapping[str, object]] = {}
    for event in supplied_events:
        if not isinstance(event, Mapping):
            raise TypeError("future-reuse events must be mappings")
        request_id = event.get("external_request_id")
        if not isinstance(request_id, str) or request_id in supplied_by_id:
            raise ValueError("future-reuse supplied event IDs are not unique")
        supplied_by_id[request_id] = event
    archived_by_id: dict[str, Mapping[str, object]] = {}
    for event in archived_events:
        request_id = event.get("external_request_id")
        if not isinstance(request_id, str) or request_id in archived_by_id:
            raise ValueError("future-reuse archived event IDs are not unique")
        archived_by_id[request_id] = event
    if set(supplied_by_id) != set(archived_by_id):
        raise ValueError(
            "future-reuse supplied events do not match archived row events"
        )
    for request_id, archived in archived_by_id.items():
        supplied = supplied_by_id[request_id]
        try:
            archived_token = json.dumps(
                archived, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            supplied_token = json.dumps(
                supplied, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "future-reuse archived events are not canonical JSON"
            ) from exc
        if archived_token != supplied_token:
            raise ValueError(
                "future-reuse supplied event conflicts with archived event "
                f"{request_id}"
            )
    return list(archived_events)


def build_future_reuse_labels(
    events: Sequence[Mapping[str, object]],
    *,
    train_end: int,
    future_window: int,
    train_window_id: str,
    test_window_id: str,
    artifact_manifest: Mapping[str, object] | str | Path,
    artifact_root: str | Path,
    artifact_row: Mapping[str, object],
    split_metadata: Mapping[str, object] | str | Path,
    config_digest: str | None = None,
    source_id: str | None = None,
    config_id: str | None = None,
    key_decoder: Callable[[object], KeyT] | None = None,
) -> FutureReuseLabelBatch:
    """Build labels only from complete future windows inside the train split.

    A key is labeled only when every observed future outcome in its window is
    consistent. Current-event outcomes are never consulted, and events at or
    beyond ``train_end`` are never used to label training decisions. Every
    event must have canonical identity/cardinality, be authoritative, belong to
    one run/cell/prefix sequence. Chronology and event contents come from the
    archived row's ``preservation_request_ids_json`` and
    ``preservation_events_json`` fields. Train/test membership comes only from
    the separately bound ``future-reuse-split-v1`` metadata artifact; raw
    serving events are never annotated. Source and config IDs are derived from
    the manifest, row, split metadata, and config artifacts; caller
    declarations, when supplied, are checked against those derived identities.
    """
    if not isinstance(artifact_row, Mapping):
        raise TypeError("future-reuse artifact row must be a mapping")
    request_ids, archived_events = _archived_row_payload(artifact_row)
    ordered_events = _bind_archived_events(events, archived_events)
    if tuple(event["external_request_id"] for event in ordered_events) != request_ids:
        raise ValueError("future-reuse archived event order does not match request IDs")
    if not isinstance(train_end, int) or not 0 < train_end < len(ordered_events):
        raise ValueError("train_end must be within the event sequence")
    if not isinstance(future_window, int) or future_window <= 0:
        raise ValueError("future_window must be positive")
    if (
        not isinstance(train_window_id, str)
        or not isinstance(test_window_id, str)
        or not train_window_id
        or not test_window_id
        or train_window_id == test_window_id
    ):
        raise ValueError("train/test window IDs must be distinct and non-empty")
    _, split_metadata_digest, window_membership = _validated_split_metadata(
        split_metadata,
        artifact_manifest=artifact_manifest,
        artifact_row=artifact_row,
        request_ids=request_ids,
        train_end=train_end,
        train_window_id=train_window_id,
        test_window_id=test_window_id,
    )
    observed_windows: set[str] = set()
    sequence_identity: tuple[object, object] | None = None
    sequence_prefix: str | None = None
    for event in ordered_events:
        if "sequence_ordinal" in event:
            raise ValueError(
                "future-reuse sequence ordinals are not archived chronology"
            )
        if not _valid_event_identity(event):
            raise ValueError("invalid future-reuse event identity/cardinality")
        if not _authoritative(event):
            raise ValueError(
                "invalid authoritative future-reuse event: "
                "events must be authoritative and complete"
            )
        identity = (event["run_id"], event["cell_id"])
        if sequence_identity is None:
            sequence_identity = identity
        elif identity != sequence_identity:
            raise ValueError("future-reuse events mix run/cell sequences")
        match = _EXTERNAL_ID_RE.fullmatch(event["external_request_id"])
        assert match is not None
        prefix = event["external_request_id"][: match.start(4)]
        if sequence_prefix is None:
            sequence_prefix = prefix
        elif prefix != sequence_prefix:
            raise ValueError("future-reuse events mix request ID prefixes")
        request_id = event["external_request_id"]
        if request_id not in window_membership:
            raise ValueError("future-reuse train/test window provenance mismatch")
        observed_windows.add(window_membership[request_id])
    if observed_windows != {train_window_id, test_window_id}:
        raise ValueError("both disjoint future-reuse windows are required")
    derived_source_id, derived_config_id, row_prefix = _validated_artifact_provenance(
        ordered_events,
        artifact_manifest=artifact_manifest,
        artifact_root=artifact_root,
        artifact_row=artifact_row,
        config_digest=config_digest,
        source_id=source_id,
        config_id=config_id,
        chronological_request_ids=request_ids,
        split_metadata_sha256=split_metadata_digest,
    )
    if row_prefix is not None and row_prefix != sequence_prefix:
        raise ValueError("future-reuse row request_id_prefix does not match events")
    for event in ordered_events:
        if window_membership[event["external_request_id"]] not in {
            train_window_id,
            test_window_id,
        }:
            raise ValueError("future-reuse train/test window provenance mismatch")
    decode = key_decoder
    votes: dict[str, set[bool]] = {}
    for index, event in enumerate(ordered_events[:train_end]):
        future_stop = index + future_window + 1
        if future_stop > train_end:
            continue
        candidate_tokens = _validated_lookup_queries(event)
        future_events = ordered_events[index + 1 : future_stop]
        future_queries: list[set[str]] = []
        for future_event in future_events:
            _event_outcomes(future_event)
            future_queries.append(_validated_lookup_queries(future_event))
        if len(future_queries) != future_window:
            continue
        for key in candidate_tokens:
            votes.setdefault(key, set()).add(
                any(key in queries for queries in future_queries)
            )
    entries: list[tuple[object, bool]] = []
    for key in sorted(votes):
        observed = votes[key]
        if len(observed) == 1:
            entries.append((_decode_key(json.loads(key), decode), next(iter(observed))))
    batch = object.__new__(FutureReuseLabelBatch)
    object.__setattr__(batch, "_entries", tuple(entries))
    object.__setattr__(batch, "_train_window_id", train_window_id)
    object.__setattr__(batch, "_test_window_id", test_window_id)
    object.__setattr__(
        batch,
        "_train_request_ids",
        tuple(event["external_request_id"] for event in ordered_events[:train_end]),
    )
    object.__setattr__(
        batch,
        "_test_request_ids",
        tuple(event["external_request_id"] for event in ordered_events[train_end:]),
    )
    object.__setattr__(batch, "_source_id", derived_source_id)
    object.__setattr__(batch, "_config_id", derived_config_id)
    object.__setattr__(batch, "_future_window", future_window)
    object.__setattr__(batch, "_split_metadata_sha256", split_metadata_digest)
    seal = hmac.new(
        _SEAL_KEY,
        json.dumps(
            _sealed_payload(batch), sort_keys=True, separators=(",", ":")
        ).encode(),
        hashlib.sha256,
    ).hexdigest()
    object.__setattr__(batch, "_seal", seal)
    batch._validate_seal()
    return batch


def build_future_reuse_labels_from_artifact(
    events: Sequence[Mapping[str, object]],
    *,
    artifact_root: str | Path,
    train_end: int,
    future_window: int,
    train_window_id: str,
    test_window_id: str,
    variant: str | None = None,
    policy: str | None = None,
    seed: int | str | None = None,
    workload: str | None = None,
    sub_workload: str | None = None,
    request_rate: str | float | None = None,
    sample_index: int | str | None = None,
    manifest_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    split_metadata_path: str | Path | None = None,
    key_decoder: Callable[[object], KeyT] | None = None,
) -> FutureReuseLabelBatch:
    """Evaluate one real, file-backed future-reuse row.

    This is the public offline evaluator entry point.  The caller may provide
    an event stream in any order, but chronology comes from the selected
    canonical CSV row and split metadata, while provenance comes from the
    sibling variant manifest, CSV, row, and split artifacts.
    """
    artifact = load_future_reuse_artifact(
        artifact_root,
        variant=variant,
        policy=policy,
        seed=seed,
        workload=workload,
        sub_workload=sub_workload,
        request_rate=request_rate,
        sample_index=sample_index,
        manifest_path=manifest_path,
        csv_path=csv_path,
    )
    split_path = (
        Path(split_metadata_path)
        if split_metadata_path is not None
        else artifact.manifest_path.parent / "future_reuse_split.json"
    )
    if not split_path.is_absolute():
        split_path = (artifact.manifest_path.parent / split_path).resolve()
    else:
        split_path = split_path.resolve()
    if split_path.parent != artifact.manifest_path.parent:
        raise ValueError("future-reuse split metadata must share the variant directory")
    return build_future_reuse_labels(
        events,
        train_end=train_end,
        future_window=future_window,
        train_window_id=train_window_id,
        test_window_id=test_window_id,
        artifact_manifest=artifact.manifest_path,
        artifact_root=artifact.root,
        artifact_row=artifact.row,
        split_metadata=split_path,
        key_decoder=key_decoder,
    )


# Keep the evaluator name discoverable for callers that describe this step as
# evaluation rather than label building.
evaluate_future_reuse = build_future_reuse_labels_from_artifact

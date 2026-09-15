# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation for MTP/EAGLE KV-offload round-trip evidence.

The validator is deliberately independent of vLLM internals.  A driver can
emit one JSON object per fresh server root, and this module verifies the
source-attributed four-phase protocol before any result is used as evidence.
Missing counters or incomplete source attribution are ``unavailable``; they
are never converted into zeroes or inferred hits.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

RoundTripStatus = Literal["pass", "invalid", "unavailable"]
PHASES = ("cold_store", "gpu_hit", "local_reset", "cpu_restore")
SCHEMA = "mtp-offload-roundtrip-v1"


@dataclass(frozen=True, slots=True)
class RoundTripResult:
    """Machine-readable result suitable for an evidence manifest."""

    status: RoundTripStatus
    errors: tuple[str, ...]
    unavailable: tuple[str, ...]
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: object, name: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if integer and not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _metric(phase: Mapping[str, Any], name: str) -> object:
    metrics = phase.get("metrics")
    if not isinstance(metrics, Mapping):
        raise KeyError("metrics")
    if metrics.get("source") != "prometheus":
        raise KeyError(f"metrics.{name}.source")
    values = metrics.get("values")
    if not isinstance(values, Mapping) or name not in values:
        raise KeyError(f"metrics.values.{name}")
    return values[name]


def _phase_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    phases = payload.get("phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise TypeError("phases must be an array")
    if [phase.get("name") for phase in phases if isinstance(phase, Mapping)] != list(
        PHASES
    ):
        raise ValueError("phases must be the ordered four-phase protocol")
    result: dict[str, Mapping[str, Any]] = {}
    for name, phase in zip(PHASES, phases):
        if not isinstance(phase, Mapping):
            raise TypeError(f"phase {name} must be an object")
        if phase.get("status") != "complete":
            raise ValueError(f"phase {name} is not complete")
        result[name] = phase
    return result


def validate_roundtrip(
    payload: Mapping[str, Any],
    *,
    expected_vllm_revision: str | None = None,
    expected_candidate_sha: str | None = None,
) -> RoundTripResult:
    """Validate one fresh-root MTP/EAGLE offload round trip.

    The result is ``pass`` only when all required source, transfer, output,
    acceptance, and timing/accounting checks are complete.  ``invalid`` means
    the supplied evidence contradicts the contract.  ``unavailable`` means a
    required observation is absent or the requested model/capability was not
    present.
    """
    if not isinstance(payload, Mapping):
        return RoundTripResult("invalid", ("payload must be an object",), (), {})

    errors: list[str] = []
    unavailable: list[str] = []
    metrics: dict[str, Any] = {}
    if payload.get("schema") != SCHEMA:
        errors.append("schema mismatch")

    dependency = payload.get("dependency")
    if not isinstance(dependency, Mapping):
        unavailable.append("dependency metadata missing")
    else:
        vllm_revision = dependency.get("vllm_revision")
        candidate_sha = dependency.get("candidate_sha")
        if not isinstance(vllm_revision, str) or not vllm_revision:
            unavailable.append("vLLM revision unavailable")
        elif (
            expected_vllm_revision is not None
            and vllm_revision != expected_vllm_revision
        ):
            errors.append("vLLM revision does not match expected revision")
        if not isinstance(candidate_sha, str) or not candidate_sha:
            unavailable.append("candidate SHA unavailable")
        elif (
            expected_candidate_sha is not None
            and candidate_sha != expected_candidate_sha
        ):
            errors.append("candidate SHA does not match expected SHA")

    model = payload.get("model")
    if not isinstance(model, Mapping):
        unavailable.append("model metadata unavailable")
    else:
        model_name = model.get("name")
        if not isinstance(model_name, str) or not model_name:
            unavailable.append("model name unavailable")
        if model.get("mtp_enabled") is not True:
            unavailable.append("MTP/EAGLE capability unavailable")

    try:
        phases = _phase_map(payload)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        phases = {}

    try:
        control = payload["control"]
        if not isinstance(control, Mapping):
            raise TypeError("control must be an object")
        greedy_tokens = control["greedy_tokens"]
        restored_tokens = phases["cpu_restore"]["greedy_tokens"]
        if not (
            isinstance(greedy_tokens, list)
            and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in greedy_tokens
            )
        ):
            raise ValueError("control.greedy_tokens must be an integer array")
        if restored_tokens != greedy_tokens:
            errors.append("CPU-restore greedy output differs from control")

        block_size = _number(control["block_size"], "control.block_size", integer=True)
        if block_size <= 0:
            raise ValueError("control.block_size must be positive")
        control_prefix = _number(
            control["control_prefix_tokens"],
            "control.control_prefix_tokens",
            integer=True,
        )
        restored_prefix = _number(
            _metric(phases["cpu_restore"], "restored_prefix_tokens"),
            "restored_prefix_tokens",
            integer=True,
        )
        if abs(restored_prefix - control_prefix) > block_size:
            errors.append("restored prefix differs from control by more than one chunk")

        for name in PHASES:
            phase = phases[name]
            for field in ("errors", "stalls"):
                value = _metric(phase, field)
                if _number(value, f"{name}.{field}", integer=True) != 0:
                    errors.append(f"{name} reports {field}")

        store_bytes = _number(
            _metric(phases["cold_store"], "store_bytes"), "store_bytes"
        )
        gpu_hits = _number(
            _metric(phases["gpu_hit"], "local_gpu_hit_tokens"),
            "local_gpu_hit_tokens",
            integer=True,
        )
        restore_bytes = _number(
            _metric(phases["cpu_restore"], "cpu_to_gpu_bytes"),
            "cpu_to_gpu_bytes",
        )
        external_tokens = _number(
            _metric(phases["cpu_restore"], "external_tokens"),
            "external_tokens",
            integer=True,
        )
        restore_gpu_hits = _number(
            _metric(phases["cpu_restore"], "local_gpu_hit_tokens"),
            "restore_local_gpu_hit_tokens",
            integer=True,
        )
        metrics.update(
            {
                "store_bytes": store_bytes,
                "gpu_hit_tokens": gpu_hits,
                "restore_cpu_to_gpu_bytes": restore_bytes,
                "restore_external_tokens": external_tokens,
            }
        )
        if store_bytes <= 0:
            errors.append("cold store produced no attributed bytes")
        if gpu_hits <= 0:
            errors.append("GPU-hit phase produced no local hit tokens")
        if restore_bytes <= 0 or external_tokens <= 0:
            errors.append("CPU-restore phase produced no attributed external load")
        if restore_gpu_hits != 0:
            errors.append("CPU-restore phase was attributed as a local GPU hit")
        if phases["local_reset"].get("reset_scope") != "local_gpu_only":
            errors.append("reset scope is not local_gpu_only")

        control_rate = _number(control["acceptance_rate"], "control.acceptance_rate")
        restore_rate = _number(
            _metric(phases["cpu_restore"], "acceptance_rate"),
            "restore.acceptance_rate",
        )
        if not 0.0 <= control_rate <= 1.0 or not 0.0 <= restore_rate <= 1.0:
            raise ValueError("acceptance rates must be in [0, 1]")
        acceptance_loss = control_rate - restore_rate
        metrics["acceptance_loss"] = acceptance_loss
        if acceptance_loss > 0.02:
            errors.append("acceptance loss exceeds 2 percent")

        control_ttft = _number(
            control["post_reset_ttft_s"], "control.post_reset_ttft_s"
        )
        restore_ttft = _number(
            _metric(phases["cpu_restore"], "ttft_s"), "restore.ttft_s"
        )
        if control_ttft <= 0 or restore_ttft <= 0:
            raise ValueError("TTFT values must be positive")
        metrics["post_reset_ttft_speedup"] = control_ttft / restore_ttft
        token_accounting = payload.get("token_accounting")
        exact_accounting = (
            isinstance(token_accounting, Mapping)
            and token_accounting.get("exact") is True
            and isinstance(token_accounting.get("restored_tokens"), int)
            and token_accounting["restored_tokens"] > 0
        )
        if metrics["post_reset_ttft_speedup"] < 3.0 and not exact_accounting:
            errors.append(
                "restore has neither 3x TTFT improvement nor exact token accounting"
            )
        metrics["exact_token_accounting"] = exact_accounting
    except KeyError as exc:
        unavailable.append(f"required observation missing: {exc.args[0]}")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    if errors:
        status: RoundTripStatus = "invalid"
    elif unavailable:
        status = "unavailable"
    else:
        status = "pass"
    return RoundTripResult(
        status, tuple(sorted(set(errors))), tuple(sorted(set(unavailable))), metrics
    )


def validate_roundtrip_file(
    path: str | Path,
    *,
    expected_vllm_revision: str | None = None,
    expected_candidate_sha: str | None = None,
) -> RoundTripResult:
    """Load and validate one JSON round-trip manifest."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return RoundTripResult("invalid", (f"cannot read manifest: {exc}",), (), {})
    return validate_roundtrip(
        payload,
        expected_vllm_revision=expected_vllm_revision,
        expected_candidate_sha=expected_candidate_sha,
    )


__all__ = [
    "PHASES",
    "SCHEMA",
    "RoundTripResult",
    "validate_roundtrip",
    "validate_roundtrip_file",
]

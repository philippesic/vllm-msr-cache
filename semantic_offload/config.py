"""Public vLLM configuration helpers for the MSR policy."""

from __future__ import annotations

from typing import Any


def msr_kv_transfer_config(
    cpu_bytes_to_use: int,
    *,
    store_threshold: int = 2,
    prefetch_budget_fraction: float = 0.05,
    prefetch_score_max_age_ms: int = 1000,
    **extra_config: Any,
) -> dict[str, Any]:
    """Return a vLLM ``kv_transfer_config`` for MSR with semantic prefetch.

    The defaults match the paused project's reviewed DRAM configuration:
    threshold-2 admission, mean semantic relevance, and a bounded 5% GPU
    prefetch budget. Additional connector options may be supplied as keyword
    arguments; connector/spec identity fields remain authoritative.
    """

    if isinstance(cpu_bytes_to_use, bool) or not isinstance(cpu_bytes_to_use, int):
        raise TypeError("cpu_bytes_to_use must be an integer")
    if cpu_bytes_to_use <= 0:
        raise ValueError("cpu_bytes_to_use must be positive")
    if isinstance(store_threshold, bool) or not isinstance(store_threshold, int):
        raise TypeError("store_threshold must be an integer")
    if store_threshold < 0:
        raise ValueError("store_threshold must be non-negative")
    if isinstance(prefetch_budget_fraction, bool) or not isinstance(
        prefetch_budget_fraction, (int, float)
    ):
        raise TypeError("prefetch_budget_fraction must be a number")
    if not 0.0 < float(prefetch_budget_fraction) <= 1.0:
        raise ValueError("prefetch_budget_fraction must be greater than 0 and at most 1")
    if isinstance(prefetch_score_max_age_ms, bool) or not isinstance(
        prefetch_score_max_age_ms, int
    ):
        raise TypeError("prefetch_score_max_age_ms must be an integer")
    if prefetch_score_max_age_ms <= 0:
        raise ValueError("prefetch_score_max_age_ms must be positive")

    config: dict[str, Any] = {
        "cpu_bytes_to_use": cpu_bytes_to_use,
        "method": "mean",
        "store_threshold": store_threshold,
        "prefetch_budget_fraction": float(prefetch_budget_fraction),
        "prefetch_selector": "semantic",
        "prefetch_score_max_age_ms": prefetch_score_max_age_ms,
    }
    config.update(extra_config)
    config.update(
        {
            "spec_name": "SemanticOffloadingSpec",
            "spec_module_path": "semantic_offload.spec",
            "cpu_bytes_to_use": cpu_bytes_to_use,
            "method": "mean",
            "prefetch_selector": "semantic",
        }
    )
    return {
        "kv_connector": "SemanticOffloadingConnector",
        "kv_connector_module_path": "semantic_offload.connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": config,
    }

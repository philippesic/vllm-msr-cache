from __future__ import annotations

import pytest

from semantic_offload.config import msr_kv_transfer_config


def test_msr_config_enables_paired_semantic_prefetch() -> None:
    config = msr_kv_transfer_config(8 * 1024**3)

    assert config["kv_connector"] == "SemanticOffloadingConnector"
    assert config["kv_connector_module_path"] == "semantic_offload.connector"
    assert config["kv_role"] == "kv_both"
    extra = config["kv_connector_extra_config"]
    assert extra == {
        "cpu_bytes_to_use": 8 * 1024**3,
        "method": "mean",
        "store_threshold": 2,
        "prefetch_budget_fraction": 0.05,
        "prefetch_selector": "semantic",
        "prefetch_score_max_age_ms": 1000,
        "spec_name": "SemanticOffloadingSpec",
        "spec_module_path": "semantic_offload.spec",
    }


def test_msr_config_keeps_identity_fields_authoritative() -> None:
    config = msr_kv_transfer_config(
        1024,
        method="minmax",
        prefetch_selector="request_order",
        spec_name="OtherSpec",
        scoring_mode="asynchronous",
    )

    extra = config["kv_connector_extra_config"]
    assert extra["method"] == "mean"
    assert extra["prefetch_selector"] == "semantic"
    assert extra["spec_name"] == "SemanticOffloadingSpec"
    assert extra["scoring_mode"] == "asynchronous"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"cpu_bytes_to_use": 0}, ValueError),
        ({"cpu_bytes_to_use": True}, TypeError),
        ({"cpu_bytes_to_use": 1024, "store_threshold": -1}, ValueError),
        ({"cpu_bytes_to_use": 1024, "prefetch_budget_fraction": 0}, ValueError),
        ({"cpu_bytes_to_use": 1024, "prefetch_budget_fraction": 1.1}, ValueError),
        ({"cpu_bytes_to_use": 1024, "prefetch_score_max_age_ms": 0}, ValueError),
    ],
)
def test_msr_config_rejects_invalid_values(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        msr_kv_transfer_config(**kwargs)

# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.latency_validation import validate_latency_row


@pytest.mark.parametrize("prefix", ["ttft", "itl"])
@pytest.mark.parametrize("values", [(1, 2, 3, 4, 5), (0, 0, 0, 0, 0), (1, 1, 1, 1, 1)])
def test_tail_percentiles_accept_ascending_or_equal_values(prefix, values):
    row = dict(
        zip(
            (f"{prefix}_{part}_ms" for part in ("p50", "p90", "p95", "p99", "max")),
            values,
        )
    )
    assert validate_latency_row(row) == []


@pytest.mark.parametrize("prefix", ["ttft", "itl"])
@pytest.mark.parametrize("values", [(1, 2, 5, 4, 6), (1, 3, 2, 4, 5), (1, 2, 3, 5, 4)])
def test_tail_percentiles_reject_each_inversion(prefix, values):
    row = dict(
        zip(
            (f"{prefix}_{part}_ms" for part in ("p50", "p90", "p95", "p99", "max")),
            values,
        )
    )
    assert any("not monotonic" in error for error in validate_latency_row(row))


def test_optional_tail_pair_still_checks_its_own_order():
    assert any(
        "not monotonic" in error
        for error in validate_latency_row({"ttft_p95_ms": 5, "ttft_max_ms": 4})
    )


def test_independent_percentile_errors_aggregate():
    errors = validate_latency_row(
        {
            "ttft_p50_ms": 1,
            "ttft_p90_ms": 2,
            "ttft_p99_ms": 3,
            "ttft_p95_ms": float("nan"),
            "ttft_max_ms": 5,
            "itl_p95_ms": 1,
        }
    )
    assert "invalid tail field ttft_p95_ms" in errors
    assert "missing required tail field itl_max_ms" in errors

"""Recompute reported intervals and memory savings from the portable paired CSV.

Run with Python 3; no third-party packages. This checks aggregate arithmetic,
not the separate physical KV proofs, byte audit, or raw measurement provenance.
Inference is conditional on the three fixed traces in result.json.
"""
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "confirmation-pairs.csv").open()))
report = json.loads((HERE / "result.json").read_text())
assert len(rows) == 12
assert len({(r["seed"], r["repetition"]) for r in rows}) == 12


def check(values, expected):
    mean = statistics.mean(values)
    half = 2.200985160082949 * statistics.stdev(values) / math.sqrt(12)
    probability = sum(
        sum(s * v for s, v in zip(signs, values)) >= sum(values) - 1e-14
        for signs in itertools.product((-1, 1), repeat=12)
    ) / 4096
    actual = dict(zip(
        ("geometric_mean_gain_fraction", "lower95_gain_fraction",
         "upper95_gain_fraction", "one_sided_exact_sign_flip_p"),
        (math.expm1(mean), math.expm1(mean - half), math.expm1(mean + half), probability),
    ))
    for key, value in actual.items():
        assert math.isclose(value, expected[key], abs_tol=1e-12), (key, value)
    assert mean - half > 0 and probability <= .05
    return actual


ratios = [float(r["candidate_throughput_tokens_s"]) /
          float(r["arc_throughput_tokens_s"]) for r in rows]
check([math.log(x) for x in ratios], report["primary"])
for metric, expected in report["throughput_per_vram"].items():
    savings = [int(r["arc_" + metric]) - int(r["candidate_" + metric]) for r in rows]
    assert savings == [58720256] * 12
    check([math.log(ratio * int(r["arc_" + metric]) /
                    int(r["candidate_" + metric]))
           for r, ratio in zip(rows, ratios)], expected)
latency_savings = [float(r["arc_all_mean_ttft_ms"]) -
                   float(r["candidate_all_mean_ttft_ms"]) for r in rows]
assert math.isclose(statistics.mean(latency_savings),
                    report["latency_arc_minus_candidate_ms"]["all_ttft_saving_ms"]["mean"],
                    abs_tol=1e-9)
assert statistics.mean(latency_savings) < 0
print("PASS: 12 paired throughput/VRAM intervals, exact sign-flip probabilities, "
      "56 MiB savings in all three memory measures, and mean TTFT regression.")

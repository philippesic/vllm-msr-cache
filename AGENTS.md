# Research context

Can exact CPU KV offload plus causal tool-wait advice improve completed,
correct agent work over matched native ARC, without unacceptable overhead
or interference with other traffic? This repository is currently foundation only.

KV means attention keys and values. Reuse requires identical prefix tokens,
model, adapter, positions and layout. Semantic scores may rank retention or
prefetch candidates; they never substitute an answer or authorize approximate KV.
ARC is adaptive replacement; a wait hint is a fallible declaration of future
need, not pinning, ownership or proof of residency. TTFT is time to first token.

Refuse comparisons across machines or cache states. A declared system comparison
may vary runtime fields within matched machine/cache state; cross-host observations
cannot isolate GPU effects. Report missing observations as unavailable.

# Working here

Run `uv run --locked python tools/check.py`; see [checks](docs/checks.md).
The future benchmark entry is `uv run --locked python -m benchmarks`
with plan/run/verify/report modes; it does not exist yet.
See [benchmarking](docs/benchmarking.md) before designing an experiment.

# Judgment the checks cannot provide

Preserve adverse and inconclusive evidence alongside gains. Choose workloads
that exercise the claimed mechanism; a pressure claim needs actual CPU restores
and, for retention, eviction-affected returns. Microbenchmarks alone do not
establish serving value. Frozen failures do not justify favorable retries.

Treat functional qualification, throughput and answer quality as separate claims.
Do not infer quality equivalence from token counts, identical selected answers,
or failure-selected reviews. Declare useful-effect and fairness margins before
confirmation. Coordinate shared hardware with its owner before future GPU work.

Use [FINDINGS](FINDINGS.md) for durable conclusions, with scope and immutable
sources. Read the actual evidence when changing a claim. Keep the old repository
and its sealed evidence intact; this step grants no authority to repair history.

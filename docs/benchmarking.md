# Benchmark contract (future implementation)

No benchmark code is implemented. The planned sole entry is
`uv run --locked python -m benchmarks`, with `plan`, `run`, `verify`, `report`.
Do not run these examples until that separately authorized work exists:

```text
python -m benchmarks plan --tier screen --config <tracked-definition> --out <external-run>
python -m benchmarks run --plan <external-run>/plan.json
python -m benchmarks verify <external-run>
python -m benchmarks report <external-run> --comparison policy
```

Definitions and small sanitized fixtures belong in Git. Generated plans, frozen
execution trees, raw prompts/responses, logs and run artifacts need an explicit
external output root, not a directory under this repository. The artifact root
and backup/retention policy remain Philip's decision. Run identities must be new;
failed attempts, partial blocks and seals must survive relocation and reporting.

## What a comparison must establish

Refuse cross-machine or different-cache-state comparisons. Match model
and tokenizer digests, runtime/build identity, CPU/GPU byte budgets and block
geometry, workload/tool corpus, cache phase, generation mode/caps, scheduling,
arm order and independent process repetitions. Different models or runtimes need
a declared system comparison with explicit limits on attribution. Never combine
cold, warm, GPU-resident and CPU-restored traffic into an unlabeled comparison.
Missing telemetry is unavailable, not zero. A future comparator must enforce
these rules; this foundation's check validates tooling, not benchmark data.

Pressure means timed returning CPU restores in both native arms; retention claims
also need CPU evictions and eviction-affected returns. Stores, ready-recency
touches and GPU preemptions cannot stand in for restores. A zero-restore workload
may be a useful GPU-resident control. Calibrate separately, then freeze geometry;
a failed activity gate remains failed without favorable retries or partial credit.

## Tiers and interpretation

| Tier | Historical default for migration | Purpose |
| --- | --- | --- |
| Smoke | 1 repeat; 4 sessions x 2 requests; concurrency 1; cap 8 | Install, exact restore/fresh-reference probe, cleanup |
| Screen | 1 repeat; 6 x 2; concurrency 2; cap 32 | Mechanism activity and bounded cost |
| Confirmation | At least 3 repeats; 12 x 3; concurrency 4; cap 128 | Frozen held-out workload, whole-process blocks and declared precision |
| Operational | 1 process/arm; 12 x 3 x 4 waves | Persistent pressure, bounds, cancellation/drain/recovery |

These are old defaults, not a sufficient sample-size claim. Stage 6 often used
6–12 process pairs. Waves within a process and requests within a run are not
independent timing repetitions. Real executed tool graphs are a workload kind,
not another overlapping ladder of launchers. No future-window oracle labels may
flow into live decisions; hints are causal, fallible information controls.

Report completed offered graph units/time with whole-process paired uncertainty,
per-class TTFT/completion latency, caps, failure denominators, mechanism activity
and quality separately. No favorable partial-block report or false complete flag.
The old 3% overhead and 5% fairness bounds were not universally met; new useful
effect/quality margins need a decision before confirmation. Preserve source-based
quality review scope; selected failures are not population accuracy estimates.

The historical measurements and exact evidence pointers live only in
[FINDINGS](../FINDINGS.md). [Runtime](runtime.md) describes qualification needed
before either GPU can be used. No GPU acquisition is part of this step.

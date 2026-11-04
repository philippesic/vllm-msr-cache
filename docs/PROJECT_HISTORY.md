# Brief project history

Completed milestones and useful negative findings. Active assignments live in
[STATUS.md](STATUS.md); detailed measurements belong in [evidence](../evidence/README.md).
Add a short dated outcome with a link when an effort finishes, not a transcript
of intermediate commands.

- **July–August 2026:** built exact KV offload, eviction/prefetch, query capture,
  scoring, and the measurement harness. Bookkeeping and capture overhead emerged
  as important costs.
- **September 2026:** E04 supported native threshold-2 admission, not a semantic
  policy advantage. A27 demonstrated exact semantic activation and consumption
  but no repeatable benefit over controls. These were other-host results, not
  2080 Ti measurements. [Reports and limitations](../evidence/README.md).
- **October 31–November 1, 2026:** 2080 Ti workloads exposed continuous-path
  overhead. Capture and batching changes improved throughput, but the general
  within-1%-of-ARC goal remained unmet; reduced observation frequency had quality
  trade-offs. Local report: `output/scoring-overhead-20261101/REPORT.md`.
- **November 1–2, 2026:** broader quantized-model testing did not establish a
  general prefetch advantage. Smaller-cache ARC could obtain similar memory
  savings; speculative reservations and wasted transfers remained costs. Local
  report: `output/quantized-realworld-20261101/REPORT.md`.
- **November 2, 2026:** CPU-only ARC recency advice produced a small conditional
  throughput/VRAM gain in exact recorded-output replay, with worse first-token
  latency. It disabled speculative GPU copies and did not establish a semantic
  ranking win. [Result](../evidence/cpu-wait-recency-2080ti-20261102/result.json).
- **November 3, 2026:** connected/synchronized the Git remote and brought the
  patched vLLM source into the repository. Added exact runtime requirements and
  a same-interpreter launcher guard; 60 focused tests passed. Fresh installation
  and GPU-launcher qualification remain open in STATUS.

- **November 3, 2026 — workflow cleanup:** separated active status, brief history,
  configuration, and evidence. Removed the unscheduled backlog and documentation
  archive at the user’s request. Added
  runtime/test/benchmark-help entry points and contextual agent notes. Runtime
  imports, 60 focused tests, 120 harness checks, CLI help, and documentation links
  passed. Active campaign files were left in place; GPU qualification is still open.

Local `output/` reports are
not distributed by Git; the retained evidence index identifies portable results.

- **November 3, 2026 — scoped live qualification:** GPU-gated CPU ARC advice with
  fixed continuation priority measured +6.10% token throughput and 485 ms lower
  mean first-token latency against equally prioritized native ARC on fresh
  natural generation. VRAM was unchanged; the p95 interval included regression.
  Another 5,328 workload plus 16 health completions covered persistence, bounds,
  capacity fallback, exact abort recovery and finite overlap. Lead and Astra
  support controlled asynchronous drafts with human review on the exact measured
  snapshot, with explicit registry lifetime and cold/burst-delay tolerance.
  Interactive/general production, autonomous quality, later telemetry edits and
  the new runtime launcher remain unqualified. Historical failures are retained;
  GPU custody and owned processes were released.
  [Assessment and operating contract](../evidence/live-optimization-2080ti-20261103/operational-final-root-assessment.json).

- **November 3, 2026 — maintained benchmarks:** added smoke, advisory screen,
  paired confirmation, and persistent operational commands with sealed artifacts,
  native activity/abort receipts and offline verification. Qualified 608 normal
  requests across 18 successful launches, including public-input graphs on 0.5B
  and 7B AWQ; real failure controls and 1,348 regression tests passed (26 historical
  artifact-dependent skips). Fixed wheel-support staging and interruption cleanup.
  GPU released; useful failures retained. Fresh installation remains unverified.
  [Commands](BENCHMARKING.md), [qualification](../evidence/benchmark-suite-20261103/README.md).

- **November 3, 2026 — workflow finalization:** unified maintained GPU reservations
  across whole invocations; promoted reusable research validators; verified an
  isolated runtime installation, 1,381 tests and real GPU smoke; added lightweight
  push/PR checks (31 portable tests). Preserved paused research and historical
  reproduction files. [Verification](../evidence/workflow-finalization-20261103/README.md).

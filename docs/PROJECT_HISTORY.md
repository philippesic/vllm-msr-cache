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

- **November 4, 2026 — hybrid qualification evidence:** resolved the interrupted
  source inspection with bounded row-group audits; verified prior pressure traces
  and completed a longer-history diagnostic with 13 restores per arm. A 711-example
  task-separated pilot did not establish primary one-second gains. Exploratory
  quarter-second richer Q/K gains survived more flexible metadata controls, but remain
  simulated cache-value evidence. Real-cache-value, causal opportunity and native
  cost gates remain unmet; no live hybrid or policy change. 170 focused and 31
  portable tests passed; GPU released, useful data/results retained.
  [Findings and stopping boundary](../evidence/msr-hybrid-qualification-20261103/README.md).

- **November 4, 2026 — semantic KV SVD serving factorial:** implemented actual
  lossy KV-factor offload, tested all four semantic ablations against ARC and
  wait-hint ARC, then refined solver batching, capacity, hint timing and staging
  leases. 54 cases completed 972 responses. Rank 4 failed quality; rank 8 and
  active prefetch did not beat wait-hint ARC after repairs and diagnostic-off
  checks. No deployment; prior work preserved.
  [Comparisons and bounded stopping decision](../evidence/semantic-svd-serving-20261104/README.md).

- **November 4, 2026 — real repository-agent evaluation:** added `bench-agent` to
  the maintained suite, replacing recorded tool observations for this workflow
  with executed searches/tests and actual multi-request model history. Fixed
  canonical request attribution, inactive CPU-advice configuration, cancelled
  hint accounting and ephemeral-stage sealing. Four contended 7B cases passed
  (128 responses, 96 tools), with active CPU refreshes, more restored tokens and
  complete native/endpoint accounting. Two reversed-order pairs are acceptance
  evidence, not a causal policy win. Corrected the earlier broad negative scope;
  1,423 tests passed, 26 skipped, GPU released and prior work preserved.
  [Measured results and reusable commands](../evidence/real-agent-eval-20261104/README.md).

- **November 4, 2026:** implemented independent HTTP admission, real tool-worker
  queueing, early CPU wait retention, primary workflow/FTL intervals and corrected
  first-admission accounting. A bounded capacity/GPU-gate/host-memory hill climb
  ended with three rotated live 7B comparisons: wait ARC +7.0% workflow throughput,
  +6.2% token rate, -0.43 s average FTL versus ARC. Refined semantic retention was
  about 3.0% slower than wait ARC; tails/VRAM did not improve, broader answer quality
  remains ungraded. All nine run sets verify, including failed/interrupted controls;
  GPU released and prior work preserved. [Evidence and command](../evidence/wait-opportunity-20261104/README.md).

- **November 4, 2026:** adapted the real-tool benchmark for one pass through ARC,
  wait-hint, semantic, pinned Continuum public preview and upstream PR #59714.
  All five measurement cases passed (320 responses / 240 tools); wait-hint finished
  the batch fastest. Eager execution, shared tool-declaration prompts, distinct
  runtimes and upstream's WSL unpinned-memory setting limit comparisons with earlier
  runs and other hardware. Published policy activation was observed. GPU released;
  reusable environments retained. [Results and reproduction](../evidence/five-way-20261104/README.md).

## Upstream pinned-memory compatibility — 2026-11-04

A separately labeled experimental patch resolves registered mmap CPU pointers to
CUDA device aliases. The reproduced native batch-copy fault disappears; three
7B requests pass with identical output tokens and two observed 163 MiB CPU
restores under pinned memory. Published source and earlier results stay intact.
GPU processes reaped and both locks released; no full rerun or performance claim.
[Evidence, patch, limitations and cleanup](../evidence/upstream-pin-fix-20261104/README.md).

## Controlled four-system comparison — 2026-11-05

One primary pass per configuration compared plain ARC without hint machinery,
active wait ARC, Continuum public preview and upstream draft plus the explicit
pinned-mmap host-pointer fix. Native upstream used CPU ARC, V2 and the same
application priority as local arms; actual pinned CPU/GPU budgets matched.
The primary 256 EOS responses / 192 tools completed: ARC 108.153 s, wait 101.628 s,
upstream 102.711 s, corrected Continuum 109.108 s. Continuum's supported
save_decode_cache=true was predeclared as a replacement irrespective timing;
35 decode saves match completed stores and the initial false row stays diagnostic.
Runtime/backend, live output, observed paging and orchestration/correction gaps
prevent a winner or isolated policy claim. Normalized test outcomes were correct;
strict requested format mostly failed and reasoning remains ungraded. Initial
orchestration failure and one-arm continuation are preserved; original four-case
files stayed unchanged. Both sealed run sets verify (516 + 139 files); 124
benchmark and 31 portable checks pass. GPU released, evidence and paused work
preserved. [Results and limitations](../evidence/fair-four-20261104/README.md).

## Four-stage wait-hint qualification — 2026-11-05

Completed the selected local causal, competitive, workload and compatibility
studies. Main-comparison batch time fell 7.51% versus ARC; fresh repository and
full-budget document reserves saved 8.27% and 18.12%, and timed graphs saved
8.87%. Tight documents and several other regimes remain unresolved; some favor
upstream/Continuum. Quality and production gates remain unmet. Eight functional
cells and both client examples pass; all 46 retained attempts and the finished
archive verify. GPU released, useful evidence and recovery material preserved.
[Findings, limits and reproduction](../evidence/wait-qualification-20261105/README.md).

- **November 6, 2026:** Stage 5 optimization completed under the user's narrowed
  final scope: 144 valid GPU cases, two retained invalid configurations, and no
  new serving variant qualified over the existing wait-hint rule. Linear eviction
  reproduced substantial CPU gains on 47 real cache states, but native eviction
  stayed below 0.1% of workflow time even with large real-document batches.
  Nine final functional waves and operational recovery checks passed. GPU released;
  Stage 6 fresh validation remains pending. [Findings and selected profile](../evidence/stage5-optimization-20261105/README.md).

- **November 8, 2026 — bounded Stage 6 evidence and transfer complete:** acquired
  524 primary configurations in 210 whole blocks; retained all failures and the
  bounded addendum's 24 QA pairs/four advice blocks. Reviewed all selected source
  outputs and all 699 archived caps. Primary and separate addendum relocation,
  offline verification and rebuilt-runtime02 GPU-free imports passed; actual
  comparison HTML was rendered and gated 5080 commands retained. Results depend
  on workload/model; Continuum wins document cells, candidate wins some mixed
  traffic, QA's 3% overhead bound remains unresolved, and misleading hints delay
  unhinted traffic. The repository addendum gate failed before serving and closed
  without replacement, leaving repeated four-system pressure evidence unmeasured.
  No answer-quality equivalence, production or 5080 performance claim. Raw
  evidence and paused AWQ remain preserved; both reservations are released and
  the existing health schedule is disabled. Stage 7 is separate pending work.
  [Findings](../evidence/stage6-2080ti-20261106/RESULTS.md),
  [handoff](../evidence/stage6-2080ti-20261106/HANDOFF_5080.md).

# Results index

Compact retained evidence, organized by experiment. Current work and next actions
live in [STATUS](../docs/STATUS.md). Raw runs remain in ignored `output/` and may
not be available in a fresh clone.

## November 5: four-stage wait-hint qualification

Selected local execution across all four stages is complete. The main comparison
used 7.51% less workflow batch time than ARC; fresh repository and full-budget
document reserves saved 8.27% and 18.12%, and timed CUDA graphs saved 8.87%.
Tight documents and several other regimes remain unresolved, upstream/Continuum
win in some regimes, and answer-quality and production gates remain unmet.
Eight functional cells, both client examples and isolated installation checks
pass; all 46 retained coordinator attempts verify, including failures and repairs.
[Findings and reproduction](wait-qualification-20261105/README.md),
[support boundaries](wait-qualification-20261105/SUPPORT_MATRIX.md), and
[adverse findings](wait-qualification-20261105/REGRESSIONS.md).

## November 5: controlled four-system repository comparison

Plain ARC without hints, active wait ARC, Continuum and compatibility-patched
upstream completed 256 EOS responses / 192 tools. Native upstream CPU ARC,
priority forwarding, V2 and pinned backing corrected earlier configuration gaps.
Wait's 101.628 s versus upstream's 102.711 s does not establish a winner; outputs,
versions/backends, paging and documented orchestration/correction gaps differ.
Continuum uses a predeclared supported decode-publication correction; its earlier
false row remains diagnostic. Strict status
format compliance mostly failed despite correct normalized test-outcome reports.
[Exact numbers, provenance and remaining limits](fair-four-20261104/README.md).

## November 4: five-system real repository comparison

One pass through ARC, wait-hint, semantic, Continuum public preview and upstream
retention PR #59714 completed 320 responses and 240 real tools. Wait-hint finished
the workflow batch fastest; upstream had the highest token rate with more output.
All five measurement cases verified. This uses eager execution and adapted shared
prompts, with separate published runtimes and explicit WSL compatibility settings.
[Numbers, limitations and reproduction](five-way-20261104/README.md).

## November 4: wait-hint opportunity in real queued serving

`make bench-agent-pressure` exposes server queues, actual tool-worker waiting and
competing CPU/GPU prefixes. Capacity, GPU-gate and host-memory refinements led to
three rotated ARC/wait/semantic comparisons: wait ARC improved workflow throughput
7.0%, token throughput 6.2% and average first-token latency 0.43 s. Semantic plus
wait was about 3.0% slower than wait ARC. All 576 final responses and 144 reported
test outcomes passed; p95/VRAM did not improve, broader quality remains ungraded.
[Results, intervals, controls, failures and reproduction](wait-opportunity-20261104/README.md).

## November 4: real repository-agent serving benchmark

`make bench-agent` runs actual repository search/test tools between live model
requests. Four contended 7B ARC/wait cases passed: 128 responses, 96 tools,
active CPU refreshes, native eviction/reuse records and consistent endpoint
accounting. Wait arms restored more tokens and had modestly shorter workflows;
different live outputs and two pairs prevent a causal performance claim.
[Commands, metrics and measured acceptance](real-agent-eval-20261104/README.md).

## November 4: actual KV SVD and semantic prefetch factorial

54 serving cases completed 972 responses across all four semantic ablations and
ARC controls. Batching/capacity/lease refinements did not establish a gain over
wait-hint ARC. Aggressive rank-4 compression damaged retrieval quality; rank 8
passed the narrow fixture but remained slower and used extra GPU workspace.
[Results, activation proofs and stopping decision](semantic-svd-serving-20261104/README.md).
Subsequent workload review found no public prefetch head start; that small scripted
workload cannot support a broad policy rejection. See its scope correction and
the real executed-tool benchmark above.

## November 3: maintained benchmark infrastructure

The four maintained tiers passed real serving qualification on this host, including
0.5B/7B AWQ public-input graph workloads, CPU wait hints, exact abort/recovery and
failure controls. The selected runs contain 608 normal workload completions;
full regression and offline artifact checks passed. This is harness qualification,
not a new policy-performance or production claim.
[Coverage, turnaround and limits](benchmark-suite-20261103/README.md),
[commands](../docs/BENCHMARKING.md).

## November 2: CPU-only ARC recency advice

On this RTX 2080 Ti with Qwen2.5-7B-Instruct AWQ / FP16, twelve paired workflows
used normal model forwards with recorded output tokens. The candidate used 2,048
GPU blocks versus ARC's 2,112, with the same CPU cache. All 1,728 requests passed
strict recorded input/output checks. Throughput improved 0.64% and serving-load
VRAM fell 56 MiB per pair; mean first-token latency worsened by 233 ms and p95 by
1.43 s. This is conditional replay evidence, not live-generation quality or a
production recommendation. Speculative GPU copies were disabled: the mechanism
was CPU retention advice, not a semantic-ranking win.

[Result and intervals](cpu-wait-recency-2080ti-20261102/result.json),
[paired measurements](cpu-wait-recency-2080ti-20261102/confirmation-pairs.csv),
[tested configuration](cpu-wait-recency-2080ti-20261102/tested-connector-config.json),
[aggregate verifier](cpu-wait-recency-2080ti-20261102/verify_aggregates.py).
The full frozen protocol and raw evidence are local to
`output/prefetch-native-20261102/`.

## November 3: scoped live serving qualification

The lead and independent Astra reviewer support the exact measured profile for
**controlled asynchronous continuation queues producing drafts for human
review** on this RTX 2080 Ti WSL host. It combines GPU-gated CPU ARC retention
advice with an eight-second return-priority credit. It does not qualify general
semantic ranking, interactive chat, autonomous task accuracy, or unrestricted
arrival rates.

Twelve paired whole workflows across three fresh fixed public SWE traces used
Qwen2.5-7B-Instruct AWQ / FP16, 1,536 GPU blocks, 2,048 CPU blocks and eight active
sequences. All 2,592 responses reached natural EOS with actual generated histories
and only remaining-context output limits. Recorded tool observations were quoted
inputs; external tools were not executed and assistant outputs were not replayed.

| Primary comparison: candidate versus native ARC with the same priority | Measured result |
| --- | --- |
| Output throughput | **+6.10%**, paired 95% interval +3.34% to +8.94% |
| Throughput per device GiB | **+6.10%**; same VRAM |
| Mean first-token latency | **485 ms better**, interval 293–677 ms better |
| p95 first-token latency | 924 ms better point estimate; interval includes 180 ms worse |
| Serving VRAM | **Unchanged**, approximately 7.52 GiB |

The intervals are conditional on three fixed traces, four repetitions each;
requests are not independent experimental units. Against native ARC with zero
priority credit, the full profile improved throughput by 11.17%, but initial
request p95 first-token latency worsened by 3.69 seconds. That comparison includes
a generic native scheduling benefit. Fixed priority also affects native
preemption victim selection; it is not an admission or wall-clock fairness
guarantee. Fresh summed process RSS was about 51 MiB higher for the candidate;
shared mappings prevent interpreting that as private memory.

Operational follow-up completed another **5,328 workload and 16 health responses**
to natural EOS. Persistent 24-wave cases exercised the 1,024-prefix and
4,096-acknowledgement caps; pending metadata cleared at every boundary, and final
engine RSS samples were nearly flat. Reduced concurrency, private waits up to
60 seconds, two directly identified native aborts with recovery, registry
exhaustion and finite overlapping bursts were also checked. At an accelerated
128-session limit, 702 advice acknowledgements declined new sessions while all
576 inference requests completed. Default-cap coverage reached 576 identities,
not 4,096. This finite horizon does not prove indefinite stability.

Adverse outcomes remain: overlapping bursts reached about 49 seconds to first
token for a new session and 64.6 seconds end to end. Initial p95 worsened by
497 ms and maximum end-to-end latency by 2.45 seconds versus matched native
priority in that case. The two original endurance abort matchers failed because
they omitted native randomized IDs; those failures remain recorded, with later
direct request-ID qualification separate. Two documentation edits stopped earlier
freezes; no failed case was replaced or relabeled. Shared model source-reading
and reasoning defects remain, so no quality-equivalence or task-success gain is
claimed.

[Exact profile and operating contract](live-optimization-2080ti-20261103/qualified-live-profile.json),
[fresh confirmation](live-optimization-2080ti-20261103/priority-confirmation-root-verification.json),
[final lead assessment](live-optimization-2080ti-20261103/operational-final-root-assessment.json),
[independent Astra assessment](live-optimization-2080ti-20261103/operational-final-astra-assessment.json),
[byte qualification](live-optimization-2080ti-20261103/return-priority-bytes-root-verification.json).
The byte qualification checked 29,120 component/corruption comparisons with
transfer-count and aggregate-byte reconciliation; per-job bytes were derived
from mapped pages rather than independently observed native job-byte fields.

Qualification binds the measured installed wheel and five-file overlay. The
concurrent vendored developer launcher and a clean installation have separate
unverified GPU-serving paths. Subsequent concurrent benchmark-telemetry edits
are [outside the measured frozen source](live-optimization-2080ti-20261103/postqualification-source-boundary.json).
Keep the exact linked configuration, causal hint
producer, one gateway priority epoch, and explicit 4,096-session registry-lifetime
management. New-session advice declines at capacity while inference continues;
generation records do not expire automatically. See [configuration](../docs/CONFIGURATION.md).

The [protocol](live-optimization-2080ti-20261103/protocol.json), rejected candidates,
frozen scripts and compressed raw evidence remain in the evidence directory and
ignored `output/prefetch-native-20261102/`. The earlier replay's 56 MiB saving is
not a live-generation memory-saving result.

## Earlier findings

- [September results and provenance](september-2026.md): E04 admission and A27
  combined retention/prefetch. Threshold-2 admission helped generically; semantic
  policy did not establish a repeatable benefit over matched controls.
- [Brief history](../docs/PROJECT_HISTORY.md): October/November scoring and
  broader-workload investigations, with locations of local detailed reports.

For new results, keep the tested hardware/model, policy activation, comparator,
memory denominator, generation mode, trade-offs, and reproduction path alongside
the claim. Store compact evidence here and bulky artifacts under `output/`.

## Repository workflow finalization

[Clean installation, shared GPU locking, validator promotion and lightweight CI](workflow-finalization-20261103/README.md). Local checks passed; remote CI awaits a future push.

## Hybrid qualification evidence

[MSR and richer Q/K pilot, source audits and native restore diagnostics](msr-hybrid-qualification-20261103/README.md).
The primary pilot was inconclusive; exploratory quarter-second representation
gains remain hypothetical. Qualification gates are unmet and no live hybrid was built.

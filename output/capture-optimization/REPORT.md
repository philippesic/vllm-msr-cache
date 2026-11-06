# Query capture optimization and bounded admission investigation

Completed the authorized investigation. Astra medium reviewed the capture path and bounded-admission design. The slice/concatenate candidate passed correctness but did not show a serving speedup, so the prior measured implementation and tests were restored. The candidate, tests and measured negative evidence remain saved outside Git. No admission policy or proactive GPU-reclamation change was deployed.

## Bounded admission: answers to the policy questions

- **How much:** use the minimum of available DRAM storage/reservations, the paused session's remaining fair share, remaining write-service tokens, and bytes that can finish within a conservative current hint lifetime. Deduplicate existing/pending copies; preserve demand-copy headroom. GPU staging has its own live-free-block budget. The current data do not identify optimal write-rate or headroom settings; these require later matched policy trials, not an unexplained percentage.
- **Which blocks:** preserve completed exact consecutive-prefix segments from the earliest missing position. Leading GPU coverage can only be relied on if it remains valid through consumption; the prior threshold0 test lost its leading GPU prefix. Use current wait evidence and past reuse/cost, not future tool-return labels. Semantic scores cannot authorize inexact KV reuse or make isolated suffix blocks into a usable prefix.
- **When GPU blocks can go:** only through native ownership/cache machinery, after the DRAM copy fence is complete and no active request references the source. Completion of a CPU store does not release GPU references. Completed-request blocks may already be reclaimable in the native free queue while retaining cached hashes. Start with ordinary pressure-driven eviction; an explicit reclaim-on-wait policy would require a separate ownership-safe implementation and validation.

Actual scale: DRAM holds199blocks at2.25MiB/16tokens each. Without a reserve, two competing sessions could receive99blocks each, three66each, six33each (74.25MiB/528tokens). Prior median per-request native eligible range was157blocks. These are arithmetic constraints, not observed simultaneous demand or tuned policy settings. See [design and safeguards](BOUNDED-ADMISSION.md) and [source arithmetic](admission-budget-arithmetic.json).

## Query-capture candidate

The candidate replaced CPU-list-to-CUDA index creation plus advanced indexing with basic row slices and a concatenation. It retained the same request rows, dtype/device, head grouping and capture cadence, with owned selected storage. It also consumed the per-execution layout before callback reentry. Astra rejected more complex reusable pinned-buffer or native metadata-index approaches for this first experiment because of lifetime/stream/speculative-row risks.

Four fresh Qwen3-4B serving arms reused the exact45-request simulated agent load, fixed1440MiB GPU KV pool,448MiB DRAM,threshold2/ARCretention/flat search. Two observer-off stride8 opportunity-scoring arms measured product behavior; two stride1 continuous-scoring profiled arms investigated mechanism. One pair each, not a significance estimate; closed-loop arrivals/cache histories and executed prompt work differ.

| Observer-off product | Baseline | Candidate |
|---|---:|---:|
| Output tokens/s | 82.06 | 80.46 |
| Mean release→first-token, ms | 4384.2 | 4370.5 |
| Median release→first-token, ms | 2829.2 | 3092.4 |
| Timed load, s | 39.775 | 40.567 |

No product speedup was demonstrated: throughput -1.95%, mean first-token -13.8ms, median +263.2ms. This is not sufficient to establish a repeatable regression either. The profiled stride1 diagnostic throughput increased0.69%, a different configuration and observer setting, not the product result.

The diagnostic revealed wait migration:

| Cumulative host boundary | Baseline | Candidate |
|---|---:|---:|
| Row selection (baseline upload plus gather) | 14.648s | 0.021s |
| Score readback | 0.058s | 11.432s |
| Capture extraction, including nested scoring | 14.979s | 11.718s |

Both diagnostics observed1139capturecallbacks/3273requestrows and1000score calls. The selection span shrinking did not eliminate all waiting: much reappeared at the next blocking readback. These inclusive host spans overlap and must not be added or equated to isolated GPU compute. ThreadCPU migration is consistent with runtime wait/spin, but does not identify the precise driver mechanism. Fine-scope CUDA observer overhead differs between the implementations. Profiles include startup/warmup and complete explicit pre-teardown export; none of the saved events remained unready. No per-step fence was added.

The next hypothesis, if separately pursued, is scheduling/readback synchronization rather than merely speeding index construction. Delaying score readback could affect freshness and callback/EMA semantics; no such change was implemented in this task.

## Validation and disposition

284nativeCPUchecks with CUDA forbidden; six tiny BF16CUDAselection/GQA/owned-storagecases;180/180realrequests with exact prompts/outputtokens/nativeorderedkeys passed. All transfer/fence checks passed. Eager execution only: not certification of CUDA graph modes. No competitive semantic ranking or native direct adoption occurred. All model servers stopped and custody is released; original source/runtime/prior experiments remain preserved. No commits or pushes.

The two edited Mac files were restored only after verifying their current bytes against the exact candidate pins and verifying the recovered originals against the frozen measured-v3 manifest. This preserves unrelated existing project work. [Candidate source](candidate-source/semantic_offload/query_capture.py), [candidate patch](results/capture_optimization/candidate.patch), [tests](candidate-source/tests/test_query_capture.py) and source pins remain recoverable.

- [Detailed GPU report](results/capture_optimization/REPORT.md), [four-arm metrics](results/capture_optimization/arms.csv), [per-request CSV](results/capture_optimization/measurements.csv), [components](results/capture_optimization/components.csv)
- [Astra review](astra-review.md), [reproduction](results/capture_optimization/REPRODUCE.md), [local verification](VERIFICATION.json)
- [Final native audit](results/capture_optimization/evidence/final-audit.json), [release](results/capture_optimization/evidence/release.json), [storage inventory](STORAGE.json)

No demonstrated memory-capacity benefit or optimal admission budget is claimed. Private-stream DMA, exclusive attention kernels, pure network time, startup-excluded allocation peak and repeated-trial noise remain unmeasured.

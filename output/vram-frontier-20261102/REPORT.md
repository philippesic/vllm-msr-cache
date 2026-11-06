# ARC versus semantic: GPU cache frontier

The broad screen suggests a useful knee near **1,792 MiB (1.75 GiB) of GPU KV cache** for this workload. Semantic prefetch did **not** extend the usable low-memory range. Both systems degrade below that region, with semantic slower and higher-latency at the two smallest valid paired budgets. This is a coarse optimization map, not an established production optimum.

[Primary frontier plot](vram-frontier.png) · [Highest-observed-ARC sensitivity](vram-frontier-highestARC-sensitivity.png) · [CPU/latency/transfer costs](vram-operational-costs.png). SVG versions are included beside each PNG.

**Measured spread.** Eight budgets, both arms, official 4-bit Qwen2.5-7B on the same RTX 2080 Ti/WSL host. Fixed 3.5 GiB CPU cache, 24 public agent sessions, up to eight active requests, three actual-output turns, EOS respected/96-token output cap, and simulated private 0–12 s tool waits. No padding or future-text/deadline oracle. Same frozen scoring, shared-prefix/JIT implementation and cache settings apart from GPU pool size. ARC restores on demand; semantic attempts earlier restoration. Source/model/data/commands are pinned. One corpus partition provides the spread; earlier three-partition results are a separate cross-check.

All 16 points were attempted: **14 passed, totaling 1,008 strict request proofs; two are validation holes**. Invalid timings do not count as curve points. Values below are whole finite-load output throughput and client p95 first-token latency. Each row is one paired screen, not a repeated estimate.

| GPU KV pool MiB | ARC tok/s | Semantic tok/s | ARC p95 TTFT s | Semantic p95 TTFT s |
|---:|---:|---:|---:|---:|
|672|48.02|41.36|49.15|63.57|
|896|56.81|54.94|38.94|41.34|
|1,120|64.45|invalid|27.55|invalid|
|1,344|invalid|70.43|invalid|22.81|
|1,792|79.77|79.01|19.84|20.39|
|2,240|81.29|79.79|20.00|20.24|
|2,688|82.04|81.31|19.85|20.30|
|3,584|79.89|78.86|20.24|20.36|

At 672 MiB semantic is about 13.9% slower than matched ARC, with first-token latency about 29% higher. At 896 MiB it is about 3.3% slower. At 1,792 MiB it is about 1.0% slower. These are descriptive workflow comparisons: generated outputs/later histories differ at active-scoring pairs, so the original exact-work 1% goal is not established. All points miss the prior absolute 15 s agent first-token limit; the burst workload does not demonstrate sustainable production capacity.

**Relative targets, not a single optimum.** Thresholds were chosen before the screen: retain 99%, 97% or 95% of high-budget ARC throughput, with p95TTFT at most 5% worse; 10% latency tolerance is separately reported. Smallest *valid measured* GPU KV budgets satisfying the 5% version are:

| Reference | Retained throughput | ARC MiB | Semantic MiB |
|---|---:|---:|---:|
|Predeclared largest-budget ARC|99%|1,792|2,240|
|Predeclared largest-budget ARC|97%|1,792|1,792|
|Predeclared largest-budget ARC|95%|1,792|1,792|
|Posthoc highest-observed ARC|99%|2,240|2,688|
|Posthoc highest-observed ARC|97%|1,792|2,240|
|Posthoc highest-observed ARC|95%|1,792|1,792|

The largest-budget ARC result (79.89 tok/s) is below the highest observed ARC result (82.04), a difference larger than the 1% target. That makes precise threshold rankings reference-sensitive. The primary normalization remains unchanged; the second is explicitly a sensitivity check, not a capacity estimate. Missing valid points and single-pass variability prevent locating an exact continuous knee. Plots use measured scatter and explicit invalid markers, with no fitted/interpolated claims.

**Functionality and costs.** Semantic numerical calls/native adoptions at valid budgets, in ascending pool order: 73/1, 64/2, 42/4, 27/5, 1/0, 0/0, 0/0 (the invalid semantic 1,120 MiB point is excluded). Low-budget tests really exercised scoring/prefetch; their slowdown is not an inactive-scoring control. Conversely, the semantic points that satisfy the 99% targets have no native prefetch adoption. They do not demonstrate a prefetch benefit. Host CPU, RSS/paging, demand/speculative transfer bytes, reservations/waste, preemptions, actual CUDA/NVML memory and full latency/goodput remain in the machine-readable results. No new scoring-quality equivalence is claimed; prior agreement evidence remains inconclusive.

At 672 MiB, serving CPU cost is 21.83 versus 25.45 CPU-seconds per 1,000 output tokens (ARC/semantic). Semantic completes 4.32 GB of speculative transfers but its intended requests consume only 0.137 GB, about 3.2%; request-owned reservations peak at 130 MiB inside the fixed 672 MiB pool. At 1,792 MiB, CPU cost is 13.16 versus 13.46 CPU-seconds per 1,000 output tokens, and intended consumption is about 51% of 1.234 GB speculative traffic. This identifies transfer usefulness and reservation cost as concrete optimization concerns; it does not isolate their causal contributions to slowdown.

Actual peak CUDA allocation at 672/1,792/3,584 MiB pools is approximately 6.564/7.739/9.618 GB for ARC and 6.573/7.747/9.627 GB for semantic. Thus shrinking from the largest pool to the knee makes about 1.88 GB of allocator space available in either system. Endpoint memory includes weights/runtime; these are decimal GB, distinct from configured MiB. Host RSS at these points remains roughly 7.2–7.3 GB. Small paging observations are retained in the results, not silently discarded.

**Recommended optimization target.** Use **1.75 GiB KV as the present practical anchor**. A useful next objective is **1.3125 GiB KV at at least 99% of ARC's 1.75 GiB throughput—about 79 tok/s—with p95TTFT no more than 5% worse**, strict correctness and meaningful scoring/adoption. That would free another 448 MiB of KV space relative to the ARC knee. Current semantic at that smaller budget delivers only 70.43 tok/s, about 88% of the anchor. This is an investigator-selected target, not an achieved result. Reducing unconsumed staging/reservation cost while preserving active capacity is a better direction than assuming more prefetch will help. No causal ablation was performed in this sweep.

GPU-pool reductions free real device allocation at startup; weights and runtime still occupy GPU memory. ARC provides the same configurable saving. Pool percentages are not total-VRAM percentages, and this 7B host result does not predict a 32 GB/27B endpoint's concurrency.

**Verification and scope.** The first 1,024-block semantic attempt failed 3/72 lookup proofs. One common observer-only fix, qualified on actual pending-transfer no-query deferrals and eight preemptions, then passed 72/72; failed raw remains invalid. Later semantic 1,280-block and ARC 1,536-block runs each failed one proof at the strict aggregate physical-destination uniqueness guard. Nonoverlapping reuse is not a full correctness proof: no validator waiver/second repair, and both remain holes. The initial valid 2,048-block pair retains its older observer identity; each budget pair uses matched observer versions. No scoring/policy/ownership tuning occurred.

No additional GPU confirmation was justified by a new advantage. The previous three paired 2,048-block runs use identical frozen source/runtime/model/corpus/math/budgets/workload; observer/submission-ledger/cap/walltime differences are explicit. Their semantic/ARC throughput ratios were 1.0228, 0.9752 and 1.0110, with unequal histories. They provide an external cross-check of a small, variable effect, not fresh pooled confirmation. See [comparability receipt](EXTERNAL_REPLICATION.json) and [scope decision](FINAL_DISPOSITION.json).

The intended 16 screen run lifecycles took **41.26 min**, including **27.11 min traffic** and **11.47 min startup/warmup**. Elapsed time from first qualification to final measurement was **73.11 min**; diagnosis/repair/planning, CPU checks, export and gaps explain the difference. Original failed-attempt/qualification timing is separately retained, rather than counted as accepted performance. Final plotting/report/cleanup followed. [Duration receipt](DURATION.json).

[Numeric results](SCREEN_RESULTS.csv) · [full results and mismatch inventories](SCREEN_RESULTS.json) · [thresholds](THRESHOLDS.json) · [reproduction](REPRODUCE.md) · [detailed methods and failures](RUNTIME_REPORT.md). Final CPU controls/analysis: **28 tests passed**. [Operational validation](FINAL_VALIDATION.json), [frozen evidence pins](FINAL_PINS.json) and [root review](ROOT_REVIEW.json) distinguish source/derived-data verification from intrusive byte qualification and performance.

All owned services/processes/listener/GPU lock were released. Scoped cleanup removed 272,186,199 logical bytes of disposable plotting packages/cache/bytecode and reduced evidence size losslessly by a further 106,791,029 logical bytes. Measured WSL free space rose from 923,820,843,008 to 924,075,847,680 bytes (observed +255,004,672; concurrent activity can affect this). Retained approximately **1.88 GB useful evidence** and the unchanged **5.58 GB reusable model cache**; prior campaigns and canonical dirty work remain protected. No Windows backing-volume reclaim claim, serving-venv change, commit or push. [Storage receipt](STORAGE.json).

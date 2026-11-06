# RTX 2080 Ti ARC and semantic agent workload results

Measured on October 31 2026. With the repository’s representative generated agent workloads, full continuous semantic eviction plus enabled wait-hint prefetch is slower than ARC: **27.4% lower throughput for one agent and 19.2% lower for six agents**. The normal matrix completed all 378 requests and generated 28,800 tokens. This configuration has not demonstrated the project’s intended gain in useful exact cache reuse, latency, or resource efficiency.

The features being enabled does not mean they had a useful opportunity. Normal runs had no CPU cache eviction and no submitted or consumed prefetch. A separate admission-stress matrix did exercise scored semantic eviction, but still produced no prefetch and no advantage. These are representative simulations of coding and investigation sessions, with generated tool outputs and background traffic; they are not production traces or measurements of an actual agent’s task success.

## Primary workload matrix

Each entry pools three fresh-server runs at seeds 1103, 2207 and 3301. Throughput is total completed output tokens divided by summed workload duration; latency percentiles pool individual requests rather than averaging per-run percentiles. TTFT means time from the request’s intended release to its first received token. ITL means the mean received intertoken interval; tokens delivered together in a network chunk share a receipt time, so this is client-observed streaming latency rather than isolated GPU token time. Smaller latency is better; larger throughput is better.

| Workload | Policy | Requests | Output tokens/s | Mean TTFT s | Median TTFT s | p95 TTFT s | Mean ITL ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | arc | 54 | 64.36 | 2.07 | 1.82 | 4.92 | 33.82 |
| single | semantic | 54 | 46.76 | 3.48 | 3.20 | 8.94 | 45.42 |
| multi | arc | 135 | 58.77 | 6.60 | 5.34 | 15.45 | 41.60 |
| multi | semantic | 135 | 47.50 | 8.67 | 6.33 | 21.49 | 54.75 |

| Workload | Semantic throughput change | Mean TTFT change | p95 TTFT change | ITL change | Throughput change across paired seeds |
| --- | --- | --- | --- | --- | --- |
| single | -27.36% | 68.35% | 81.61% | 34.27% | -28.51% to -25.41% |
| multi | -19.17% | 31.38% | 39.07% | 31.61% | -20.68% to -17.64% |

Both workloads include the same twelve independent background requests per run. “Single” means one six-turn agent session plus those requests. “Multi” means six agent sessions, with five completing six turns and one abandoning after three turns, plus background traffic. Max active GPU sequences is three, so six agents create genuine queuing. The normal matrix contains 54 requests per policy for single and 135 per policy for multi. All three paired seeds favor ARC for throughput. The sample is small: the p95 values are descriptive tails, not a service guarantee or a significance test.

## Agent and background latency

| Workload | Policy | Role | Requests | Mean TTFT s | p95 TTFT s | Mean request elapsed s | Mean queue s | Mean service s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | arc | agent | 18 | 1.23 | 3.41 | 3.31 | 1.06 | 2.24 |
| single | arc | background | 36 | 2.49 | 5.17 | 5.73 | 2.38 | 3.33 |
| single | semantic | agent | 18 | 2.01 | 8.93 | 4.80 | 1.81 | 2.99 |
| single | semantic | background | 36 | 4.22 | 8.20 | 8.56 | 4.10 | 4.46 |
| multi | arc | agent | 99 | 5.29 | 14.73 | 8.02 | 4.66 | 3.34 |
| multi | arc | background | 36 | 10.19 | 15.84 | 13.84 | 10.03 | 3.80 |
| multi | semantic | agent | 99 | 6.92 | 21.24 | 10.49 | 6.25 | 4.23 |
| multi | semantic | background | 36 | 13.47 | 22.14 | 18.34 | 13.30 | 5.02 |

| Workload | Policy | Mean load duration s | Mean agent session completion s | Max agent session completion s | Mean release lateness ms |
| --- | --- | --- | --- | --- | --- |
| single | arc | 23.86 | 23.86 | 24.90 | 1.43 |
| single | semantic | 32.85 | 32.85 | 34.21 | 1.35 |
| multi | arc | 55.54 | 48.18 | 57.31 | 1.96 |
| multi | semantic | 68.71 | 61.77 | 69.58 | 1.90 |

Agent arrivals are closed loop: a later turn follows the prior completion and its actual tool wait. Slower serving therefore changes batch and cache histories. Background arrivals are fixed relative to workload start. Session completion includes intervening waits and is measured from workload start; the abandoning session ends at its last successful LLM completion in this table. Request elapsed is submit to stream completion. Server queue and service timings come from preserved scheduler/connector records; they are not substitutes for client TTFT.

## Cache work and feature activation

| Workload | Policy | Executed prompt tokens | GPU hit tokens | CPU restored tokens | CPU stored blocks | CPU evicted blocks | Query captures | Rows in score updates | Prefetch jobs ready or consumed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | arc | 35,114 | 36,528 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| single | semantic | 35,114 | 36,528 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| multi | arc | 234,018 | 56,400 | 656 | 29 | 0 | 0 | 0 | 0 / 0 |
| multi | semantic | 240,530 | 49,904 | 640 | 31 | 0 | 2780 | 1522 | 0 / 0 |

| Workload | Policy | Demand H2D MiB | D2H MiB | Reported H2D worker ms | Reported D2H worker ms | Speculative H2D MiB | Preemptions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | arc | 0.00 | 0.00 | unavailable | unavailable | 0.00 | 0 |
| single | semantic | 0.00 | 0.00 | unavailable | unavailable | 0.00 | 0 |
| multi | arc | 17.94 | 12.69 | 984.313 | 1.327 | 0.00 | 0 |
| multi | semantic | 17.50 | 13.56 | 38.164 | 5.916 | 0.00 | 0 |

H2D means host CPU memory to GPU; D2H is the reverse. Byte totals and reported worker durations sum terminal transfer attribution and can include repeated movement; they are not unique cache footprint or exclusive critical-path time. Transfer durations are unavailable when no job occurred. H2D durations vary substantially across ARC seeds (75, 696 and 213ms), while semantic totals are 11–14ms per seed; these CUDA event intervals are affected by the run’s execution/stream history and do not establish an isolated copy-bandwidth advantage. The normal single-agent run retains returning prefixes on GPU and never populates the CPU cache. Continuous scoring skips extraction when its durable CPU summary bank is empty, but the dispatch mode still surrounds model execution. Multi-agent runs restore a small amount of CPU KV. Their CPU caches never fill, so semantic eviction cannot improve retention here. Semantic executes 2.78% more prompt work in multi despite the same paired inputs: differing closed-loop scheduling/cache histories are part of the measured outcome, rather than a fixed-batch microbenchmark.

| Workload | Policy | Peak ready CPU blocks | Minimum free GPU blocks | Peak referenced GPU blocks | Peak speculative reservations |
| --- | --- | --- | --- | --- | --- |
| single | arc | 0 | 372 | 267 | 0 |
| single | semantic | 0 | 381 | 258 | 0 |
| multi | arc | 10 | 62 | 577 | 0 |
| multi | semantic | 11 | 47 | 592 | 0 |

Across all ordinary and stress runs, speculative nominations with candidates, submitted staging jobs, ready staging jobs, consumed staging blocks, GPU splice bytes, and speculative waste bytes are zero. No competitive semantic prefetch selection occurs. Prefetch lead time, fence-before-forward, delivery latency and waste efficiency are therefore unavailable as performance observations, not zero-duration successes. The full implementation was enabled; its usefulness under an actual prefetch opportunity remains unestablished on this host.

Query captures count worker callback batches. Rows in score updates count rows delivered to the CPU manager, including repeated updates; they do not count every worker vector comparison. Detailed component counters retain those comparisons in the separate profiled runs.

## Admission stress matrix

The frozen protocol calls for this supplement if the normal matrix has no CPU eviction or no consumed prefetch. It changes only the admission threshold from two to zero in both arms at seed 1103, admitting one-use blocks. All other model, memory and workload controls remain unchanged. This diagnostic deliberately increases churn and is excluded from normal headline results.

| Workload | Policy | Requests | Output tokens/s | Mean TTFT s | CPU stored blocks | CPU evicted blocks | Semantic eviction calls with score state | CPU restores | Prefetch consumed blocks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | arc | 18 | 69.30 | 2.09 | 832 | 633 | 0 | 0 | 0 |
| single | semantic | 18 | 43.80 | 4.24 | 1013 | 814 | 16 | 0 | 0 |
| multi | arc | 45 | 56.70 | 6.99 | 6052 | 5853 | 0 | 0 | 0 |
| multi | semantic | 45 | 42.05 | 9.80 | 6052 | 5853 | 110 | 0 | 0 |

Semantic throughput changes by -36.79% versus ARC in single. Its CPU cache reaches the full 199-block capacity. This is actual semantic eviction with nonempty scores, not merely a policy configuration flag.

Semantic throughput changes by -25.84% versus ARC in multi. Its CPU cache reaches the full 199-block capacity. This is actual semantic eviction with nonempty scores, not merely a policy configuration flag.

For the semantic multi stress run, hint responses include 60 prefixes already on GPU, 23 without a ready contiguous CPU prefix, and six expired nomination deadlines, alongside registration and unknown-event responses. No hint reaches a competitive selection. The code checks GPU coverage and the next ready exact CPU prefix before staging. This points toward useful prefix availability and admission/retention as a prerequisite to testing ranking quality. It does not prove one eviction rule caused every missed opportunity or that a faster search index would recover them. Removing the admission guard produces substantial CPU write traffic without useful restores in this trace.

## CPU memory GPU and energy

| Workload | Policy | Sampled CPU s | Peak summed RSS GiB | RSS GiB seconds | Mean GPU utilization % | CUDA allocated peak MiB | CUDA reserved peak MiB | Device peak GiB | Device energy kJ | J per output token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | arc | 72.33 | 3.34 | 236.81 | 31.56 | 3,411.49 | 3,448.00 | 4.65 | 8.35 | 1.812 |
| single | semantic | 98.22 | 3.31 | 325.00 | 24.91 | 3,411.49 | 3,448.00 | 4.65 | 9.72 | 2.110 |
| multi | arc | 175.87 | 3.34 | 550.20 | 53.68 | 3,411.49 | 3,458.00 | 4.66 | 26.13 | 2.669 |
| multi | semantic | 218.33 | 3.35 | 687.28 | 44.87 | 3,411.54 | 3,458.00 | 4.66 | 28.99 | 2.960 |

These totals cover the three measured load windows per cell. CPU/RSS and NVML samples are collected approximately every 250ms. Sampled process CPU is an estimate with recorded boundary overhang; short-lived or non-server work can be missed. Summed RSS can double-count shared pages. CUDA peaks reset after warmup and measure PyTorch allocations/reservations, while device memory includes driver/display allocations. Device energy integrates whole-device sampled power and includes non-model activity. Clocks use normal dynamic scaling, not locked frequencies. Raw power, clocks, temperature, utilization and process samples are preserved, so these are resource indicators rather than an isolated hardware-efficiency or provider-price claim. There is no measured VRAM or useful-capacity win.

## Separate component profiles

These four runs enable additional instrumentation and remain outside primary throughput and latency pooling. Instrumentation can enlarge or change the gap. Runner execution wall time and thread CPU time include nested preparation, Python dispatch, launches and callbacks; they overlap other reported components and must not be added together.

| Workload and policy | Load s | Runner calls | Runner wall s | Runner CPU s | Captured query batches | Dispatch interceptions | Score wall s | Preservation logging wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi-arc | 55.57 | 1160 | 37.03 | 36.90 | 0 | 0 | 0.000 | 0.249 |
| multi-semantic | 71.26 | 1156 | 65.73 | 65.41 | 917 | 709551 | 12.438 | 0.236 |
| single-arc | 22.46 | 629 | 19.84 | 19.69 | 0 | 0 | 0.000 | 0.083 |
| single-semantic | 34.59 | 687 | 31.18 | 31.02 | 0 | 416544 | 0.000 | 0.078 |

The single-agent profile has no extracted query batches or score work, yet semantic runner execution consumes substantially more wall and CPU time and records 416,544 dispatch interceptions. Source inspection confirms that the empty-bank check occurs inside capture after model execution has already entered TorchDispatchMode. That makes bypassing capture machinery when no score can be consumed a strong candidate for a controlled ablation. It does not attribute the entire slowdown to the dispatch body or establish the speedup a change would deliver. Preparation, metadata, scheduler decisions, transfer polling, scoring, logging and sampled CUDA stream scopes are retained in components.csv and profile-detail.json. CUDA scopes sample every 16 calls and overlap host/other stream activity; their sums are neither exclusive attention-kernel time nor total GPU busy time. All final synchronized exports have no pending profile CUDA samples or async observations.

The multi-agent semantic profile also spends 12.44s inside query scoring during its 71.26s load, despite the absence of eviction or prefetch consumption in that run. Scoring and capture therefore need separate cost ablations; the data do not justify attributing the whole gap to one component.

## Output integrity and reliability

Primary output lengths, successful completion and authoritative terminal coverage all match the fixed budgets. Every primary request has complete ordered native keys and physical block/table evidence. Transfer jobs in completed matrix cells report no failure. Exact paired outputs match for 183 of 189 primary request pairs. All 135 multi-agent pairs match exactly.

The exception is single seed 2207. Its first agent request has identical input tokens, no GPU prefix hits and no external restore in either arm, but output first differs at token 58. Five later turns inherit different generated history. All twelve independent background outputs match. This initial divergence cannot have come from consuming an incorrect restored CPU cache on that request. Ordinary FP16 batching/numerics is a hypothesis; cache and inference correctness are not certified solely by that observation.

| Separate comparison | Requests | Same prompts | Same outputs | Same ordered key chains |
| --- | --- | --- | --- | --- |
| main-single-2207-arc versus repeatability-single-2207-arc | 18 | 13 | 12 | 13 |
| main-single-2207-semantic versus repeatability-single-2207-arc | 18 | 18 | 18 | 18 |
| main-single-2207-arc versus main-single-2207-semantic | 18 | 13 | 12 | 13 |

The one separately declared ARC-only repeat also diverges from the original ARC output, while matching all 18 semantic prompts, outputs and ordered key chains exactly. This establishes baseline output instability in this ordinary FP16 test setup and supports treating cross-arm identity carefully. It does not prove every semantic path correct. The repeat remains separate and never replaces or contributes to the headline samples.

The first ARC multi profile had one immediate ServerDisconnectedError: 42 requests succeeded, one failed, and two dependent turns were never submitted. Other sessions continued and the engine exported successfully; a stale keep-alive connection is plausible but unproven. The first semantic multi profile completed all 45 requests, then failed in the post-load log reader on a partial JSON record before synchronized export. Both attempts remain under failed-attempts and are excluded from serving aggregates. ARC was repeated unchanged; semantic was repeated with only a tolerant post-load reader. A launcher import-path error before server startup is also preserved in continuation2.log. No primary run was replaced. The final successful matrix comprises 21 cells and 648 requests including stress, profiles and the separate repeat.

The selected CPU regression suite passes 419 tests with CUDA initialization forbidden, using the same task-local runtime overlay and passive wrappers. Earlier failed guard/runtime setup attempts are retained alongside the final receipt. This is targeted regression evidence, not a production release qualification. See VALIDATION.json, repeatability.json, DIAGNOSTIC-RETRY.md and cpu-validation-final.log.

## Configuration and comparability

The model is Qwen2.5-1.5B-Instruct, FP16, TP1, eager Triton attention, V2 runner, WSL mapped pinned memory. Both policies use max model length 8192, three running sequences, 1024 batched tokens, 16-token blocks, 640 GPU blocks (280 MiB KV) and 199 CPU blocks (87.0625 MiB KV). The native null block reduces usable GPU entries by one. These deliberately small controlled KV pools create pressure; they do not use the whole 11 GiB GPU as cache or measure a memory-capacity frontier. Weights and all other allocations sit outside those KV budgets.

ARC uses the stock ARCCachePolicy through the common instrumented connector, with query capture/scoring and speculative prefetch disabled. Semantic uses the real SemanticPolicy with mean keys/heads, default alpha 0.5 blend, continuous synchronous flat scoring at stride one, and semantic wait-hint selection with a 5% staging allowance inside the same GPU pool. Both use admission threshold two in the main matrix. Safe prefix-preserving staging is enabled as an experimental opt-in; native adoption is disabled because the current BF16-only guard excludes this FP16 GPU. Thus this tests the full available FP16 semantic eviction/prefetch configuration, not every experimental BF16 delivery path or the pure default allocator option.

The policy receives a causally observed initial exact prefix and a fixed 250ms tool-wait estimate. Actual generated waits of 75–2400ms and abandonment are private to the load generator. It does not receive future tool outputs, actual wait deadlines or future reuse labels. Input preparation, model download, startup, the identical 16-token warmup and post-load synchronized exports are excluded from client throughput. End-to-end closed-loop duration includes tool-idle gaps between requests. Required preservation logging, passive method counters/timing and resource samples are included symmetrically, so these are instrumented serving results.

The installed vLLM prerelease needs retained wait-hint utility forwarding methods. A pinned task-local five-module overlay supplies these and the existing companion block-pool API; the shared wheel and production checkout are unchanged. Runtime, source, model revision, generated inputs, commands and environment are pinned in the evidence. This setup is narrower than broad library compatibility.

The tested source is the copied working checkout, including its pre-existing uncommitted changes. Its local branch tracking state was two commits behind origin/main at capture; no pull or source update was performed during this controlled comparison. These results describe the pinned working state rather than an unverified latest remote revision.

The earlier near-parity result documented in README used ARC retention with opportunistic, sampled semantic prefetch scoring, and a different host/model/workload. It did not establish full semantic retention parity or competitive semantic ranking utility. The new matrix intentionally restores full continuous semantic retention and scoring. Its gap is therefore not a controlled regression from that older number, nor can the older result override these measurements.

## Work needed toward the end goal

The end goal is to save useful exact KV work at equal memory and acceptable latency/CPU cost. The current state provides an integrated experimental path and detailed accounting, but this host’s representative sessions show overhead without a useful prefetch opportunity. The following priorities follow from these measurements; they are proposed work, not changes made during this benchmark.

1. **Make the inactive path cheap.** Ablate dispatch/layout/callback machinery when the durable bank is empty or no decision can consume a score. Preserve lifecycle accounting and compare the full configuration against the cheaper ARC-retention opportunity mode under identical controls. Acceptance requires preserved correctness plus reduced ordinary-workload CPU/latency cost, not only faster microkernels.
2. **Create and preserve useful exact CPU prefixes economically.** Instrument why the next contiguous prefix disappears, then study admission and chain retention while retaining the threshold-two control. Holdout traces must actually have positive CPU-prefix opportunities and real prefetch consumption; measure writes, displacement, misses and downstream prompt work. Do not remove the admission guard simply to make eviction counters increase.
3. **Test whether semantic ranking earns its cost.** Only after multiple eligible sources genuinely compete for a limited budget, compare semantic selection with matched ARC retention, demand-only and simple prefetch controls. Measure useful consumed bytes, ready-before-demand lead time, duplicate/wasted traffic, first-token latency and executed prompt tokens. An isolated faster search index cannot establish ranking utility here.
4. **Resolve numerical repeatability and expand qualification.** Use fixed-history/fixed-batch cache-on/off isolation for the divergent request. Then use longer independent traces and a multi-point equal-memory/service-target capacity comparison, followed by a real application trace. Three short generated seeds and a small 1.5B model do not establish production readiness or paid-task quality.

## Evidence and reproduction

All results and large raw logs remain in this ignored output directory; no production code was changed, committed or pushed. The report tables are generated from pooled-detail.json and measurements.json. Per-request CSV preserves timestamps-derived latency, cache/transfer evidence and nulls for unavailable metrics; per-run raw JSON preserves prompts, outputs, hints, commands, process/device samples and terminal events. Server logs and raw JSON are losslessly compressed with verification receipts.

- [Frozen protocol](PROTOCOL.md), [environment](environment.json), [pins](frozen.json), [runtime overlay](runtime-overlay.patch) and source recovery snapshot.
- [All per-run results](runs.csv), [paired comparisons](comparisons.csv), [per-request metrics](requests.csv), [full aggregates](measurements.json), [pooled role and resource detail](pooled-detail.json).
- [Component metrics](components.csv), [selected profile exports](profile-detail.json), [correctness and configuration audit](VALIDATION.json), [repeatability](repeatability.json), [failed-attempt summaries](failed-attempt-summary.json).
- [CPU validation receipt](cpu-validation-final.json), [diagnostic retry explanation](DIAGNOSTIC-RETRY.md), [output validation addendum](VALIDATION-ADDENDUM.md), [storage and process handback](STORAGE.json).

Use the project virtual environment. setup.py reconstructs the isolated runtime seam; run.py prepares and executes fresh cells, execute.py records the original finite ordering, and continuation scripts preserve the diagnostic recovery sequence. Existing run directories are never overwritten: archive the whole task directory and update the task path for a new campaign before re-execution. analyze.py, postprocess.py and report.py only regenerate derived data/report from preserved evidence. Source pins guard against comparing a changed checkout silently. The separate diagnostic_retry.py alters only the post-load log reader, and should not be confused with a product change.

Storage handback retains approximately 160.7 MiB of evidence and 2.89 GiB of reusable model content. Lossless compression reduced 1,024.1 MiB of raw JSON/logs to 153.8 MiB. Disposable task bytecode was removed; the interrupted profile write contains unique counters and was preserved. WSL free bytes changed from 938,022,993,920 to 938,023,067,648 during cleanup, an observed increase of 73,728 bytes. All sampled owned process identities are gone, the benchmark port is closed, the GPU lock is free and device utilization is zero at handback. Windows backing-volume reclaim was not measured. Shared caches remain available for reuse.

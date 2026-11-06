# Exact gather screen

Four frozen runs completed without operational failure. The gather candidate does not improve throughput over frozen prefix-mean; this single screen does not justify confirmation or adoption. All prompts, native keys and terminal/physical proofs match. Exact outputs fail in three comparisons: ARC640 request16 differs from gather, baseline and ARC400. Gather, baseline and ARC400 match outputs exactly; all-arm exact-output qualification fails. No attribution to gather is established.

| Arm | Tokens/s | Requests/s | CPU s | Return TTFT ms | CUDA allocated/reserved/NVML MiB |
|---|---:|---:|---:|---:|---|
| ARC640 | 31.1203 | 0.7294 | 25.86 | 104.27 | 3411.691/3446.000/4130.688 |
| gather400 | 30.6865 | 0.7192 | 26.52 | 102.36 | 3308.542/3362.000/4046.688 |
| baseline400 | 30.7851 | 0.7215 | 26.66 | 104.18 | 3308.542/3362.000/4046.688 |
| ARC400 | 30.6079 | 0.7174 | 26.50 | 150.71 | 3306.691/3360.000/4044.688 |

| Gather versus | Throughput delta | CPU delta | Return TTFT delta |
|---|---:|---:|---:|
| baseline | -0.320% | -0.525% | -1.748% |
| arc_full | -1.394% | +2.552% | -1.833% |
| arc_small | +0.257% | +0.075% | -32.080% |

Gather and baseline configuration and geometry are identical, with400 GPU blocks/175MiB and2048 CPU blocks/896MiB. ARC400 cache budgets match; ARC640 uses280MiB GPU KV. Source identity differs only as explicitly frozen by the plan. All source/harness pins verified after execution.

Both semantic runs have33 capture calls/36 query rows,12 numerical prefix evaluations,6 accepted/scored candidates and6 native physical-block adoptions. All-fit nomination supplies numerical scoring but no constrained winner-selection quality evidence. No failures, preemptions, completed-prefetch waste, return demand H2D or splice bytes.

[Structured results](gather-screen-results.json) retain actual configurations, memory, latency distributions, all four arms and six exactness comparisons. [Plan](gather-screen-plan.json), zero-exit receipt and execution log are retained. No retries, source edits, run relabeling or evidence exclusions.

# Prefix-mean follow-up confirmation and diagnostics

The repeated short workload does not meet the predefined every-pair throughput target. All six pairs, including the original failed pair, are retained. Separate constrained quality diagnostics show meaningful numeric selection but only 2/5 top-choice agreement with the full-cadence block-EMA reference. Exact KV and output correctness passes throughout.

## Six paired short confirmations

The combined analysis contains21 performance runs: six candidates, six large ARC controls, six matched-budget ARC controls, and three original semantic references. Quality and component diagnostic runs are excluded.

| Role | Runs | Tokens/s | CPU s/run | Overall TTFT mean/p95 ms | Return TTFT mean/p95 ms | CUDA allocated/reserved/NVML MiB |
|---|---:|---:|---:|---|---|---|
| arc_full | 6 | 30.9154 | 26.238 | 410.69/843.42 | 102.38/114.46 | 3411.691/3446.000/4130.688 |
| candidate | 6 | 30.5287 | 26.688 | 414.16/841.93 | 106.42/116.94 | 3308.542/3362.000/4046.688 |
| arc_small | 6 | 30.5658 | 26.552 | 427.18/839.89 | 153.93/194.61 | 3306.691/3360.000/4044.688 |
| original | 3 | 21.0383 | 38.387 | 590.32/1038.38 | 306.32/460.42 | 3315.863/3366.000/4050.688 |

Candidate throughput is -1.251% versus large ARC, -0.121% versus matched ARC, and +45.111% versus original semantic. CPU is +1.715%, +0.515%, and -30.475%, respectively. Return TTFT is +4.041ms (+3.947%) versus large ARC, -47.512ms versus matched ARC, and -199.902ms versus original. Mean inter-token latency is 32.387ms candidate, 31.942ms large ARC, 32.155ms matched ARC, 47.930ms original.

Candidate and matched ARC use 400 GPU blocks/175MiB and 2048 CPU blocks/896MiB. Large ARC uses 640 GPU blocks/280MiB and the same CPU budget. Actual peak allocated/reserved/NVML reductions versus large ARC are 103.150/84/84MiB; candidate versus matched ARC adds 1.850/2/2MiB.

| Pair | Candidate / large ARC % | Candidate / matched ARC % |
|---|---:|---:|
| short-1103-r0 | 97.4592 | 98.6302 |
| short-2207-r0 | 99.2540 | 100.5670 |
| short-3301-r0 | 99.5998 | 100.4958 |
| short-1103-r1 | 98.8065 | 100.5224 |
| short-2207-r1 | 98.5943 | 100.2016 |
| short-3301-r1 | 98.7980 | 98.8750 |

Only 2/6 large-ARC pairs and 4/6 matched-budget pairs meet 99%. Pooled matched throughput is within 1%, while pooled large-ARC throughput is outside 1%. All 21 runs pass integrity and all 30 same-seed comparisons preserve exact outputs and ordered native keys. All six candidates exercise numerical prefix scoring, verified ARC retention/semantic ordering, and native physical-block adoption. These performance runs still contain no constrained competing selection.

## Quality diagnostic: descriptive policy agreement only

Three independent seeded diagnostic runs use a 1536-block constrained GPU budget, 250ms post-peer gap and full-cadence shadow observer. They are excluded from all timing, CPU and memory performance comparisons. Reference: existing mean/mean block ranks, exact compose_relevance_updates and manager EMA, averaged over the same ready prefix keys, on the deployed ARC trajectory.

| Seed | Worker batches / composed key updates | Scheduler observations / applied key updates | Actual multi-candidate decisions | Fully comparable constrained decisions | Top1 agreements | Native adoptions |
|---|---|---|---:|---:|---:|---:|
| 1103 | 682 / 3146764 | 682 / 3146764 | 3 | 2 | 1 | 4 |
| 2207 | 679 / 3137521 | 679 / 3137521 | 3 | 1 | 0 | 5 |
| 3301 | 682 / 3146764 | 682 / 3146764 | 3 | 2 | 1 | 4 |

All worker/scheduler errors are empty, every observer reports valid=true, and cross-process delivery is positive. All 5 actual constrained decisions have full numeric comparison coverage; top1 and full-order agreement are both 2/5 (40%). Four additional staging decisions are unconstrained and have raw prefix scores absent (`None`), though shadow key coverage is complete; they remain explicitly incomparable, not counted as agreement or disagreement.

All three diagnostics pass run integrity and all 12 comparisons against both repeated same-seed large/matched ARC controls preserve exact prompt/output token IDs and native ordered KV keys. No physical/terminal proof is waived.

This is a small descriptive agreement sample, not ground-truth reuse quality or a counterfactual original-retention evaluation. The new scorer preserves exact KV identity but changes ranking policy; 40% agreement is a material observed trade-off. Added observer work changes staging/latency, so diagnostic native adoption counts and delays are descriptive only.

## Current-prefix component profile

The explicitly authorized candidate400 versus largeARC640 short seed1103 pair ran under phase diagnostic, activating the existing component profiler. Both paths were verified absent before execution; neither was overwritten. Both runs pass integrity and the pair preserves exact outputs/native keys. Their raw ordinary-diagnostic performance_eligible field is true, but the explicit plan and assessment exclude all timing from confirmation.

Totals below are inclusive, overlap, and must not be added into an end-to-end total. Values are host wall/thread CPU milliseconds across the profile load window.

| Component | Candidate calls | Candidate wall/CPU ms | ARC calls | ARC wall/CPU ms |
|---|---:|---|---:|---|
| model.execute_host | 624 | 18885.410/18709.871 | 624 | 18547.528/18427.716 |
| summary.build_job | 66 | 34.746/33.808 | 36 | 0.185/0.127 |
| transfer.submit_store | 66 | 109.257/108.286 | 36 | 50.193/50.062 |
| capture.prepare_layout | 594 | 539.219/459.353 | 594 | 486.876/425.610 |
| capture.extract | 594 | 16.426/14.983 | 0 | 0.000/0.000 |
| capture.group_query | 33 | 5.082/5.040 | 0 | 0.000/0.000 |
| scoring.callback | 33 | 4.378/4.325 | 0 | 0.000/0.000 |
| scoring.prefix_mean | 624 | 21.965/20.956 | 624 | 3.062/2.129 |
| scheduler.build_metadata | 624 | 443.956/427.470 | 624 | 420.041/398.695 |
| scheduler.consume_worker | 1250 | 145.903/143.829 | 1250 | 104.933/102.895 |
| metadata.bind_worker | 624 | 54.175/52.429 | 624 | 31.908/30.257 |
| metadata.build_worker | 624 | 32.116/31.215 | 624 | 46.727/29.191 |
| logging.preservation | 18 | 164.618/150.919 | 18 | 159.103/140.746 |
| logging.json_encode | 212 | 147.224/143.533 | 30 | 129.869/128.296 |
| logging.stdout_flush | 212 | 35.475/23.385 | 30 | 29.500/12.426 |

Summary construction is 33.81ms CPU/34.75ms wall across 66 jobs. Transfer submission is 108.29ms CPU versus 50.06ms ARC; its inclusive scope includes summary work. Sampled summary CUDA timings cover 5 jobs, total 0.784ms/max0.305ms; all CUDA samples drained. Sampling stride 16 means this is not total GPU cost and must not be extrapolated as a measured aggregate.

Candidate load-only stores: 66 jobs/1770 source blocks, job sizes 24×64 +6×32 +30×1 +6×2. Among 36 multi-block jobs, 8 have one contiguous physical run and 28 are scattered; 12 jobs have 64 distinct runs. ARC has 36 jobs/1746 blocks, also 8 contiguous/28 scattered multi-block jobs. These are actual request-owned transfer source IDs in preserved order, excluding warmup. No tensor-stride or per-job timing correlation is recorded.

The profile supports a bounded gather screen at most. Summary construction alone offers roughly 34ms of observed host cost across a 24–25s measured workload, so eliminating it entirely would recover only about 0.14%; any larger benefit remains unverified. Larger measured scopes include model execution and layout/metadata preparation.

## Evidence and handback

- [Combined analysis](confirmation-short-combined-results.json), with exact comparisons, all individual pairs and pooled CPU/latency/memory; associated CSV files retained.
- [Quality assessment](quality-prefix-assessment.json), including positive transport, per-decision numeric values, coverage, pins and 12 exact ARC comparisons.
- [Component assessment](diagnostic-short-profile-results.json), including raw component file pins and exactness.
- [Reproducible diagnostic assessment script](capture-followup-analysis.py); run only with GPU idle.
- Executed plans: [quality](quality-prefix-more-plan.json), [short repetition](confirmation-short-r1.json), [component profile](diagnostic-short-profile-plan.json), each with zero-exit receipt/log retained.
- [Process check](confirmation-followup-process-check.json) and [retained inventory](confirmation-followup-inventory.json).

No canonical, locked source, existing harness or frozen plan files were edited. One new authorized pinned component plan and analysis/report files were created. No cases retried or excluded; no additional GPU jobs beyond the 13 newly authorized cases. Final cleanup/release is recorded by the process receipt. All raw negative and diagnostic evidence is retained.

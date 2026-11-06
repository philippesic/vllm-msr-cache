# Frozen confirmation custody report

Both finite plans completed without operational failures: 33 runs, 594 completed requests. All source/harness pins remain valid. No canonical, frozen harness, plan or source files were edited, no cases were retried, and no additional GPU jobs were run.

The every-pair throughput target passes on the long workload and fails on the short workload. Both plans exercise numerical scoring and native prefetch, but neither exercises constrained competing semantic selection; scoring quality is not established by these confirmations.

## Pooled measurements

Throughput pools output tokens divided by the sum of measured load elapsed time. CPU is sampled process CPU seconds per run, not wall time. TTFT is return-request mean first-token latency. Memory columns are actual load peak CUDA allocated, CUDA reserved and sampled NVML device memory in MiB.

| Workload | Role | Tokens/s | CPU s/run | Return TTFT ms | CUDA MiB | Reserved MiB | NVML MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| long | arc_full | 7.5215 | 106.695 | 177.087 | 4251.700 | 4296.000 | 4980.688 |
| long | candidate | 7.4843 | 107.635 | 202.143 | 3825.172 | 3916.000 | 4600.688 |
| long | arc_small | 7.3838 | 108.595 | 764.373 | 3817.700 | 3908.000 | 4592.688 |
| long | original | 6.1234 | 132.007 | 967.188 | 3854.735 | 3958.000 | 4642.688 |
| short | arc_full | 30.8877 | 26.287 | 100.407 | 3411.691 | 3446.000 | 4130.688 |
| short | candidate | 30.5063 | 26.660 | 106.594 | 3308.542 | 3362.000 | 4046.688 |
| short | arc_small | 30.5393 | 26.583 | 155.425 | 3306.691 | 3360.000 | 4044.688 |
| short | original | 21.0383 | 38.387 | 306.324 | 3315.863 | 3366.000 | 4050.688 |

Long candidate throughput is -0.494% versus large ARC, +1.362% versus equal-budget ARC, and +22.226% versus original semantic. CPU is +0.881%, -0.884%, and -18.462%, respectively. Return TTFT is +25.055ms (+14.149%) versus large ARC, -562.230ms versus matched ARC, and -765.045ms versus original. Overall mean/p95 TTFT is +0.248%/+0.041% versus large ARC; return p95 is +16.501%.

Short candidate throughput is -1.235% versus large ARC, -0.108% versus equal-budget ARC, and +45.004% versus original semantic. CPU is +1.420%, +0.288%, and -30.549%, respectively. Return TTFT is +6.188ms (+6.162%) versus large ARC, -48.831ms versus matched ARC, and -199.730ms versus original. Overall mean/p95 TTFT is +0.938%/+0.638% versus large ARC; return p95 is +9.100%.

The long KV pool is 686MiB versus large ARC 1120MiB, with identical 3584MiB CPU budgets; matched ARC uses the same 686MiB GPU pool. Actual peak allocation saves 426.528MiB, reserved memory 380MiB, and sampled NVML 380MiB versus large ARC. The short pool is 175MiB versus large ARC 280MiB, with identical 896MiB CPU budgets; actual peak allocation saves 103.149MiB, reserved 84MiB, and sampled NVML 84MiB. Candidate versus matched ARC adds 7.472MiB allocated/8MiB reserved/8MiB NVML on long, and 1.850MiB/2MiB/2MiB on short.

## Individual paired throughput ratios

| Pair | Candidate / large ARC % | Candidate / matched ARC % | Both >=99% |
|---|---:|---:|---|
| long-1103-r0 | 99.0483 | 101.7569 | yes |
| long-2207-r0 | 99.3817 | 100.5374 | yes |
| long-3301-r0 | 99.9359 | 101.5556 | yes |
| long-1103-r1 | 99.8048 | 101.2979 | yes |
| long-2207-r1 | 99.4324 | 101.8209 | yes |
| long-3301-r1 | 99.4259 | 101.2056 | yes |
| short-1103-r0 | 97.4592 | 98.6302 | no |
| short-2207-r0 | 99.2540 | 100.5670 | yes |
| short-3301-r0 | 99.5998 | 100.4958 | yes |

Short seed1103 is a preserved negative result: 97.4592% versus large ARC and 98.6302% versus matched ARC. Pooled short matched throughput is within 1%, but the predefined every-pair condition is false. Pooled short large-ARC throughput is also outside 1%. The latency materiality criterion has no automatic threshold; the positive return-TTFT deltas versus large ARC must remain visible. These finite repeated/seeded measurements do not prove statistical equivalence.

## Exactness and semantic evidence

All 33 runs pass integrity, request completion, ordered actual native-key identity, terminal/physical-block-table proof and process-cleanup checks. All 39 same-seed candidate/control comparisons (702 matched request comparisons) preserve prompt/output token IDs and exact ordered native KV keys. All equal-budget comparisons have identical GPU and CPU cache bytes. No preemptions, prefetch allocation failures or transfer failures occurred.

All 9 candidate runs have numerical prefix-mean scoring, ARC cache semantic ordering verification and native adoption preserving physical block IDs. Combined candidates captured 765 observations (792 request rows), performed108 prefix-mean evaluations with 50 applied candidates, and natively adopted 54 staged jobs/15,606 blocks. Candidate return demand-H2D bytes and splice bytes are 0, and completed-prefetch waste bytes are 0.

**Limitation:** constrained semantic decisions are 0 in every candidate; both `all_candidate_constrained_semantic_selection_exercised` and the stricter `all_candidate_semantic_functionality_exercised` are false. No scoring-quality diagnostic was enabled. Numerical scoring does occur, but these all-fit nomination traces do not establish useful winner selection or parity with the original scoring trajectory.

## Reproduction and evidence

- Plans: [long](confirmation-long.json), [short](confirmation-short.json).
- Complete analyses including pins and raw paths: [long](confirmation-long-results.json), [short](confirmation-short-results.json); associated `-runs.csv`, `-pairs.csv`, `-matched-pairs.csv`, `-pooled.csv` are retained.
- Zero-exit receipts: [long](confirmation-long-receipt.json), [short](confirmation-short-receipt.json); [long log](confirmation-long.log), [short log](confirmation-short.log).
- Run each frozen plan with the project venv Python: `python -B output/prefetch-vram-20261101/batch.py output/prefetch-vram-20261101/confirmation-{long,short}.json` (one plan at a time).
- Reanalyze idle runs with `python -B output/prefetch-vram-20261101/confirm_results.py --plan output/prefetch-vram-20261101/confirmation-long.json --output output/prefetch-vram-20261101/confirmation-long-results.json --require-complete` and equivalent short paths. Do not rerun frozen cases into the same paths without an explicit preservation plan.

## Process and storage handback

Original long driver was adopted via verified PID/start identity and Linux pidfd; it exited. Short driver and both analyzers exited 0. Final GPU/process checks are recorded in `confirmation-custody-process-check.json`; no owned live model, driver or waiter remains. Sole GPU/process custody is released after that final check.

All 33 run directories, including compressed raw negative evidence, are retained: 844,162,422 logical bytes (844,673,024 allocated bytes), approximately 805MiB. No disposable fixtures/downloads/cache/worktrees were created during custody. No evidence was deleted. Inventory and free-space measurements: [inventory](confirmation-custody-inventory.json). Free-space change is measurement/file-creation activity, not reclaimed storage; Windows backing-volume reclaim was not measured.

Recommendation for the research owner: retain this candidate as a long-workload memory/throughput result, not an unconditional goal completion. Investigate the short fixed cost and obtain real constrained-selection/quality evidence before recommending it broadly.

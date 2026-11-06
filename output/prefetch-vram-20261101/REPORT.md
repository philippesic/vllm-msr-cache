# Semantic capture, scoring and exact prefetch investigation

**The general 1% overhead goal was not reached. A useful long-context capacity result was confirmed:** the retained prefetch configuration saves 426.5 MiB of peak allocated GPU memory and stays within 1% of large-cache ARC throughput in all six pairs. At the **same smaller cache budgets**, it is 1.36% faster than ARC and cuts return first-token latency by 73.6%. Short-context throughput remains 1.25% below large-cache ARC; equal-budget pooled throughput is within 1%, but two of six individual pairs fail. This is a workload-specific result, not general equivalence or production qualification.

The earlier capture/scoring investigation remains in [its report](../scoring-overhead-20261101/REPORT.md): matched-budget scoring stress improved from 42.946 to 63.075 tokens/s single-session and 42.284 to 56.747 concurrent, but remained 1.42%/2.24% behind ARC. Its 71 screens, 22 paired comparisons and quality limitations are preserved. The follow-up below tests actual offload and prefetch with real numerical scoring.

## Confirmed results

Same RTX 2080 Ti WSL host, i9-7980XE, cached Qwen2.5-1.5B-Instruct, FP16 eager Triton, TP1, block size 16, two active sequences, 1024 batched tokens. [Host](host.json), model revision, source, runtime and harness identities are pinned. Each fresh-server run has the same warmup, then 18 requests/768 output tokens across three rounds. Each round warms two observed sources, runs two serial peer tools, then releases concurrent returns. Long source/tool prompts contain 6144/12288 tokens; short prompts contain 1536/3072. Three content seeds and two balanced repetitions give six pairs per workload. Three true pre-task original-semantic references per workload use the same small budgets. The [long](confirmation-long.json) and [combined short](confirmation-short-combined.json) plans retain every planned run, including failures; the second short repetition was added after mixed first-repetition results.

| Workload | Arm | Tokens/s | Server CPU s/run | Return TTFT ms | Peak CUDA MiB |
|---|---|---:|---:|---:|---:|
| Long | Candidate | 7.4843 | 107.635 | 202.14 | 3825.17 |
| Long | Large ARC | 7.5215 | 106.695 | 177.09 | 4251.70 |
| Long | Equal-budget ARC | 7.3838 | 108.595 | 764.37 | 3817.70 |
| Long | Original semantic | 6.1234 | 132.007 | 967.19 | 3854.74 |
| Short | Candidate | 30.5287 | 26.688 | 106.42 | 3308.54 |
| Short | Large ARC | 30.9154 | 26.238 | 102.38 | 3411.69 |
| Short | Equal-budget ARC | 30.5658 | 26.552 | 153.93 | 3306.69 |
| Short | Original semantic | 21.0383 | 38.387 | 306.32 | 3315.86 |

Long candidate throughput is **−0.494% / +1.362% / +22.226%** against large ARC / equal-budget ARC / original semantic; CPU is **+0.881% / −0.884% / −18.462%**. Short throughput is **−1.251% / −0.121% / +45.111%**; CPU is **+1.715% / +0.515% / −30.475%**. CPU counts sampled server descendant-process CPU seconds, excluding load generation and hint-client CPU. Sampling overhang is retained, and unstable process identities are rejected.

| Pair seed/repetition | Long vs large / matched ARC % | Short vs large / matched ARC % |
|---|---:|---:|
| 1103/0 | 99.048 / 101.757 | 97.459 / 98.630 |
| 2207/0 | 99.382 / 100.537 | 99.254 / 100.567 |
| 3301/0 | 99.936 / 101.556 | 99.600 / 100.496 |
| 1103/1 | 99.805 / 101.298 | 98.806 / 100.522 |
| 2207/1 | 99.432 / 101.821 | 98.594 / 100.202 |
| 3301/1 | 99.426 / 101.206 | 98.798 / 98.875 |

Throughput divides all successful output tokens by complete load elapsed time, including tools, hints, prompt processing and return delays. TTFT means release-to-first-token latency. Compared with large ARC, overall mean/p95 TTFT changes are +0.248%/+0.041% long and +0.844%/−0.177% short. **Return TTFT still rises 25.1 ms (14.1%) long and 4.0 ms (3.95%) short**; no automatic materiality threshold was assumed. Long return p95 is 230.6 versus 198.0 ms; short 116.9 versus 114.5 ms. Return inter-token time is 33.89 versus 33.21 ms long and 33.20 versus 32.83 ms short. Equal-budget ARC return latency improves substantially, but the large-cache latency trade-off must remain visible. These are descriptive repeated pairs, not a statistical equivalence proof.

[Full long measurements](confirmation-long-results.json) and [full short measurements](confirmation-short-combined-results.json) include individual latencies, CPU, allocator/device memory, counters and exact comparisons. The older three-pair short file is retained as history, not substituted for the combined result.

## Actual VRAM and prefetch mechanism

Long GPU KV allocation falls from 2560 blocks/1120 MiB to 1568 blocks/686 MiB; all arms have 8192 CPU blocks/3584 MiB. Actual peak CUDA allocation falls **426.5 MiB**, reserved/device-NVML peaks **380 MiB**. Short allocation falls from 640 blocks/280 MiB to 400/175 MiB, all with 2048 CPU blocks/896 MiB; actual CUDA/reserved/NVML savings are **103.1/84/84 MiB**. At equal budgets the candidate adds 7.47 MiB of CUDA allocation long and 1.85 MiB short. Logical free blocks alone are not counted as VRAM savings. NVML sampling can miss spikes and includes display memory. Larger-model feasibility and production traces were not tested.

The retained implementation uses ARC retention and exact complete-prefix prefetch. Applications nominate only an already observed prefix and current pending-tool state. The client sees received source tokens and a constant 250 ms estimate; it receives no future return prompt, final-tool completion, private release time or selected winner. Returns have **zero added post-tool idle**, release independently, and never await a hint acknowledgment. Prefetch overlaps normal tool execution.

Implemented changes, all in the [task-only patch](task.patch):

- Dense FP16 native adoption validates exact source identity, namespace, generation, key order, physical GPU IDs, readiness and ownership. Full-prefix staging refuses partial or already-GPU-cached duplicates and checks budget again at submission.
- Exact-source refreshes acknowledge the existing canonical lease without renewing timestamps. Active admitted prefill prevents staging; completed obsolete scratch can be reclaimed for demand while preserving bound-source protections.
- Full-block source identities can include **actually computed** output KV. Finished stores exclude in-flight work and the final sampled but unexecuted token. Decode-store audit ranges use the actual computed extent; prompt-demand accounting remains prompt-only.
- Mean-only private summaries preserve the same FP32 means. Prefix scoring uses bounded owned centroids and separate scalar transport instead of per-block rank readback/EMA; source/query/publication/residency/expiry checks remain mandatory. No fake block-score counters are emitted.

The [final manifest](canonical-final-pins.json) and [source verification](final-source-verification.json) show canonical package equality to the confirmed `sources/prefix-mean`. Defaults remain off for the new prefix/transcript modes. The rejected gather optimization exists only in a private frozen snapshot and was **not integrated**.

## Real scoring and quality limits

Every one of the 12 performance candidate runs captured queries, consumed numerical prefix scores and natively adopted all six staged prefixes: **72 jobs/17,352 blocks**, with no return demand H2D, preemptions, failed transfers or completed-prefetch waste. All-fit scoring is real work, but those runs have **zero constrained semantic choices**. The strict combined scoring-functionality criterion therefore remains false; their speed/memory result does not demonstrate useful winner selection.

Separate three-seed diagnostics force competing choices with 1536 GPU blocks and a recorded 250 ms post-tool gap. All-cadence shadow capture/scoring deliberately invalidates every diagnostic performance measurement. The original block-rank EMA executes on an isolated shadow manager along the same deployed ARC trajectory; it cannot alter real staging. Positive typed worker-to-scheduler delivery produced 2043 observations and 9,431,049 key updates. All **five constrained decisions** had complete fresh key coverage and numerical scores. New raw-prefix and old EMA top choices/full ordering agree in **2/5 (40%)**. Four other actual decisions were unconstrained and lacked raw scores; they are excluded, not counted as agreement. [Quality assessment](quality-prefix-assessment.json) preserves coverage, margins, individual decisions and exactness checks.

The new policy uses a common fresh active-peer query; the old policy accumulates own-source per-key ranks over time. Same-input centroid arithmetic matched dense GPU computation within 2.98e-8, but that does **not** make these policies equivalent. Five choices cannot establish future-reuse accuracy, and EMA agreement is not ground truth. Bounded unscored fallback remains possible when no fresh complete query/features exist. Treat this as a distinct capacity-oriented policy, not a quality-preserving replacement for full-cadence semantic retention.

## Correctness, negative evidence and stopping decision

All **42 frozen confirmation runs** pass authoritative terminal, exact prompt/output token, ordered native-key, physical adoption/block-table and cleanup checks. All three quality diagnostics also match same-seed ARC prompts, output tokens and ordered keys, despite timing perturbation. The canonical CPU suite passed **623 tests with CUDA forbidden**; mean-only and centroid CUDA qualifications cover ownership, publication invalidation and bounds. The private gather candidate passed 478 CPU tests and 16 CUDA gather/reduction cases bitwise, but supplied no speed improvement. [Native transport qualification](native-byte-validation.json) separately checks **21,102,592 transferred bytes** through the installed D2H/H2D handlers using **synthetic packed pages matching the 28-layer K/V layout**, reordered/duplicate source IDs, post-fence source reuse and a one-byte corruption control rejected by the comparison oracle. That standalone test includes no model forward or native adoption. It does not certify every concurrent schedule or explain a model-output difference.

**An isolated later ARC control failed exact-output comparison.** During the rejected gather screen, ARC640 request 16 agreed on the first output token then diverged in 53/64 positions; eight other same-seed ARC controls agree with the gather, frozen baseline and ARC400. Prompt tokens and 99 native prompt keys match, but generated KV bytes/logits were not captured. That ARC run used no H2D, prefetch, adoption or preemption. Its cause remains **unresolved**: neither floating-point variation nor KV corruption has been established. The failed comparison is retained and not retried, waived or pooled into a passing claim. [Evidence](gather-screen-output-crossreference.json) and [independent code review](output-divergence-review.md) describe the missing generated-KV/logit evidence. This limits production confidence even though the prior frozen comparisons and subsequent independent screen pass.

[Negative results](NEGATIVE_RESULTS.md) retain failed/stalled candidates and incorrect hypotheses. Important limits include async scoring regression, CPU partitioning confounds, zero-score opportunity runs, incomplete generated-prefix reuse, insufficient 1536-block capacity at zero gap, sparse-cadence quality loss, and invalid screen CPU sampling after WSL time correction. The final bounded alternatives were also negative: ordered gather was 0.32% slower than the frozen baseline; public last-tool submission activation cut hint RPCs 102→42 but was 1.82%/1.70% below fresh large/equal-budget ARC. The latter is causal but can change query/lease timing, so it is a distinct producer policy. [Gather](gather-screen-report.md), [hint activation](hint-event-screen-report.md) and [current component profile](diagnostic-short-profile-results.json) preserve these outcomes. Current summary construction costs only 34.75 ms per short run (about 0.14% of load time), limiting gather headroom; broader source-finalization or deferred-store lifecycles are not justified by that measured cost and remain unimplemented.

**Recommendation:** retain the opt-in exact transcript/full-prefix/native-adoption path and bounded prefix scorer for further long-context capacity trials. Keep original/default scoring available, do not integrate the gather candidate or claim a universal 1% result, and do not enable this broadly until the ARC divergence is understood and ranking benefit is evaluated on real application traces. The tested local alternatives have not closed the short-case gap; fused scoring or a new summary/store lifecycle would be further architectural research, not a verified recommendation.

## Reproduction and cleanup

Use the existing venv and pinned cached model on this host. The cloner verifies source, harness, external checkout dependencies, package versions and model identity, then creates a fresh sibling directory without overwriting results:

```bash
SEMANTIC_PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
"$SEMANTIC_PY" -B output/prefetch-vram-20261101/reproduce.py output/prefetch-replay
"$SEMANTIC_PY" -B output/prefetch-replay/batch.py output/prefetch-replay/confirmation-long.json
"$SEMANTIC_PY" -B output/prefetch-replay/batch.py output/prefetch-replay/confirmation-short-combined.json
"$SEMANTIC_PY" -B output/prefetch-replay/confirm_results.py --plan output/prefetch-replay/confirmation-long.json --output output/prefetch-replay/long-results.json --require-complete
"$SEMANTIC_PY" -B output/prefetch-replay/confirm_results.py --plan output/prefetch-replay/confirmation-short-combined.json --output output/prefetch-replay/short-results.json --require-complete
```

Run serially and analyze only after GPU shutdown. The clone also includes `quality-prefix-enabled-plan.json` and `quality-prefix-more-plan.json`; run each with `batch.py` separately and exclude their performance. `--plan NAME` selects other preserved plans, including negative screens. Failed historical setup `quality-prefix-plan.json` omitted the observer-enable flag and yielded no quality evidence; the repair is recorded rather than interpreted as a scoring result. [Reproduction validation](reproduction-validation.json) checked a real fresh bundle (126 pinned files, 111 Python sources, all 45 planned inputs); it did not rerun the campaign or fabricate run receipts.

No commit or push. Existing unrelated dirty work is preserved. Final process checks, bounded retention inventory, removal of the disposable reproduction clone/bytecode and measured WSL free-space changes are recorded in [STORAGE.json](STORAGE.json). Frozen sources, compressed raw negative/positive evidence and reusable shared model/runtime caches are retained; deleting WSL files does not establish Windows VHDX reclamation.

Cleanup removed the validated reproduction clone and task bytecode (9.87 MB logical). WSL available space increased from 934,615,384,064 to 934,625,636,352 bytes (observed +10,252,288). Approximately 2.00 GiB of current evidence and 914.5 MiB of protected prior-phase evidence remain, principally compressed raw runs and frozen sources. Final GPU occupancy and owned-process checks are empty.

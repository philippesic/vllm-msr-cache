# Runtime handoff: quantized public-workload campaign

The finite campaign is complete. It does **not** establish the 1% exact-work throughput target, a plain production win, a sustainable concurrency limit, or 40% less total model VRAM. All 12 new agent runs and all seven diagnostic runs passed strict native proofs. The original 24-run matrix remains archived with its one invalid candidate proof. No commit or push was made.

## Measured scope and controls

Executed all 24 original frozen runs, then all 12 newly frozen agent runs after the observer-only repair, then all seven predeclared diagnostic runs. The accepted core uses 12 old valid chat runs and 12 new agent runs (1,296/1,296 strict request proofs); these retain different observer revisions. Diagnostics add 296/296 strict proofs. The old 12 agent runs are retained separately, including the failed cell, and never pooled with the new agent cohort. Additional feasibility, learning, failed-integration and intrusive qualifications remain under runs/.

Official Qwen2.5-7B-Instruct AWQ4bit, actual Marlin kernel on RTX2080Ti SM75, FP16 activations/KV, eager Triton V2, TP1. CPU cache is 4,096 blocks / 3.5GiB in every accepted arm. Large ARC GPU pool: 3,072 blocks / 2,688MiB; other arms: 2,048 / 1,792MiB. Per-block geometry is 16 tokens / 917,504 bytes. Exact identities and commands are pinned in the three plans and individual raw records.

Public WildChat user content and mini-SWE-agent repository/tool observations; actual generated outputs become subsequent history. Historical assistant answers are excluded. Recorded tools are external replay, not execution of newly generated commands. Chat: 12 clients / at most four active sequences / zero wait. Agent: 24 clients / at most eight active sequences / seeded synthetic private 0/2/4/8/12-second waits. The policy sees no future prompt or actual return time; its declared estimate is fixed at 750ms. EOS is respected, maximum96 output tokens. Three paired corpus partitions with balanced arm order provide descriptive repeated comparisons, not an identical-repetition confidence interval.

## Core results

Ranges below span the three paired corpus partitions. Full p50/p95 latency, queue, request rate, SLO goodput, CPU/RSS, CUDA/NVML, traffic and mismatch inventories are in final-results.json and final-results.csv. CPU is sampled aggregate server-process CPU seconds over the measured load; TTFT/ITL are client receipt measurements.

| Cohort / arm | Output tokens/s | Server CPU s | p95 TTFT s | p95 ITL ms | p95 E2E s |
|---|---:|---:|---:|---:|---:|
| chat / largeARC | 90.54–99.12 | 31.67–36.62 | 7.32–9.43 | 43.12–45.36 | 10.88–12.00 |
| chat / smallARC | 90.87–101.95 | 30.24–36.08 | 7.26–9.38 | 43.11–44.93 | 10.87–12.04 |
| chat / candidate | 88.92–100.59 | 30.59–37.08 | 7.39–9.62 | 43.91–46.21 | 11.10–12.15 |
| chat / original | 60.86–69.55 | 43.66–53.71 | 10.78–13.34 | 61.34–66.28 | 16.04–17.83 |
| agent / largeARC | 70.05–85.56 | 83.04–99.37 | 17.73–22.59 | 349.35–527.22 | 24.90–34.07 |
| agent / smallARC | 68.39–83.58 | 85.51–101.96 | 17.77–23.72 | 348.33–524.69 | 26.10–35.46 |
| agent / candidate | 66.70–84.50 | 84.99–105.35 | 17.73–25.86 | 338.84–553.40 | 25.12–37.39 |
| agent / original | 44.37–53.72 | 132.80–157.11 | 27.64–42.23 | 475.51–571.59 | 43.56–59.48 |

Exact-matched chat candidate/smallARC throughput ratios are0.98663,0.98598,0.97861 (geomean0.98373): none meets the1% target. These short chat runs do not exercise scoring/prefetch. The new agent candidate/smallARC descriptive ratios are1.02276,0.97523,1.01101 (geomean1.00279); candidate/largeARC geomean0.97453, with none of its pairs within1%. Candidate/original geomean1.55567. All agent pairs have generated-output/downstream-history differences and are ineligible for an exact-work performance claim. Original uses continuous dispatch/all-cadence mean block ranks with EMA and MSR retention; candidate uses opportunity raw mean-query cold-suffix scores and ARC retention. This is an integrated configuration comparison, not a pure scoring ablation.

All closed-loop core cohorts miss the predeclared aggregate TTFT SLO (chat5s; agent15s). Candidate agent seed2207 has p95TTFT25.86s vs smallARC23.72s and CPU105.35s vs101.96s. Candidate preemptions are0/2/0 vs zero in all new controls. One candidate seed1103 has29.4MB global host swap-out; this is flagged and retained, not silently excluded. Full host paging and stable process faults remain in summaries.

## What saves memory, and what prefetch costs

The production-like mechanism preserves exact KV: completed prefix pages are offloaded to a host cache; a causal pending-tool hint nominates already-observed cold blocks, asynchronously copies an exact contiguous suffix to reserved GPU pages, then publishes only ready/fenced pages immediately before normal native prefix lookup. Shared local prefix pages retain native hashes, physical identity and owners. There is no KV approximation or wait for speculative completion at demand admission.

The allocation difference is896MiB / one-third of the **KV pool**, not one-third of total model memory. Smaller ARC provides the same reduction. In the new agent cohort, actual peak CUDA allocation is8.678GB for largeARC,7.739GB for smallARC, and~7.747GB for candidate: approximately0.931GB /10.7% less candidate total allocated memory than largeARC. NVML includes variable desktop/display occupancy; allocator measurements are the more stable attribution. The weights and activation/runtime memory remain on GPU. This does not measure concurrency on a32GiB27B endpoint.

Candidate actually executed53 numerical scoring calls over113 candidates,27 constrained nominations and11 native adoptions across the new agent cohort. Completed speculativeH2D is3,269,984,256 bytes; named intended requests consumed1,529,479,168 (~46.8%). Remaining completed bytes were not consumed by the named intended request; unrelated future cache utility is not measured. Bound reservation peaks447/367/590blocks (~391/321/516MiB) occupy the fixed GPU pool and can displace active KV. These reservation costs and the two observed preemptions are material.

## Bounded diagnostics

- Shadow originalEMA diagnostic:72 strict proofs;888 transported observations /2,685,265 key updates, fresh reference coverage for all396 keys in the only actual multi-candidate staging decision. The real candidate raw scores were absent at that decision, so zero fully comparable decisions and agreement **unknown**. The extra all-cadence scoring/capture perturbed opportunity timing and invalidates performance comparisons. No task-answer-quality preservation is established.
- Abandoned-wait pair:64 requests/arm, four causally abandoned clients. Candidate81.978TPS vs ARC82.733 (-0.91%), p95TTFT20.364s vs20.078s, CPU77.81s vs76.17s; four native adoptions. Completed prefetch875.30MB, named consumed416.55MB. None of the four abandoned sessions was actually staged (two had hint bindings), so this run does not qualify in-flight cancellation-waste behavior. Across all sessions,152.31MB completed bytes expired and306.45MB retired for other reasons. Histories differ; the rate ratio is descriptive.
- Finite open-loop first-turn chat: fixed ARC-only nominal1.27575requests/s calibration, offered0.63787/1.08439requests/s,24 requests/arm. Candidate vs ARC TPS50.382/50.372 at low rate and80.206/80.120 at high; p95TTFT308/276ms and1761/1656ms. All requests meet the declared chat SLO. Zero scoring/adoption makes this a no-reuse control, not evidence of scoring performance or sustainable capacity. Identical prompts but1/2 output mismatches at the two rates also prohibit exact-work claims.

## Correctness qualification and preserved failures

The old candidate agent seed1103 has one invalid physical proof after native preemption. The old orchestrator mistakenly gated on schema/timing validity, ran the remaining old matrix, and is preserved unchanged; confirmation-progress.json contains the acceptance correction. No failed raw record was repaired. The observer-only revision requires actual V2 scheduler output, same live request, advanced observed preemption episode/count, exact token/hash chain, full live native physical table and positively owned computed pages, then invokes the existing strict full replacement validator. It changes no native scheduling/KV/transfer/scoring policy. It is applied through a common hook to every new arm, including original source.

The first640-block public diagnostic failed on native-vs-serialized key representation in the observer fixture and an unconfigured optional local-prefix audit. It is retained. Corrected CPU tests use the actual connector key helper:125 passed including invalid episode/request/hash/count/table controls and existing mapper cases. The second actual-model640-block stress passed24/24 strict proofs with four actual V2 preemptions,209 transfer jobs and5,074,714,624 component bytes equal, including rejected one-byte mutations. Its synchronization/timings are excluded. Local-prefix byte audit was explicitly not configured in that run; prior dedicated qualification separately verifies all local prefix components unchanged through adoption and all transferred suffix bytes equal. Intrusive synchronization may mask races, so strict physical publication/adoption proofs remain necessary in performance runs.

Additional retained negative evidence: initial0.92 GPU startup reservation failure; initial7GiB CPU cache host paging and stopped control; invalid1000ms hint/provenance integration;12- and24-client original full-prefix eligibility failures due naturally shared local blocks; publication-at-enqueue suffix copies missed at native lookup; unresolved cross-arm output differences, including ARC-versus-ARC; all unsuccessful and slower candidates. See RESEARCH_CONTRACT.md, READMISSION_REVISION.md, PREFLIGHT*.json and raw runs for provenance.

## Changes, applicability and handoff

Private algorithm files are sources/shared-lookup/semantic_offload/connector.py plus runtime-shared-lookup/vllm/v1/core/block_pool.py and core/sched/scheduler.py. PRIVATE_PATCH.diff/PRIVATE_PATCH_MANIFEST.json show exact changes against frozen candidate/runtime. Feature flags remain private/default-off; tested on this dense single-group TP1 FP16KV model and pinned vLLM, not a production-general integration. The original full-prefix path and invalid ownership/fence/prefix controls were retained in the302 CPU combined checks and actualAWQ qualification.

Observer/orchestration files are readmission_observer.py, hooks-readmission/sitecustomize.py, run_readmission.py and confirm_strict.py. run_diagnostics.py/quality_hooks_readmission add separate instrumentation only. Corpus and metrics were independently prepared and checked by their assigned owner (144 sessions, future-history negative controls,32 metrics tests); CPU transport check passed with CUDA forbidden.

Recommend retain the private exact-suffix/JIT mechanism as research evidence; do not advertise99% throughput,40% total-VRAM saving, more supported production sessions, or unchanged scoring quality. ARC right-sizing is the measured simpler control. Production qualification would still need representative multi-GPU/model/runtime validation, comparable scoring-quality evidence, sustained offered-load/SLO measurements and resolved exact-work interpretation. No further algorithm branches were run.

Reproduction: REPRODUCE.md. Machine-readable results: final-results.json/final-results.csv and per-run summaries. Final source pins and runtime custody are in FINAL_VALIDATION.json; storage/cleanup inventory in STORAGE.json. Valuable compressed evidence and the reusable model cache are retained. No service, GPU process or listener from this work remains; port19372 and GPU flock are released. Canonical dirty changes and old campaigns are preserved.

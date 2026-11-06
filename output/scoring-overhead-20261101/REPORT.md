# Semantic capture and scoring overhead

**The overall 1% goal remains unmet.** Against matched ARC, the retained candidate was 1.42% slower in single-session scoring stress and 2.24% slower in concurrent-session stress. It improved throughput over the original semantic configuration by 46.9% and 34.2%, respectively.

Ordinary concurrent-session confirmation met the observed pooled throughput criterion: -0.64% versus ARC, with mean first-token latency +1.70%. This workload-dependent result does not establish general 1% equivalence, latency non-regression or unchanged scoring quality.

## Repeated paired measurements

Each cell pools three fresh-server runs at seeds 1103, 2207 and 3301. Stress uses admission threshold 0, admitting one-use blocks to exercise scoring and eviction:

| Workload | Arm | Output tokens/s | Mean TTFT ms | p95 TTFT ms | CPU core seconds/run |
|---|---|---:|---:|---:|---:|
|single|arc|63.982|2089.0|4917.5|23.94|
|single|original|42.946|4209.6|10155.8|35.79|
|single|candidate|63.075|2042.8|4546.7|24.35|
|multi|arc|58.045|6923.2|15225.2|59.57|
|multi|original|42.284|9825.3|23465.1|81.55|
|multi|candidate|56.747|7017.9|15561.3|61.19|

A separate ordinary threshold-2 confirmation excludes the earlier promising screen:

| Workload | Arm | Output tokens/s | Mean TTFT ms | p95 TTFT ms | CPU core seconds/run |
|---|---|---:|---:|---:|---:|
|multi|arc|59.982|6610.3|15067.4|57.53|
|multi|original|47.612|8631.5|21194.3|72.90|
|multi|candidate|59.600|6722.5|15338.8|58.09|

Ordinary candidate/ARC throughput differences were +0.50%, -1.29% and -1.13% by seed; mean TTFT differences were +5.35%, +2.03% and -2.19%. The pooled ordinary result therefore does not hold for every pair.

Throughput divides all output tokens by the complete load interval, including queueing and background traffic. TTFT runs from intended release to first token; p95 is its 95th percentile. CPU measures sampled server-process-tree core seconds, including GPU waits, at 250 ms intervals. Window overhang, median TTFT, inter-token latency, memory and individual pairs remain in [stress results](confirmation-results.json), [ordinary results](ordinary-confirmation-results.json) and the [run index](run-index.csv). No unrelated tests ran during confirmation. Dynamic GPU clocks and three pairs limit precision at a 1% margin.

## Matched host, budgets and scoring activity

All arms used this RTX 2080 Ti WSL host, local Qwen2.5 1.5B Instruct, FP16 eager Triton attention, one installed vLLM prerelease/runtime overlay, 640 GPU and 199 CPU KV blocks (280 MiB/87.0625 MiB), three active sequences and 1024 batched tokens. Prepared inputs and model revision are pinned.

Single-session runs contain 18 requests/1536 output tokens; concurrent runs contain 45/3264. Generated closed-loop sessions append actual outputs between turns; background arrivals are fixed. These are simulations, not production traces. The ordinary single-session screen had no scoring bank and cannot count as success.

The repeated stress candidate applied 6,129/44,676 key-score updates and encountered 48/325 eviction calls with scored resident candidates. Ordinary confirmation applied 1351 updates across 215 calls, restored 608 tokens and had no scored eviction. ARC/original restored 624/784 ordinary tokens. Scoring was active, but no semantic retention benefit was demonstrated.

ARC restored 1344 stress tokens versus none for either semantic arm; GPU prefix hits also differed. The remaining end-to-end gap includes cache history and policy outcomes, not solely capture overhead.

## Changes and quality trade-offs

Canonical changes retain the opt-in attention implementation hook, batched summaries, flat scoring-cache stacks, ordered-slice compaction and corresponding tests. Defaults remain dispatch capture and all-step observations. Exact KV identities and transfer fencing are unchanged; eager FP16 Triton serving checks do not certify compiled execution or graph replay.

The confirmed candidate sets `SEMANTIC_CAPTURE_BACKEND=attention`, `SEMANTIC_CAPTURE_STEPS=multi_token`, stride 1 and synchronous continuous flat mean scoring. It preserves dense scores at observed steps but skips batches where every request schedules one token, including single-token prefills. Mixed batches retain every nonempty row.

Reduced observations change the EMA trajectory. An optimistic common-key hold-last diagnostic measured 63.1% mean top-10 overlap and 0.766 rank correlation against fresh queries. It initializes first observations and excludes missing new keys; this is neither EMA fidelity nor cache-hit quality. Keep reduced capture opt-in. The attention hook with `all` capture is the conservative recommendation, subject to backend validation; it has not met the throughput target.

Slices retain whole reduction buffers or previous scoring-cache storage. Maximum post-load retained summary storage across stress runs was 1.93 MiB single/0.91 MiB concurrent versus 0.78 MiB originally. These exclude stack/pinned caches and are not metadata peaks. Equal configured KV budgets do not imply equal total allocations.

## Negative evidence and remaining limits

The [run index](run-index.csv) retains 71 runs: 67 complete and four failed screens. Experiments covered direct capture, batching, GEMM, cadence, deferred delivery, immutable async banks, CPU scoring and pinned copies. Two GEMM implementation failures and two log-parser failures are excluded from performance confirmation. Every confirmation used the recovered exact pre-task source; see [baseline provenance](baseline-provenance.json).

An instrumented all-step run recorded 17.48 s in inclusive score readback versus 0.239 s dispatching vector math. These overlapping spans include preceding GPU waits and cannot be summed as isolated overhead. CPU scoring initially used 18 threads; one-thread execution cut CPU cost from 112.17 to 64.80 core seconds, yet full-cadence throughput remained 54.42 versus matched ARC's 57.30 tokens/s. Pinned stride-16 and CPU multi-token screens reached 56.44/56.42, still below target. Async delay and dropped observations also change quality; those sources remain experimental.

CPU math matched all 50 [sampled real-workload rankings](runs/cpu-quality-multi-1103-semantic/cpu-quality.json) (9290 scores; maximum error 4.58e-05), but [synthetic near ties](asynccpu-numerical-quality.json) reduced top-10 overlap to 96.7%. Reasonable tested local approaches did not close the stress gap. Fused device-side scoring remains untested future work.

## Correctness and reproduction

The final CPU suite passed 454 tests with CUDA initialization forbidden; [validation log](cpu-validation-handback.log) records the executed checks. GPU FP16/FP32 checks found bitwise-equivalent summaries/key mappings across variable and duplicate block layouts, preserved values after KV reuse and rejected an intentionally wrong mapping. All 1161 requests across the confirmation and ordinary-control sets completed with authoritative/native-key/physical-identity coverage, zero failed transfers and no retained descendants. All 22 paired comparisons matched prompts, output tokens and native keys; see [integrity](final-integrity.json). These checks support correctness on exercised paths.

Reproduce with the existing environment and cached pinned model, using a fresh destination:

```bash
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B output/scoring-overhead-20261101/reproduce.py output/scoring-overhead-replay --suite confirmation --run
```

Use another destination and `--suite ordinary-confirmation` for the ordinary suite; run suites serially. [Stress](confirmation-protocol.json) and [ordinary](ordinary-confirmation-protocol.json) protocols, [task-only patch](task-changes.patch) and [manifest](task-manifest.json) preserve settings and the original dirty-checkout baseline. No commit or push. [Storage inventory](STORAGE.json) records cleanup and retention.

Cleanup removed the disposable reproduction clone, task bytecode and duplicate temporary backups. WSL free space increased from 936,894,963,712 to 936,908,849,152 bytes (observed +13,885,440 bytes); about 914.5 MiB of useful compressed results and reproduction inputs remain outside Git. No GPU compute processes remained. Shared model/runtime caches were preserved.

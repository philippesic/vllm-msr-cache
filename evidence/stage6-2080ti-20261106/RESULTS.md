# Stage 6 measured outcomes

The separately authorized [repository recovery](repository-recovery-03/README.md)
completed all 18 configurations on 2026-11-08 using pinned rebuilt-01. Four paired
pressure blocks give candidate/ARC **1.0210 [0.9885, 1.0546]**, Continuum/ARC
**0.8627 [0.8311, 0.8954]**, upstream/ARC **0.9526 [0.9096, 0.9976]**.
The candidate gain is unresolved. ARC pressure is positive in every fixed block.
All 288 finals, 272 selected source-review items and the one capped intermediate
were audited; quality equivalence is not established. The historical gap below
describes the original sealed addendum only; it is now measured separately.
Primary comparisons and failed-gate history remain unchanged.

Acquisition and source review are complete. Results are conditional on the
2080 Ti/WSL2 host, frozen inputs and configured systems. They support
workload-dependent performance, not universal dominance or quality equivalence.
Both [primary](primary-portability-proof.json) and [addendum](addendum-portability-proof.json)
real relocation/offline verification and GPU-free rebuilt-runtime imports passed.
The [handoff](HANDOFF_5080.md) requires new-host hardware preflight; no 5080 job ran.

The primary matrix completed 524 configurations in 210 matched blocks: twelve
workloads on Qwen3-8B-AWQ, and five each on Ministral3-3B and Qwen3-4B. External
systems cover five anchor workloads only; this is not a full cross product.

## Native candidate versus ARC

Means below are serving elapsed seconds, excluding startup. The ratio is the
geometric mean of paired completed offered graph units/s, with whole-process
log Student-t 95% intervals. It is not the ratio of arithmetic mean times, a
quality-adjusted utility metric, or a multiple-comparison-adjusted winner test.
Within-run requests are not independent repetitions.

| Model / workload | n | Candidate mean s | ARC mean s | Throughput ratio [95% CI] |
| --- | ---: | ---: | ---: | --- |
| qwen3_8b_awq / B01 chat | 12 | 53.423 | 53.156 | 0.9950 [0.9823, 1.0079] |
| qwen3_8b_awq / B02 independent QA | 6 | 23.221 | 22.872 | 0.9861 [0.9198, 1.0571] |
| qwen3_8b_awq / B03 long-document summary | 12 | 228.708 | 228.877 | 1.0007 [0.9977, 1.0038] |
| qwen3_8b_awq / B04 hot/cold scan | 12 | 23.510 | 23.282 | 0.9904 [0.9756, 1.0055] |
| qwen3_8b_awq / B05 sequential repository | 12 | 156.353 | 156.138 | 0.9986 [0.9905, 1.0069] |
| qwen3_8b_awq / B06 repository GPU-reuse control | 8 | 30.247 | 30.283 | 1.0012 [0.9824, 1.0203] |
| qwen3_8b_awq / B07 roomy repository control | 12 | 30.495 | 30.272 | 0.9924 [0.9745, 1.0107] |
| qwen3_8b_awq / B08 tight documents | 12 | 208.855 | 212.942 | 1.0196 [1.0105, 1.0288] |
| qwen3_8b_awq / B09 standard documents | 12 | 211.008 | 212.495 | 1.0071 [1.0043, 1.0098] |
| qwen3_8b_awq / B10 team control | 12 | 25.604 | 26.241 | 1.0243 [0.9788, 1.0718] |
| qwen3_8b_awq / B11 mixed repository + QA | 8 | 70.406 | 79.162 | 1.1244 [1.1090, 1.1401] |
| qwen3_8b_awq / B12 external-priority repository control | 12 | 30.444 | 30.181 | 0.9914 [0.9788, 1.0042] |
| ministral3_3b / B02 independent QA | 6 | 22.379 | 22.037 | 0.9850 [0.9520, 1.0192] |
| ministral3_3b / B06 repository pressure | 6 | 90.364 | 93.574 | 1.0355 [1.0206, 1.0507] |
| ministral3_3b / B08 tight documents | 6 | 170.319 | 172.294 | 1.0117 [0.9980, 1.0256] |
| ministral3_3b / B09 standard documents | 6 | 172.832 | 172.842 | 1.0001 [0.9986, 1.0015] |
| ministral3_3b / B11 mixed repository + QA | 6 | 93.226 | 97.288 | 1.0438 [1.0143, 1.0740] |
| qwen3_4b_2507 / B02 independent QA | 10 | 13.661 | 13.617 | 0.9966 [0.9775, 1.0160] |
| qwen3_4b_2507 / B06 repository pressure | 10 | 123.428 | 124.226 | 1.0067 [0.9899, 1.0238] |
| qwen3_4b_2507 / B08 tight documents | 10 | 243.334 | 238.091 | 0.9786 [0.9657, 0.9917] |
| qwen3_4b_2507 / B09 standard documents | 10 | 241.328 | 235.701 | 0.9766 [0.9685, 0.9848] |
| qwen3_4b_2507 / B11 mixed repository + QA | 10 | 123.866 | 127.848 | 1.0322 [1.0193, 1.0453] |

## External configured systems

Both external systems share the offered workload/capacity envelope, but differ
in runtime, backend, chunking, generated outputs and hint interface. Native ARC
and candidate share the instrumented runtime. These are system comparisons,
not isolated eviction-policy comparisons. Missing external native counters are
unavailable, not zero.

Continuum is Hanchenli/vllm-continuum at
`316a58794a6ff86b216e579b74fd56ed0c5a911f`, LMCache 0.3.6 and Torch 2.8/cu128.
Pinned upstream is **vLLM PR #59714**, karen-sy/vllm at
`7966f717dd4eb09461dd97102db1352c924bbf50`, with the declared
mapped-host-pointer compatibility patch and Torch 2.13/cu130. SAE CPU-offload
PR #50422 is a separate, source-only feasibility note; it was not measured.

| Anchor cell | System | n | System mean s | Candidate mean s | System/candidate throughput [95% CI] |
| --- | --- | ---: | ---: | ---: | --- |
| B06 | continuum | 8 | 30.286 | 30.247 | 0.9986 [0.9837, 1.0137] |
| B06 | upstream | 8 | 30.994 | 30.247 | 0.9758 [0.9617, 0.9902] |
| B08 | continuum | 12 | 145.228 | 208.855 | 1.4381 [1.4261, 1.4501] |
| B08 | upstream | 12 | 214.074 | 208.855 | 0.9755 [0.9672, 0.9839] |
| B09 | continuum | 12 | 139.732 | 211.008 | 1.5101 [1.5014, 1.5189] |
| B09 | upstream | 12 | 213.952 | 211.008 | 0.9862 [0.9824, 0.9901] |
| B11 | continuum | 8 | 72.053 | 70.406 | 0.9773 [0.9613, 0.9935] |
| B11 | upstream | 8 | 74.593 | 70.406 | 0.9439 [0.9281, 0.9600] |
| B12 | continuum | 12 | 30.402 | 30.444 | 1.0013 [0.9918, 1.0109] |
| B12 | upstream | 12 | 30.981 | 30.444 | 0.9826 [0.9716, 0.9937] |

Continuum is much faster on B08/B09; candidate is faster on B11. B08
Continuum has 67 length-capped answers in 768 responses, compared with zero
in the native arms. B09 has no caps, yet answer contents and quality still differ.
Neither global candidate dominance nor global competitor dominance follows.
B06/B12 do not resolve pressured repository retention benefit.

## Pressure and the bounded addendum

Anchor B06's eight-session cohort has zero returning CPU restores and zero
CPU-eviction-affected returns in both native arms. Preserve it as a GPU-reuse
control. B07 cannot establish removal of pressure absent from B06; B10 is a
resident team control and B12 uses the same small repository cohort. Existing
sixteen-session operational runs establish local pressure, but one process per
arm across six waves cannot replace repeated four-system timing comparisons.

Every completed secondary B06/B08/B09/B11 native process records actual CPU
pressure. Their loaded-token and eviction-affected-return totals are retained in
[the machine-readable comparisons](results/comparisons.json). For example,
Qwen3-4B B06 restored 171,568 ARC versus 347,148 candidate tokens across ten
processes; eviction-affected returns were 410/480 versus 409/480. Greater
restoration does not itself prove useful recomputation was prevented or improve
overall speed. Secondary document throughput losses coexist with mixed-traffic gains.

The addendum completed all 24 independent QA pairs and four three-arm advice
blocks (60 valid configurations). The repository purpose gate failed before
serving because the first coordinator used the test interpreter. The failure is
retained and the repository section is closed: gate replacement was forbidden.
The original study left the repeated four-system pressured repository comparison
**unmeasured**; the separately authorized recovery above now supplies it.
No capacity/policy tuning, favorable-result retries or budget reset occurred.

Fresh enabled/no-hints QA O2/O0 throughput is **0.9667 [0.9539, 0.9797]**:
a 3.33% estimated loss, with the declared 3% noninferiority bound unresolved.
The original six pairs remain separate. Misleading/ARC unhinted mean TTFT is
**1.1456 [1.1066, 1.1860]**, exceeding the 5% bound. Unhinted completion-latency
ratio is **1.1252 [1.0337, 1.2250]**, unresolved at that bound. All four blocks
record baseline CPU pressure and false-positive ready-CPU recency touches.
Accurate/ARC throughput is 1.0060 [1.0032, 1.0087]; misleading/ARC is
1.0032 [0.9953, 1.0113]. Aggregate throughput therefore conceals interference.
Class p95/p99 are recorded descriptively; 16 unhinted requests per process makes
p99 unstable. Scripted accurate hints are an information control, not a predictor.

The advice cohort offers 36 units / 84 model requests / 48 real tools per process:
16 complete held-out repository tasks, four exposed initial-only abandoned tasks,
and 16 unhinted QA requests. Four returning sessions wait at least 45 seconds;
false hints refresh abandoned sessions for 30 seconds. Ordinary declarations use
750ms, a 5s TTL and 1s refresh; this is CPU recency advice, not pinning, speculative
staging or a reserved transfer. Key-touch sums repeat keys and are not exact
saved recomputation. Tools/graphs are driver-selected, not autonomous agents.

## Quality and scope

Objective checks cover 10,616 primary final outputs and 3,504 addendum outputs.
Explicit source review covers all selected failures/samples and caps: 6,998 primary
items / 638 distinct full contexts, 2,467 addendum items / 96 contexts, and all
699 archived capped intermediate/final outputs including failed attempts.
Identical full visible contexts reuse explicit judgments, never automatic passes.
The failure-selected sample is not an unbiased population accuracy estimate.
For anchor B02, both native arms meet JSON schema in 384/384 outputs but
reference-exact matching is 132/384; the reference span appears in 312/384. Fresh
QA O0/O2 schema passes 1536/1536 each, reference exact 480/1536 versus 479/1536,
and span present 1416/1536 each. Repeated fixed questions are not independent
accuracy samples. [Full objective and reviewed-sample counts](quality-final-summary.json)
retain every group and its denominators.

Strict JSON/span/status failures coexist with valid expanded answers; other
answers invent validation failures, wrong test counts, unsupported restrictions,
or confuse historical proposals with modern runtime support. Team synthesis may
ignore contradictory specialist findings. Graph completion and served tokens are
not correctness-adjusted useful work. No answer-quality equivalence is certified.

All three models use dense full-attention GQA (32 query / 8 KV heads, head size
128), TP1 and FP16/eager on WSL2. Qwen3-8B-AWQ and Qwen3-4B-Instruct-2507 have
36 layers and 147,456 KV bytes/token; Ministral3-3B-Instruct-2512-BF16 runs
text-only in FP16 with 26 layers and 106,496 KV bytes/token. They span two
families, not hybrid/sliding/MHA/MQA architectures. Calibration freezes GPU
blocks 1024/1408/576 respectively and CPU capacity 18,944 tokens per model.
No 5080, multi-GPU, other-engine, production or broad architecture claim follows.

[Actual chart HTML](results/comparisons.html) and [all comparison inputs](results/comparisons.json)
retain all 42 pairs and observed work/caps. Raw evidence lives under
`output/stage6-2080ti-20261106/development-15/analysis/` and
`addendum-02/analysis/`; audit identities and exact paths are in the
[post-acquisition receipt](post-acquisition-audit-checkpoint.json).

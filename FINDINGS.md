# Durable findings

This index separates the new port's functional qualification from historical
research results. Historical evidence root:
`/home/philip/vllm-semantic-cache-ext/` on DESKTOP-T4JKGJB (RTX 2080 Ti/WSL2).
The links below freeze the inspected revision, not mutable main. Paths under
`output/` stay in the old repository's storage and were not copied or relocated.
Confidence intervals below are conditional historical 95% intervals, not fresh
reanalysis, population-quality estimates or selection-adjusted winner tests.

## Minimal wait-hint port, 2026-11-10

The port implements exact CPU KV offload with causal wait-hint ARC recency advice,
an authenticated endpoint, bounded observations/replay state and transfer-failure
bookkeeping. Semantic scoring and GPU prefetch are absent. The advice-disabled
control shares all correctness guards.

The pinned CPU lane passed 12 integration tests with CUDA uninitialized. On this
host's RTX 2080 Ti, three hardware tests passed: exact GPU/CPU bytes across CPU
eviction, restoration versus fresh computation in both advice modes, and real
authenticated HTTP delivery through async engine RPC. This qualifies the tested
dense FP16 TP1 configuration, not other hardware/models or performance.
[Qualification and limitations](evidence/minimal-port-20261110.md).

The lightweight fresh-agent exercise is a separate usability checkpoint, reported
in that record. Benchmark redesign and high-level agent operations remain deferred.

## Unchanged wait-hint candidate, 2026-11-05

Stage 5's fixed development pressure panel measured existing hints versus native
ARC at **+6.15% workflow throughput [4.99%, 7.32%]**, across three repeats.
No new optimization qualified; this is development evidence, not fresh confirmation.
The selected profile is CPU recency advice, not semantic scoring/GPU staging.
[Results](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage5-optimization-20261105/README.md)
and [selected settings](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage5-optimization-20261105/selected-candidate.json).

## Faster eviction did not establish serving value, 2026-11-05

Exact-state replay sped large eviction calls roughly 5–7x, sometimes 23x.
Native eviction was nevertheless below 0.05% of primary/tight workflow time.
A larger real-document batch cut 159 ms to 18 ms, with native eviction only
0.09% of workflow time and differing live outputs. The microbenchmark does not
establish an end-to-end gain.
[Accounting](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage5-optimization-20261105/EVICTION.md).
Raw states: `output/stage5-optimization-20261105/eviction-round8/` and
`output/stage5-optimization-20261105/closeout-round9/00-live-documents/`.

## Semantic/SVD costs and an inadequate advance window, 2026-11-04

On RTX 2080 Ti/Qwen2.5-1.5B, 54 cases produced 972 responses. Final public
semantic variants were 24–33% slower in requests/s than wait-hint ARC.
Public wait/hint timing gave no advance prefetch window; this bounds costs for
a scripted workload, not all semantic/prefetch opportunity. Lossy SVD changed
outputs; rank 4 failed 36/72 authored QA assertions, while rank 8's bounded QA
success did not establish general quality equivalence. Exact-KV defaults exclude it.
[Scope, measurements and caveats](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/semantic-svd-serving-20261104/README.md).
Raw: `output/semantic-svd-serving-20261104/`.

## Stage 6 gains and losses coexist, 2026-11-06 through 2026-11-08

The 2080 Ti primary study completed 524 configurations / 210 matched blocks.
Paired completed-graph-units/s ratios use whole-process log Student-t intervals.
Qwen3-8B-AWQ mixed B11 candidate/ARC was **1.1244 [1.1090, 1.1401]** (8 pairs);
Qwen3-4B document B08 was **0.9786 [0.9657, 0.9917]** and B09
**0.9766 [0.9685, 0.9848]** (10 pairs each). Benefits are workload/model dependent.
[Primary results and denominators](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage6-2080ti-20261106/RESULTS.md).

Continuum/candidate was **1.4381 [1.4261, 1.4501]** on B08 and
**1.5101 [1.5014, 1.5189]** on B09 (12 pairs each). These are serious external
baselines, but runtime, chunking and outputs differ. B08 Continuum had 67 capped
answers/768 responses against zero native caps. Neither pure policy attribution
nor global dominance follows. Pinned upstream is retention PR #59714, not #50422.
[External comparison receipts](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage6-2080ti-20261106/results/comparisons.json).
Raw primary: `output/stage6-2080ti-20261106/`.

## The original pressure premise failed; recovery was a separate study

Anchor eight-session Qwen3-8B B06 had **zero returning CPU restores and zero
eviction-affected returns** in both native arms. Preserve it as a GPU-reuse
control; it cannot establish pressured retention benefit. Secondary model cells
did show pressure. [B06 correction](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage6-2080ti-20261106/RESULTS.md).

The original addendum repository gate failed before serving due to interpreter
selection. Separately authorized recovery completed 18 configurations, including
four complete four-system pressure blocks. Candidate/ARC was **1.0210
[0.9885, 1.0546]**; the gain remains unresolved. All four ARC blocks had actual
pressure (25 eviction-affected returns each). Preserve the failure and recovery
as distinct identities; do not rewrite the old gap as a pressure win.
[Recovery protocol, counters and audit](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage6-2080ti-20261106/repository-recovery-03/README.md).
Raw recovery: `output/stage6-2080ti-20261106/addendum-03/`.

## Enabled without hints and misleading hints harmed traffic

Fresh QA enabled/no-hints throughput was **0.9667 [0.9539, 0.9797]**, an estimated
**-3.33%** across 24 pairs; the declared 3% noninferiority condition remained
unresolved. Misleading-hint/ARC unhinted mean TTFT was **1.1456
[1.1066, 1.1860]**, or **+14.56%**, exceeding the 5% bound across four advice
blocks. Aggregate throughput concealed interference. Per-process unhinted p99
has only 16 requests and is descriptive.
[Addendum outcomes and traffic definitions](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage6-2080ti-20261106/RESULTS.md).

## Answer-quality equivalence remains unestablished

Objective coverage was 10,616 primary and 3,504 addendum finals, with explicit
selected source/cap reviews. Unsupported claims, invented validation, format
failures and truncation remain. Failure-selected review denominators are not
population accuracy estimates; completed graphs and served tokens are not
correctness-adjusted utility. Recovery audited all 288 finals and selected
272 source items, without establishing quality equivalence.
[Quality denominators](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage6-2080ti-20261106/quality-final-summary.json)
and [recovery audit](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage6-2080ti-20261106/repository-recovery-03/quality-final.json).

## 5080 binary capability

On DESKTOP-E78LJSI, the November 8 campaign failed all five Continuum anchor
warmups at `multi_layer_kv_transfer`: the LMCache binary had neither sm_120 nor
PTX fallback. All five completed zero measured requests. Integrity of 41 attempts /
3,142 files did not establish successful acquisition or hardware qualification.
[Closed failure](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261108/RESULTS.md)
and [capability receipt](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261108/capability-failure.json).
Raw remote: `/home/philip/vllm-semantic-cache-ext/output/stage6-5080-20261108/`.

The November 9 same-source LMCache 0.3.6 rebuild passed an exact **393,216-byte**
bidirectional transfer, all five Continuum screens, and 54 screen configurations /
22 matched blocks. It repaired the binary capability problem, not the entire
study. At the pinned snapshot confirmation/overhead/operational/final audits were
incomplete. B06 had 31 valid measurements/32 attempts, seven/eight complete
blocks: one candidate had zero restores plus preemptions; it stays excluded.
[Incomplete results](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261109/RESULTS.md),
[build](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261109/lmcache-build.json),
[qualification](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261109/qualification.json).
These pinned result files were re-read over read-only SSH for this foundation.
The current live campaign state was not inspected; do not treat this snapshot as
its final outcome or compare hosts as a GPU-only experiment.

## Historical credential exposure and earlier work

A webhook credential appeared in the remote historical
`evidence/stage6-5080-20261108/CONTINUE.md`, line 174 before the commit titled
“Keep completion endpoint private.” Carry only the fact/location. No value or
private endpoint file was read or copied in this step; revocation is **unverified**.
The location was established in the earlier read-only porting investigation;
the secret-bearing historical text was deliberately not reopened.

Earlier compact work remains available under the old
[a27](https://github.com/philippesic/vllm-msr-cache/tree/1319f09bafac46bab89df465dd433c67441f4b26/evidence/a27),
[e04](https://github.com/philippesic/vllm-msr-cache/tree/1319f09bafac46bab89df465dd433c67441f4b26/evidence/e04)
and [live wait-hint investigation](https://github.com/philippesic/vllm-msr-cache/tree/1319f09bafac46bab89df465dd433c67441f4b26/evidence/live-optimization-2080ti-20261103).
Those scopes are not requalified here. The complete external inventory reconciles
7,438 old tracked paths; that foundation inventory imported no old code/history/raw
outputs. The subsequent minimal source extraction is documented separately above.

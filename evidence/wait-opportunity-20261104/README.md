# Wait-hint opportunity in live repository serving

**Use `make bench-agent-pressure` for this local workflow comparison.** Refined
CPU wait-hint ARC delivered a repeatable gain over ARC: **+7.00% completed-workflow
throughput, +6.23% output-token throughput, and 0.430 s lower average first-token
latency**. Three fresh rotated repetitions support these estimates. Adding the
refined semantic policy did not improve on wait-hint ARC: workflow throughput was
3.01% lower. Keep early CPU wait retention for development; keep semantic ranking,
SVD and prefetch experimental. No deployment or qualification change was made.

## Final comparison

Qwen2.5-7B-Instruct-AWQ, FP16, one RTX 2080 Ti, FULL_DECODE_ONLY graphs; sixteen
sessions / sixty-four requests per case. The final campaign has nine cases,
576 successful EOS responses, 432 actual tool executions and 144/144 correct
reported test outcomes. Every case passed native/endpoint source, request-count
and preemption accounting (45 identities); every initial prompt matched across
policies within each repetition. Later live prompts/outputs differ and remain
retained, so this is a closed-loop workflow experiment, not identical-work causal
attribution. The confidence intervals apply to these inputs and hardware.

| Metric (mean per case unless noted) | ARC | ARC + early CPU wait hints | Semantic + wait hints |
| --- | ---: | ---: | ---: |
| Workflow time, s | 106.448 | 99.486 | 102.580 |
| Workflows/min | 9.019 | 9.650 | 9.359 |
| Output tokens/s (pooled tokens / elapsed) | 119.25 | 126.67 | 122.90 |
| Offered first-token mean, s | 7.890 | 7.461 | 7.637 |
| Returning HTTP first-token mean, s | 6.113 | 5.522 | 5.726 |
| Tool-ready to first-token mean, s | 6.124 | 5.532 | 5.736 |
| Offered first-token p95, s | 21.398 | 21.475 | 21.541 |
| First-admission computed prompt tokens | 63,982 | 51,927 | 54,841 |
| First-admission CPU-restored prompt tokens | 37,125 | 57,360 | 56,485 |
| Returning compute above completed-prefix floor | 14,923 | 2,869 | 5,781 |
| Completed physical load traffic, GiB | 1.983 | 3.063 | 3.017 |
| Preemptions, total | 0 | 0 | 0 |
| Mean sampled device peak, MiB | 8,736 | 8,737 | 8,790 |
| Server-tree CPU seconds | 116.22 | 108.57 | 111.84 |
| Client/tools CPU seconds | 140.18 | 140.18 | 141.63 |
| Sampled GPU energy, kJ | 25.851 | 24.004 | 24.470 |
| Applied semantic update batches, total | 0 | 0 | 128 |
| CPU-key refresh observations, total | 0 | 330,075 | 315,154 |
| Reported test-status checks correct | 48/48 | 48/48 | 48/48 |

Ratios use geometric paired whole-case estimates; absolute values above are
summaries, so their ratios can differ slightly. Three pairs give:

| Paired estimate | Wait / ARC | Semantic / wait |
| --- | ---: | ---: |
| Workflow rate change (95% interval) | +7.00% [+5.90%, +8.10%] | -3.01% [-5.83%, -0.12%] |
| Output-token rate change (95% interval) | +6.23% [+5.43%, +7.03%] | -2.98% [-5.42%, -0.47%] |
| Offered first-token mean difference, s (95% interval) | -0.430 [-0.817, -0.042] | +0.176 [-0.225, +0.578] |

Semantic / ARC workflow rate was +3.77% with interval [-0.12%, +7.82%], which does
not resolve a workflow-rate win. Its comparison to wait ARC is the relevant
incremental test. Applied-score batches prove the scoring pipeline activated;
per-victim counterfactual score effects were not traced in the timed campaign. First-token tails are dominated by the initial server queue;
there is **no p95 latency improvement or VRAM saving**. Semantic buffers add about
54 MiB device peak. Broader reasoning/task-success quality remains ungraded;
correct test-status reporting alone cannot establish unchanged answer quality.

[Compact aggregates and paired intervals](comparison.json),
[all attempts in JSON](results.json), [all cases in CSV](results.csv),
[activation/accounting and matched initial prompts](activation.json),
[offline artifact verification](verification.json),
[sealed final report](../../output/wait-opportunity-20261104/bounded-confirm/REPORT.md),
[sealed final result](../../output/wait-opportunity-20261104/bounded-confirm/result.json).

## What helped, what hurt, and the stopping decision

The client previously limited active HTTP requests to the server's sequence limit,
which hid waiting requests from the scheduler. Independent admission now exposes
the queue. CPU backing also needs to outlive GPU residence: 2,048 CPU blocks below
2,560 GPU blocks produced almost no useful CPU restores. Raising backing to 3,072
blocks restored real cold prefixes; making it large enough for everything instead
made wait advice redundant. Enough GPU space for the active batch matters too:
1,280 blocks caused wait's extra restores to increase preemptions from 11 to 32,
trading less original-prompt work for more physical traffic and worse throughput.

The optional GPU-residency gate was the last major retention obstacle at the
selected capacity. It skipped CPU advice while the full prefix was on GPU, giving
CPU roots time to disappear before GPU eviction. Protecting already-ready CPU
prefixes earlier gave +6.86% workflow rate in the selecting screen, then the final
bounded-worker confirmation reproduced the gain. This is an explicit **ungated
CPU-only experimental profile**; it is not a claim that the previous gated default
reproduces the historical 6%/500 ms result under every geometry. Both arms have
identical scheduling credit and hint production, so those common features are not
credited as wait-hint gains. CPU-only advice reserves no GPU blocks; native
speculative jobs/bytes stayed zero in all final arms.

Concurrent tools had caused multiple GiB of host swap-out. Four real subprocess
workers reduced that substantially; tool queue and execution clocks are separate.
The first ARC case still swapped out about 368 MiB and the final wait case
about 306 MiB; subsequent ARC cases had about 3 and 1 MiB and still lost to wait.
Swap-in was small throughout the final campaign. All raw host counters remain available. The final comparison includes
real tool cost, queueing, hint RPCs, score capture, scoring, cache policies,
transfers, auditing and telemetry. No uninstrumented-overhead estimate is claimed.

Plain mean-semantic retention at stride 8 was active but recomputed more prompt
work and took 97.5 s against wait's 88.7 s with unbounded tool workers. The refined
profile uses existing prefix/chain protection and stride 2. Under the bounded tool
profile it came much closer, but remained slower in all three repetitions and
computed about 2,915 extra prompt tokens per case versus wait. Worker limits,
structural protection and sampling changed together between these configurations;
their individual contributions were not isolated.

Stop this hill climb here. Wait ARC removed about 81% of returning computation
above the completed-prefix transcript bound, leaving roughly 2,869 tokens per
case above that bound. The semantic candidate doubled that residual and added
memory/scoring cost. Attention salience has not supplied useful additional
exact-prefix reuse information in these fully returning histories. More arbitrary
capacity or alpha sweeps lack a supported path to a useful incremental gain;
a new reuse predictor or a different real application would need fresh evidence.
The benchmark has now established a useful operating point, rather than rejecting
a broken or inactive first run. Recommend early CPU wait retention and this
pressure profile for local development, with broader quality and workload
qualification still open.

## Retained experiments

All nine artifact sets verify offline, including their original failed or
inconclusive measurement status. There are 26 complete cases plus two deliberately
interrupted cases, 1,665 raw request records / 1,636 successful responses. The
final confirmation alone is the nine-case, 576-response comparison above.
Selecting screens and differing operating points are excluded from its interval.

| Run | What it established |
| --- | --- |
| [screen-medium](../../output/wait-opportunity-20261104/screen-medium/REPORT.md) | Sixteen sessions, 1,280 GPU / 2,048 CPU blocks. Failed original source-accounting identity; exploratory timing only. |
| [screen-large](../../output/wait-opportunity-20261104/screen-large/REPORT.md) | 4,096 CPU blocks retained the needed working set; advice had little useful work to save. Original per-request/global-byte equality failed. |
| [fast-screen](../../output/wait-opportunity-20261104/fast-screen/REPORT.md) | Eight sessions / four server sequences. Accounting passed; shorter wait workflow had fewer output tokens, worse token rate/FTL and no saved prompt work. |
| [pressure-confirm](../../output/wait-opportunity-20261104/pressure-confirm/REPORT.md) | First corrected pair passed; insufficient GPU working-set space made extra restores/preemptions harmful. Remaining repetitions intentionally stopped and retained. |
| [headroom-screen](../../output/wait-opportunity-20261104/headroom-screen/REPORT.md) | 2,560 GPU / 2,048 CPU blocks removed preemptions but almost all CPU reuse disappeared. Original baseline-positive-restore gate made ARC inconclusive; zero hits are a legitimate baseline outcome. |
| [backing-screen](../../output/wait-opportunity-20261104/backing-screen/REPORT.md) | 3,072 CPU blocks enabled real reuse and competition. Gated wait was nearly tied on workflow rate (+0.23%). |
| [ungated-screen](../../output/wait-opportunity-20261104/ungated-screen/REPORT.md) | Early CPU protection selected: +6.86% workflow rate, -0.461 s offered FTL, fewer computed tokens. |
| [three-way-confirm](../../output/wait-opportunity-20261104/three-way-confirm/REPORT.md) | First ARC/wait/mean-semantic comparison complete. Host-memory and prefix-ranking issues prompted a deliberate stop before further repetitions; partial next case retained. |
| [bounded-confirm](../../output/wait-opportunity-20261104/bounded-confirm/REPORT.md) | Final nine-case rotated comparison, complete and passed; primary results above. |

## Reproduction and implementation

The benchmark now separates offered HTTP requests, the server's running batch,
and actual subprocess workers. It measures a burst of sixteen distinct repository
investigations, each with four live model responses and three actual tools.
There are no simulated tool sleeps, replayed responses, padded prompts or known
future tool results. Requests retain real growing histories and compete for CPU
and GPU KV capacity.

Use the maintained [pressure command](../../docs/BENCHMARKING.md#real-repository-agent-workflow).
`make bench-agent-pressure PY=... MODEL=... OUT=...` runs ARC versus explicit CPU
wait-hint ARC. Add `--repeats 3 --semantic-config benchmarks/research_checks/fixtures/semantic_wait_prefix.json` through `BENCH_ARGS` for rotated ARC,
wait and semantic comparisons. Each output directory is new and sealed; verify
with `make bench-verify PY=... OUT=...`.

The selected profile is sixteen HTTP sessions, eight server sequences, four real
tool workers, 2,560 GPU KV blocks / 2,240 MiB, and 3,072 CPU KV blocks / 2,688 MiB.
The GPU cache has headroom for active requests while the full growing workload
exceeds cache capacity. Both arms use the same 8-second continuation scheduling
credit. Hints protect the completed prefix before a tool starts, during actual
tool-worker waiting/execution, and through returning-request queue residence until
first token. Removing the optional GPU-residency gate allows early CPU protection.
All identity, ownership, no-speculative-dispatch and runtime audit guards remain.
This is isolated experimentation, with no deployment.

Read [the protocol and retained refinements](PROTOCOL.md) for the selecting
screens, accounting repairs, controls, incomplete campaigns and planned analysis.
All runs are finished and offline-verified; failed/inconclusive measurement
statuses remain unchanged. No further GPU work is active.

## What the measurements mean

- Completed workflows per second include all four responses, actual tool waits,
  tool-worker queueing, client admission and the final drain. Request rate is
  exactly four times workflow rate when every session completes. This is a bounded
  burst workflow rate, not an indefinitely sustained production capacity estimate.
- Offered first-token latency starts when the application has the model request
  ready. It includes client and server queues. Tool-ready first-token latency also
  includes transcript assembly and submission. HTTP TTFT is retained separately.
- First-admission computed/local-GPU/CPU-restored prompt sources are cross-checked
  against native endpoint counters. Cumulative original-prompt work and physical
  transfer traffic are separate. A CPU-restored token is a cache source observation;
  a net speed benefit requires less work or lower waiting/elapsed time after costs.
- Positive CPU refresh and semantic score-update counters prove activation only.
  Prior CPU eviction of a returning prefix establishes competing reuse; GPU
  allocation alone does not. Reclaimable GPU blocks include idle cached prefixes.
- GPU memory, server/tool CPU and RSS, power/energy, host available memory/swap,
  queue occupancy, transfers, preemptions, hint RTT/deadlines/cancellations and tool
  execution/queue times remain available for diagnosis. Sampling can miss peaks;
  summed RSS can double-count shared pages. Unsupported metrics remain unavailable.
- Quality checks compare final reported test status to the real subprocess exit
  status. Strict formatting, contradictions and unparseable claims are separate.
  Code-reasoning correctness and task resolution are ungraded. Exact prompts,
  streams, outputs, commands, stdout and fingerprints remain available for review.

The semantic arm replaces ARC eviction with the project's SemanticPolicy and
composes it with CPU wait advice. It does not preserve ARC internally. The refined
arm uses existing prefix/chain protection and more frequent score capture. KV
values remain exact; SVD and GPU prefetch are disabled in this comparison.
Their prior results are preserved, and this new workload does not qualify them.

Compact event emission preserves runtime audit logic, source/traffic counters,
timings and eviction observations while dropping verbose serialized transport/key
histories. Such logs are explicitly not complete KV byte/lifetime proofs. Initial
full-detail runs are retained. Full measurement costs are included in both arms;
there is no independent uninstrumented-overhead estimate.


## Verification and handoff

453 affected pinned-runtime CPU tests passed; `make check-light` passed 31 portable
tests plus compilation. The maintained suite tests cover independent HTTP
admission, bounded real tool workers, hints stopping at first token, priority,
semantic advice without score/reference fabrication, first-admission source
snapshots, compact audit receipts, paired intervals and offline aggregate checks.
`git diff --check` passed. Verification re-read all nine seals, re-derived raw
request/agent aggregates and, for the final campaign, recomputed primary paired
comparisons. Failed accounting runs remain failed; artifact verification does not
turn them into valid policy wins.

Prior dirty files were backed up before edits. Forty-three unrelated pre-existing
files still match their captured hashes, including the earlier SVD transport,
worker/spec changes, qualification records and vendored scheduler patch. The
original backup and changed-file patch remain protected under
`output/wait-opportunity-20261104/`. Intended implementation changes remain
uncommitted for review. No deployment, push, shared-setting change or resumed AWQ
work occurred. [Cleanup proof](cleanup.json), [prior-work preservation](preservation.json), and
[bounded retained inventory](../../output/wait-opportunity-20261104/storage-inventory.json).
All useful raw evidence/recovery (about 1.54 GiB) remains protected. Removed 28
empty task-owned import-stage directories. WSL free space measured before/after
cleanup was 906,833,215,488 / 906,833,330,176 bytes (observed +114,688 bytes; this
is not a Windows backing-volume reclaim claim).

# Semantic cache and actual KV SVD serving comparison

**Scope correction after workload review:** these runs used small, scripted
repository-summary conversations and recorded tool text. The final public waits
were 0.2 and 1.0 seconds, while hint publication delayed by `min(wait, 1.0)`;
therefore they provided no advance tool-wait window for prefetch. QA runs did
exercise consumption. The public results below remain valid measurements of
that workload's costs and outputs, but the broad stopping recommendation does
not establish that semantic techniques or wait hints lack benefit on sustained,
contended agent workflows. A real executed-tool benchmark is being developed in
`output/real-agent-eval-20261104/`; historical artifacts remain unchanged.

Keep wait-hint ARC for the previously established operating scope. This bounded
experiment did not establish a useful semantic/SVD/prefetch improvement over it.
The implementation is isolated behind default-off options; nothing was deployed.
Prior qualification work and paused AWQ work were preserved.

54 real serving cases completed 972 workload responses on the RTX 2080 Ti with
Qwen2.5-1.5B-Instruct, FP16/TRITON_ATTN, actual generated histories, equal GPU
capacity and matched nominal CPU budgets. The maintained serving suite supplied
HTTP generation, metrics, resource sampling, process cleanup and offline
verification; a thin versioned coordinator supplied factorial arm order.

## Final public-input comparison

One six-arm pass, six public sessions × three turns per arm, 128 GPU blocks,
80 MiB nominal CPU KV budget, concurrency two. Rank 8 expands compressed slot
capacity, stride 8 samples scoring, hints arrive one second into existing waits,
prefetch uses a 25% budget and one-second lease. All 108 responses reached EOS;
none hit the 128-token cap. This is a bounded scripted workload, not deployment
or a population-wide confidence estimate. Inputs reuse retained public sources.

| Variant | Requests/s | Output tokens/s | Mean first token (s) | p95 first token (s) | Sampled GPU peak (MiB) | Summed RSS peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC | 0.644 | 35.08 | 1.082 | 2.985 | 4044 | 3397 |
| Wait-hint ARC | 0.648 | 35.27 | 1.079 | 2.946 | 4044 | 3406 |
| Semantic base, both off | 0.437 | 24.78 | 1.710 | 4.005 | 4050 | 3447 |
| Base + KV SVD | 0.490 | 25.21 | 1.587 | 4.345 | 4334 | 3521 |
| Base + prefetch | 0.439 | 24.91 | 1.702 | 4.012 | 4050 | 3441 |
| Base + both | 0.466 | 24.40 | 1.401 | 4.450 | 4332 | 3546 |

Semantic variants were 24–33% slower in completed requests/s than wait-hint ARC.
Different live outputs/history lengths prevent interpreting token throughput as
an isolated policy/kernel speedup. ARC and wait-hint ARC matched all 18 outputs;
base/prefetch matched 14 and SVD variants matched eight. RSS sums shared pages;
GPU samples cover the whole device, not exact allocator peaks.

## What activated, helped and hurt

- **Actual KV compression:** both K and V page values are factored, retained as
  FP16 low-rank factors on CPU, and reconstructed into full GPU pages before
  completion. No uncompressed CPU shadow exists. This is not feature SVD or
  reduced-rank attention. Rank 8 retains 53.125% of KV payload at equal slot
  count; expanding slots buys about 1.88× capacity at equal payload budget.
- **Codec refinement:** per-layer codec calls were batched; a subsequent solver
  batch limit cut the large temporary GPU workspace. A supplemental cost probe
  measured about 928 MiB reserved at batch 2800 versus 134 MiB at batch 256,
  with about 4 ms extra warmed codec time. Serving confirmed a much smaller
  penalty, but final public SVD still used about 290 MiB more sampled VRAM.
  Larger slot counts and workspace also raised measured RSS.
- **Semantic ranking:** applied native relevance receipts were positive (93–106
  update batches in final public semantic cases). Prior CPU evictions were
  observed under pressure. Ranking/scoring bookkeeping cost exceeded useful
  avoided work; this experiment does not establish predictive ranking value.
- **Prefetch:** delayed publication activated ready-CPU refreshes, semantic
  nominations, speculative transfers and actual consumption in QA workloads.
  In the refined repeated QA cases, each combination consumed three sources
  after five submitted jobs. In the public workload, all five jobs in each
  prefetch arm completed, but none was consumed: demand reclaimed 63 blocks
  without SVD and 196 with SVD. Public transfer traffic and nomination cost
  therefore bought no speculative hit.
- **Reservation repair:** 60% staging with a five-second lease caused repeatable
  six-second first-token tails in the rank-8 combination. Reducing staging to
  25% and the lease to one second removed that tail: mean QA first-token latency
  was 254–267 ms and per-case p95 391–503 ms, still behind wait-hint ARC's
  168–179 ms mean. Ownership fences and demand reclamation remained enabled.
- **Diagnostics off:** the final source disabled codec residual collection and
  JSONL codec logging for a separate serving check. Wait-hint ARC completed
  3.132 requests/s, SVD 2.781 and the combination 2.766. Corresponding mean
  first-token latencies were 186, 242 and 254 ms. Observer cost did not explain
  the deficit. Native transfer intervals in this final source use CUDA events;
  full host codec cost was separately recorded in earlier codec JSONL receipts.

## Answer quality

The authored retrieval fixture checked unique recovery codes and badge colors.
Rank 4 expanded capacity about 3.76× but failed nine of 18 assertions in each of
four balanced cases (36 failures): invented codes, wrong colors and longer wrong
answers inflated token throughput. It is rejected on quality, not speed alone.

Rank 8 passed all assertions in both balanced final QA repetitions and the
lease/diagnostic follow-ups. Its differing QA outputs preserved facts and changed
format/case. This does **not** establish general answer-quality equivalence.
Public summaries changed substantively, including unsupported conclusions about
repository observations; the controls also made unsupported inferences. The
selected [grounding review](../../output/semantic-svd-serving-20261104/quality-review.json)
is qualitative, not a blinded accuracy estimate. Scheduling changes outputs even
without SVD, so the public differences cannot all be assigned to compression.

## Bounded hill climb and stopping decision

The first small-cache screen lacked complete prefixes; it was retained as an
inconclusive diagnostic, not a verdict. Follow-ups increased CPU capacity until
real restores were substantial, delayed hints until prefixes were cold, and used
the existing partial-prefix path when a shared GPU system block prevented the
full-cold-prefix profile from activating. Exact identity and allocation/lifetime
checks were retained. Tested branches included ranks 4/8, same-slot storage savings
versus same-byte capacity expansion, stride 1/8, per-layer/batched/bounded codec
execution, early/delayed hints, full/partial prefixes, staging/lease reduction,
reverse-order repeats and diagnostic-off execution.

Stop here: more aggressive compression damages quality at rank 4; rank 8's
extra capacity still loses after workspace and observation-cost refinement.
Dense attention still requires reconstructed GPU pages, so this codec saves host
payload, not resident GPU KV capacity. Speculative copies compete with demand and
are frequently wasted. The semantic base already loses before adding either
technique. Further knob tuning on these measured workloads has no supported path
to a net quality-preserving gain. A different asynchronous codec or low-rank
attention kernel would be a new implementation hypothesis, not a demonstrated
improvement or a reason to deploy this one.

Nothing remains externally blocked. Untested: other GPUs/models, wider ranks and
layouts, asynchronous codec ownership, long natural tool waits/real traffic,
indefinite endurance, and blinded application quality. No general rejection of
all semantic caches or all low-rank KV methods follows from this bounded result.

## Evidence and reproduction

- [Portable case summaries and activation receipts](results.json)
- [Original protocol and recovered design](PROTOCOL.md)
- [Final checks and cleanup](verification.json)
- Raw runs, exact plans/configurations, prompts, outputs, SSE, resource samples,
  source/model hashes, manifests and retained failed setups:
  `output/semantic-svd-serving-20261104/` (ignored, local).
- `final-qa/`: balanced 12-case rank-8 factorial with the larger staging budget.
  `lease-refinement/`: six matched repaired-prefetch cases.
  `public-final/`: all six arms on public inputs; `observer-off/`: final-source
  diagnostic-off serving. Other directories retain the preceding experiments.
- Reusable coordinator: `python -m benchmarks.research_checks.semantic_svd_campaign
  --help`. Supply local model, corpus, new output, CPU/GPU geometry and explicit
  arm/rank/stride/prefetch knobs. It holds the shared GPU lock across idle gaps.
- Run `python -m benchmarks.research_checks.semantic_svd_report RUN --seal` only
  when a run is finished. `make bench-verify PY=... OUT=RUN` checks its sealed
  artifacts through the maintained verifier. A passing measurement is not a win.

Older native request records label transfer time as CUDA-event time even though
initial experimental codec results supplied full host wall time. Historical
codec JSONL `seconds` is the authoritative codec cost; it includes residual
measurement and synchronization. The final source corrects the native interval
measurement. Compressed physical transfer bytes must not be mixed with logical
uncompressed staging/consumption bytes. Full raw records remain unmodified.

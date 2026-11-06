# Four-system repository workflow comparison

Wait ARC and corrected upstream were close in this single pass: **101.628 s versus
102.711 s**, a 1.05% difference in batch time. Upstream produced more output and
had the slightly higher token rate. These observations do **not** establish a
winner, repeatability, or isolated retention-policy superiority. Runtime/backend
differences, unequal observed host paging, live output differences and an explicit
orchestration interruption remain material limits. Continuum uses the predeclared
supported decode-publication correction described below; its earlier default
false row remains diagnostic.

Each primary configuration ran once: sixteen workflows, 64 EOS model responses and 48 successful real
tools, with the same initial prompts, local Qwen2.5-7B-Instruct-AWQ weights,
FP16, eager execution, eight server sequences, four tool workers, 2,240 MiB GPU KV
and 2,688 MiB pinned CPU backing. Plain ARC produced and processed no wait hints.
Semantic scoring, SVD and speculative transfers were disabled in the local arms.

| System | Batch time (s) ↓ | Output tokens | Tokens/s ↑ | Tool return to first token, mean / p95 (s) ↓ | Strict final status format |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARC without hints | 108.153 | 7,405 | 68.468 | 6.927 / 12.578 | 2/16 |
| Active wait ARC | 101.628 | 7,453 | 73.336 | 5.258 / 11.771 | 2/16 |
| Continuum public preview, decode publication enabled | 109.108 | 7,693 | 70.508 | 4.634 / 26.282 | 6/16 |
| Upstream draft + host-pointer compatibility fix | 102.711 | 7,574 | 73.741 | 5.443 / 11.520 | 2/16 |

Batch time includes live tools, queueing and all hint production/processing; model
startup and the identical tiny warmup are excluded. Return latency begins at the
actual tool completion. Continuum had the shortest mean return latency and the
longest observed p95. A p95 from 48 returns is descriptive, not a tail guarantee.
Wait's batch was 6.03% shorter than plain ARC here; later live outputs differ even
between those two arms, and one pair provides no repeatability estimate.

**Quality is narrowly checked.** All four arms reported the actual passing test
outcomes correctly under the frozen `normalized-status-v2` scorer (16/16 each),
with no contradictory or unparseable outcomes. That scorer accepts normalized
forms such as `test status: PASS`; strict requested `test_status=PASS` formatting
failed 14/16 ARC, 14/16 wait, 10/16 Continuum and 14/16 upstream. The scorer is
byte-identical to the earlier five-way frozen scorer, not a post-run relaxation.
Neither score grades code reasoning or task resolution; the strict-format failures
remain failures. [Scorer provenance](../../output/fair-four-20261104/quality-provenance.json).

## Controls and remaining differences

The reviewed [predeclared manifest](FAIRNESS.md) fixes the setup and its known
limits. Native upstream CPU eviction now explicitly uses ARC, and its bridge
forwards the same application priority/8-second continuation credit as both local
arms to the native priority scheduler. All actual allocations passed checks before
measured requests: 2,818,572,288 pinned CPU bytes and 2,348,810,240 GPU KV bytes.
Continuum's effective GPU token capacity and its actual LMCache allocation were
checked; native arms were checked from worker tensor bytes. Equal byte budgets do
not imply equal effective cache behavior.

Local/upstream CPU chunks are 16 tokens; Continuum's LMCache chunks are 256 tokens.
Continuum retains its own scheduler, LMCache LRU backend and published V1 runner;
it omits the paper's automatic timeout estimator. Its LMCache CPU KV backing is
pinned, while its older vLLM input buffers use the supported WSL unpinned path.
The initial Continuum row missed a supported publication control: its default
`save_decode_cache=false` differed from the native arms. That row is now diagnostic.
Following the [predeclared correction](REPLACEMENT.md), the primary row enables
`LMCACHE_SAVE_DECODE_CACHE=true` and was chosen before timing irrespective of its
result. The earlier false row took 107.303 s / 7,560 tokens / 70.455 tokens/s;
it is retained in [the initial table](comparison-initial-four.json), with no claim
that the two live-output rows isolate the flag's effect. The correction used one
fresh pass with the same selected corpus, prompt construction, tools, budgets,
256-token chunks, scheduler and warmup. It adds read-only decode-save observation;
its cost is included and not separately isolated.

Versions differ: local vLLM 0.26 development / torch 2.13, upstream vLLM 0.30
development / torch 2.13, and Continuum vLLM 0.10.2 / torch 2.8 / LMCache 0.3.6.
Attention uses the versions' Triton paths; quantization kernels differ (native
AutoAWQ Marlin versus the older Continuum AWQ path). Continuum's pinned Marlin
config requires compute capability 8.0; this card is 7.5, so that path is not a
supported equalization flag for this runtime. The external arms also use a
thin AsyncLLM HTTP bridge and different native instrumentation. Incomparable
external request-source/recomputation accounting remains unavailable.

The upstream row is explicitly **compatibility patched**. The pristine published
revision, exact [host-pointer fix](../upstream-pin-fix-20261104/mapped-host-pointer.patch),
all actual Python including submodule support, and native content hashes are
retained. Preflight reconstructs the changed module from published source plus
patch and rejects other Python changes. A separate V2 pinned-memory probe passed
three identical 32-token outputs and two 170,655,744-byte CPU restores. Original
published checkouts remain clean; no unpinned or older-runner fallback was used.

The initial predeclared shuffle was wait → ARC → Continuum(false) → upstream,
seed 17012. The explicitly selected Continuum(true) replacement ran afterward;
the primary table spans those invocations rather than an uninterrupted sequence. Model
and OS caches were already warm, with no cache flushing, clock locking or system
settings changes. **Observed host paging differed substantially:** measured-window
swap-out was 783.1 MiB wait, 242.6 MiB ARC, 0.79 MiB corrected Continuum and 0.26 MiB upstream;
swap-in was 13.31, 9.51, 0.18 and 0.25 MiB. GPU temperatures ranged 30–50°C across
arms and clocks varied. These are additional confounds, not inferred causes or
corrections. This is not a fully matched host-state ranking.

## Activation, interruption and verification

All 256 responses stopped at EOS; no context/output caps or response errors were
observed. All 192 tools succeeded. Initial token prompts match across every pair.
Initial outputs match 16/16 between local arms, but only 1/16 local versus each
external system; later prompts and responses diverge in every pair.

- Plain ARC: zero hint publications, hint RPCs, observed-prefix metadata and hint
  processing events. Actual native stores/restores and cache competition observed.
- Wait ARC: 666 publications, 622 CPU recency refresh acknowledgements and all
  48 tool windows with advance acknowledgements; actual stores/restores observed.
- Corrected Continuum: 23 published scheduler pins, seven positive LMCache
  retrieval log records and 35 measured decode-save decisions, all matched by
  request ID/token boundary to native completed-store records. The initial false
  row had 22 pins and nine retrieval records. These are backend observations,
  not our request receipt protocol or KV byte-equality proof.
- Upstream: 48 applied priority updates, 550 priority selections, 33 successful
  native restores totaling 3,267,231,744 bytes, with pinned mmap and 28 corrected
  device aliases. These observations prove activity, not counterfactual victim value.

The main coordinator stopped **after three complete, cleaned measurement arms**
because its post-case source check lacked an import. No upstream measurement had
started. The approved bounded continuation ran upstream exactly once through the
same unchanged frozen `run_case`, plan, corpus and runtime identities under both
GPU locks. Previous case files hash-match their original seal. The coordinator
began 158.027 seconds after the third case's cleanup; the fourth case began
168.346 seconds after that cleanup. Gap/host state and original failure are
retained. **This was an interrupted and explicitly continued comparison.** The
[post-run one-line import fix](post-run-coordinator-fix.patch) is outside the
initial four-arm measured source, included in the replacement freeze, and tested.
No initial arm was rerun for that recovery. The later Continuum correction is
separately declared and selected by configuration matching, not by timing.

- [Exact numbers, quality and resource audit](comparison.json), with
  [primary-selection verification](../../output/fair-four-20261104/primary-selection-verification.json).
- [Initial sealed report](../../output/fair-four-20261104/comparison-final/REPORT.md)
  and [result](../../output/fair-four-20261104/comparison-final/result.json), which
  retain the diagnostic false Continuum row unchanged. The primary replacement is
  [separately sealed](../../output/fair-four-20261104/continuum-decode-matched/REPORT.md)
  with [its own result](../../output/fair-four-20261104/continuum-decode-matched/result.json).
- [Continuation identities and host state](../../output/fair-four-20261104/comparison-final/continuation.json),
  [frozen recovery coordinator](../../output/fair-four-20261104/comparison-final/continuation-coordinator.py),
  [original failure](../../output/fair-four-20261104/comparison-final/orchestration-failure.json).
- [Offline verification](../../output/fair-four-20261104/verification.log): 516 files,
  four initial cases and 256 request records; original comparisons remain
  inconclusive. The [replacement verification](../../output/fair-four-20261104/verification-decode-matched.log)
  covers another 139 files / 64 records; the primary selection is three initial
  rows plus the corrected Continuum row, 256 primary responses / 192 tools.
  The extra 64-response false row is diagnostic, not another primary repetition.
  [Decode activation proof](../../output/fair-four-20261104/continuum-decode-activation.json)
  matches all 35 decisions to completed stores. Both source sets retain the
  identical preexisting quality scorer.
- [Benchmark checks including decode observation](../../output/fair-four-20261104/bench-tests-decode.log):
  124 passed; [portable checks](../../output/fair-four-20261104/check-light.log):
  31 passed plus compilation (also repeated after correction in
  [final checks](../../output/fair-four-20261104/check-light-final.log)).
  The import fix itself required no further GPU run; publication matching did.
- [Cleanup](../../output/fair-four-20261104/cleanup.json): all case cleanup receipts
  pass, GPU has no compute clients and both locks are available.
- [Storage inventory](../../output/fair-four-20261104/storage-inventory.json):
  approximately 269 MiB retained, including useful failed preparation/source
  freezes. Disposable temporary reconstruction files were automatically removed;
  shared runtimes/caches and paused work remain preserved. No commit, push or deployment.

## Reproduction

The exact runtime manifest is
[here](../../output/fair-four-20261104/runtimes-decode-matched.json); it requires the retained
isolated installations and explicitly labeled compatibility source. The
[benchmark guide](../../docs/BENCHMARKING.md#explicit-four-system-comparison)
describes four-arm controls. Use the full command and settings in the sealed
`plan.json` with a fresh output path, explicit four-arm selection and order seed;
set `LMCACHE_SAVE_DECODE_CACHE=true` for Continuum as in the replacement manifest.
The two retained source freezes and correction declaration explain this run's
provenance.
Do not reuse earlier three-/five-way timings as additional repetitions.

# Last-peer hint activation: causal alternative, unmeasured

Read-only review, 2026-11-01. No source/harness changes, imports, tests, GPU operations, or performance measurements.

## Recommendation

`EXPERIMENT_HINT_START=last_peer` is an honest production-style DAG signal worth a bounded controlled screen. It delays hint production until the orchestrator starts submitting the second serial peer request. It uses no future return input, final peer completion, actual remaining tool duration, private return skew, or chosen return winner.

Treat it as an **alternative activation policy**, not a proven equivalent removal of useless HTTP calls. It preserves the scoring algorithm and legal inputs but can change which captured query supplies a score, when the score lease starts, and consequently the ranking/selection. Numerical scoring, constrained selection, and native adoption must be verified again. No gain has been measured.

## Exact existing implementation

In [competing_workload.py](competing_workload.py:73), each hint producer optionally waits on a shared `asyncio.Event` before entering its existing publish/200 ms refresh loop. In `last_peer` mode the event is created per round before the warm requests. The producer tasks still exist after each warm completion, but remain parked until activation. ARC arms create no producer task and send no hint RPCs.

The driver then awaits pressure A, invokes pressure B with `started=activation`, and only after pressure B completes computes the common return release. Inside [request()](competing_workload.py:131), the event is set after the payload and request record are constructed and immediately **before** `client.post('/v1/completions', ...)` is entered. Exact provenance is therefore the orchestrator's second-peer submission attempt, not server acceptance, first forward, or successful tool execution. A failed submission can activate hints, but the driver then fails the run; it does not fabricate a successful peer-start acknowledgment.

This signal is available to a production orchestrator that owns the same sequential tool DAG. It is not automatically available to a generic cache connector with no orchestration integration. No run-specific GPU occupancy, hidden future selection, or measured remaining-time oracle is needed. Generalizing beyond this DAG would require an explicit public workflow phase supplied by that orchestrator, rather than deriving phase from benchmark request IDs.

The second peer cannot be submitted before the first peer has completed in this serial DAG. Accordingly, the public phase signal indirectly reveals first-peer completion. The protocol's generic `policy_excludes: ['peer completion', ...]` wording is too broad when describing `last_peer`; the accurate claim is that policy gets a public second-peer-submission phase and receives **no final-peer completion timestamp or future return deadline**. This is a reporting qualification, not a causality failure. The existing dataset contract already names this optional public phase follow-up.

## Release controls remain causal

Warm inputs, peer inputs, return inputs/output counts, order of requests, and configured private delays are unchanged. Both return continuations still use `common_release = actual_second_peer_completion + fixed_gap`, then add their existing private offsets. Neither awaits a hint response, canonical binding, or scoring/prefetch completion. A return reads only an acknowledgment that already arrived; the producer is canceled after demand completion as before.

Thus **the relative release rule is identical**. Absolute wall-clock release times are not fixed across arms: reducing host contention can change actual peer completion, which necessarily shifts the subsequent closed-loop release. Report that normal workload dependency rather than claiming identical absolute timestamps. Setting the mode consistently on ARC controls retains the same public event bookkeeping; those controls still send no hint RPCs.

## Capacity motivation is credible but does not prove all early work useless

For the short 400-block case, the nominated completed prefixes are about 97 blocks each and the first peer about 194. For physical allocation, a conservative allowance for the final partial warm block is 98 per warm: `2*98 + 194 + 1 null = 391 < 400`. For long 1,568 blocks, a corresponding allowance is `2*386 + 770 + 1 = 1,543 < 1,568`. These capacity counts are compatible with both warm sources remaining GPU-resident throughout the first peer. Verify residency from existing audits before treating the arithmetic as a replacement-policy proof, especially with older cached blocks or other reservations.

Whole-prefix staging rejects a source while any of its source keys remain on GPU. Therefore, if the sources do remain resident, early hints cannot yet cause that native whole-prefix movement. Delaying them could avoid HTTP parsing, canonical refresh/deduplication, repeated candidate processing, and logging during the first peer and the earlier warm interval. The saved fraction of calls is not necessarily one half: source 1 starts producing before warm 2 completes, durations differ, and each refresh is scheduled after the preceding HTTP exchange plus a 200 ms wait.

Importantly, [connector._retry_prefix_mean_prefetches](../../semantic_offload/connector.py:2946) requests numerical scores **before** the prefill/capacity movement gate. Its explicit intent is to compute while an active peer exists even if staging cannot yet act. A valid accepted scalar batch can be reused until its original bounded lease expires. Earlier hints may therefore establish a real score against an earlier active query; later activation may instead first score against peer B, or briefly lack a fresh eligible query. No feature timestamp may be renewed to conceal that difference. Existing missing-score bounded fallback remains a real risk that a screen must expose.

## Smallest meaningful screen

Use the same frozen source, data, cache budgets, constant 250 ms declared wait estimate, refresh period, capture settings, and actual gap for `pending` versus `last_peer`. Include matched ARC controls with the same request timing rule and CPU budget. Balance run order; compare raw outputs/native-key correctness, end-to-end throughput, CPU time, return TTFT/ITL, actual hint count and phase timing, accepted prefix numerical observations, scored/constrained candidates, native same-ID adoption, demand bytes, and wasted/consumed prefetch bytes. Report any changed query capture identities/timestamps and changed selections as policy trade-offs.

Require the first emitted hint to be at or after the recorded pressure-B request-start event; verify the event was not scheduled from its completion or a return timer. Require nonzero accepted numerical prefix scores and meaningful constrained decisions before treating reduced overhead as success. An apparent gain accompanied by unscored fallback or missing native adoption is not an acceptable result. Existing performance runs remain untouched and must not be pooled with a new activation policy without labeling it.

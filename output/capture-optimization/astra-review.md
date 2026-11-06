# Capture optimization review

Recommendation: first replace the CPU-list-to-device index construction with basic row slices plus one concatenation, preserving the existing GQA reduction and callback. This is the smallest independently reviewable candidate for the measured maximum-three-request workload. Reusing verified runner GPU indices is a promising subsequent fast path; a mutable pinned staging buffer is unnecessary complexity for the first attempt. No speedup is established by this review.

## Scope and evidence

Read-only source review on Mac, 2026-10-30. No tracked source edits, tests executed, installs, GPU/PC work, delegates, shared settings changes, commits or pushes. Only this report was created; no extracted archives or persistent processes. Existing working-tree changes belong to other work.

Reviewed `semantic_offload/query_capture.py`, `profiling.py`, capture installation/head gates and callback paths in `worker.py`, `tests/test_query_capture.py`, exact-chain admission in `manager.py`, and saved diagnostic/admission reproduction metadata. The completed checkpoints authorize no automatic campaign continuation; this review follows the new explicit review request.

Evidence:
- [Diagnostic and timing limitations](../scoring-efficiency/results/diagnostic/REPORT.md), [frozen geometry/configuration](../scoring-efficiency/results/diagnostic/evidence/frozen.json), [reproduction requirements](../scoring-efficiency/results/diagnostic/REPRODUCE.md).
- [Admission results](../admission-trial/REPORT.md), [current checkpoint](../admission-trial/CURRENT.json).
- Additional source inspection: `/Users/pippo/github/vllm-semantic-cache/vllm/v1/worker/gpu/model_runner.py` (`prepare_inputs`, approximately lines 903–1076) and `input_batch.py` (`InputBatch`, `combine_sampled_and_draft_tokens`, approximately lines 305–411). These local native files were NOT verified identical to the measured PC runtime. The inspected reproduction archives retain extension code/configuration and runtime evidence, but no matching native runner/input-batch source was found. Confirm the installed native implementation before enabling a metadata fast path.

The diagnostic observed 1,139 captures: extract 13.615 s host, nested index construction 13.217 s host/13.221 s thread CPU, gather 44.2 ms, head reduction 98.7 ms, scoring 195 ms. The 72 index-upload CUDA samples totaled 4.262 ms. CPU time may include driver spinning/waiting for preceding work. Neither 13 s of GPU compute nor 13 s of removable serving latency follows. Moving synchronization downstream can shrink this boundary without improving throughput.

## Exact proposed capture change

In `install.capture_query`, retain the stride counter and the paired request-ID/nonempty-boundary filtering. Replace only:

```python
last_q = query[torch.tensor(last_indices, device=query.device)]
```

with the equivalent selection:

```python
last_q = torch.cat([query[i:i + 1] for i in last_indices], dim=0)
```

The existing empty-request return must precede this. Keep the current head divisibility check and `view(n_reqs, num_kv_heads, num_queries_per_kv, head_dim).mean(dim=2)` unchanged. Basic slices use host integer bounds, avoiding a CPU→CUDA index tensor; concatenation provides owned gathered storage. Keep even the one-row concatenation initially to avoid introducing an aliasing special case. A later one-row view optimization can be assessed separately.

This launches a gather-like copy and still allocates the selected query tensor, as the original advanced indexing did. It is not allocation-free. It creates O(batch requests) Python views; large batches may favor an already available GPU index. The frozen experiment has max_num_seqs=3, 32 query heads, 8 KV heads, head dimension 128, BF16, TP1. Do not generalize performance beyond it. Do not introduce an arbitrary sampling cap or drop requests to bound cost.

Use a private selection helper if that makes testing the exact selection contract easier. Do not fold in scoring schedule, IVF, capture cadence, dtype promotion, mean bypass for group-size one, or worker snapshot changes. Opportunity/capture_only still pay capture extraction before storing snapshots; reducing scoring work does not remove index creation. Worker head grouping uses local attention head count divided by the probe's KV head count; preserve that relationship and existing topology restrictions.

## GPU metadata alternative and rejection conditions

The inspected local runner sorts `req_ids` with `sort_batch_req_ids`, builds `num_scheduled_tokens` in that order, and creates `query_start_loc` from its cumulative sum. Scheduler dictionary order and token `positions` are not substitutes: positions are within sequences, not flattened query-row indices.

The local `combine_sampled_and_draft_tokens` kernel writes logits rows from `query_end - num_logits` through `query_end - 1`. Thus `result.logits_indices` is reusable only after proving one logit per real request in the same prepared order. Speculative decoding and modes with zero new sampled tokens invalidate a generic substitution. Equal total length alone does not prove one row per request.

A guarded candidate would add an optional tensor reference to `_BatchLayout`, taken from that exact `prepare_inputs` result. Validate metadata on CPU: actual request count, all positive scheduled counts, no drafts, per-request logits counts all one (e.g. verified CPU cumulative-logit metadata), one-dimensional int64 GPU indices, exact request-count shape. At capture require identical query device and sufficient query rows. Use the retained tensor only in the owning execution; otherwise use slices. Do not validate values using GPU `.item()`, `.tolist()`, `equal()` or CPU readback in the hot path.

Another option is `query_start_loc[1:n_reqs+1] - 1`, performed on device at the eligible capture. This avoids host upload but adds a device operation/allocation, references a runner-reused buffer, and requires a zero-count filtering strategy. Prefer slices for zero-count/fallback cases. Never include padded requests. Never overwrite runner metadata. Neither metadata route should be enabled from an attribute name alone without confirming the installed producer and stream contract.

## Correctness and lifecycle objections

- **Rows and identities:** select the last scheduled real row per positive-count request, not the final full-context row, a padded row, or the last row of the entire batch. Chunked prefill, mixed prefill/decode and eager decode all need the same mapping. `zip` currently silently truncates mismatched lists; a new helper should reject inconsistent lengths, negative counts or noncontiguous boundaries rather than silently reassign identities. Valid zero counts must remain supported.
- **Heads/dtype/layout:** preserve contiguous GQA grouping and query dtype/device. Do not average all heads or reinterpret query heads as KV heads. Test strided input as well as contiguous input; concatenation should make the same logical rows available to the existing view/reduce. Shape/rank errors should be explicit, not disguised by fallback.
- **Dispatch consumption:** the current layout is cleared in a `finally` *after* `capture_query` invokes the callback. Ordinary dispatch temporarily removes the current mode, but direct reentrant invocation or stacked modes can still expose a live layout. Consume the layout before extraction/callback, retaining it in a local variable; preserve exception cleanup and other-layer behavior. Add a reentry test before changing this. Do not blanket-disable all dispatch modes to hide recursion.
- **Nested executions:** ContextVar resets correctly restore outer execution state. The handle counts active executions but does not serialize same-runner calls. Do not interpret its lifecycle lock as exclusive ownership of a shared GPU or host buffer. Nested mode stacks and shared stride state deserve explicit tests; runner-concurrent execution is not newly made safe by this optimization.
- **Streams/lifetime:** slices/cat/reduce should run on the current model stream and finish producing an owned grouped tensor before downstream reuse. No side stream is needed. Retained runner metadata is safe only if its production and consumption are ordered and its storage is not overwritten by another execution. Existing asynchronous scoring creates copies and readiness events; preserve those. `close()` waits for host executions, not arbitrary GPU DMA completion, so a new reusable buffer would require separate retirement fencing.
- **Reusable-buffer objection:** `non_blocking=True` does not make overwriting a pinned CPU buffer safe before DMA finishes. GPU overwrite must also follow every gather using that buffer. A correct design needs bounded slots, completion events, device/capacity keys, no reuse on incomplete events, retirement on growth/device changes, and fallback on exhaustion/reentry. Allocating/growing inside graph capture is another hazard. Do not use this design until simpler candidates fail.
- **Graphs:** keep FULL mixed-graph rejection. Full decode replay has no Python capture callback; preserve that opportunistic behavior. Dummy runs should not supply a real layout. Do not move selection outside the eligible probe hook into every prepare call: that pays for stride-skipped and replay-only steps and changes the execution contract. Never claim replay creates fresh captured queries. A guard against active CUDA capture, if required, must be applied consistently to the entire callback path and tested; graph-mode preflight alone does not certify all warmup/capture paths.

## Meaningful CPU tests and minimal GPU validation

Extend existing query-capture tests; the current suite mostly verifies ownership, stride, shutdown and exception behavior, not successful multi-request row/GQA equivalence.

1. Compare against the old advanced-index reference with sentinel values encoding token/head/channel. Use prepared IDs deliberately different from scheduler order; counts `[2, 0, 1, 3]`, padding with poison values, and expected rows `[1, 2, 5]`. Verify IDs, dtype, shape and grouped values. Include one request, all-zero counts, decode-only counts, changed batch sizes and a noncontiguous input.
2. Exercise group sizes 1 and 4, distinct KV-head values, FP32 and supported CPU BF16; verify invalid head divisibility. For ordinary finite inputs require exact equality when possible; retain numerical checks appropriate to backend reduction on GPU.
3. Precreate the input, then forbid `torch.tensor`/device index conversion during selection to catch accidental reintroduction of upload. Mock CUDA initialization as forbidden for CPU tests. A numerical comparison alone would miss the performance mechanism.
4. Use fake attention plus the existing patched-runner fixtures: other layer leaves layout intact, matching layer fires once, same layer repeated does not capture twice, padding/undersized query behavior stays defined, callback reentry cannot reuse the layout, callback failure clears it, nested execution restores outer state, close/reinstall does not retain indices or callbacks. Verify stride is per eligible dispatch, including empty and rejected-layout cases against the current contract.
5. If metadata fast path is attempted, add fallback tests for drafts, zero counts, missing metadata, incorrect dtype/device, padded lengths and unequal per-request logits counts. Verify retained indices are replaced each execution. CPU fixtures cannot establish CUDA stream safety.

For the execution owner, not run here: first do a small warmed CUDA correctness check on the installed runtime with identical queries/layouts for baseline and candidate, including BF16 GQA, padding, mixed/chunked prefill and eager decode. Check stream-ordered consecutive steps, graph warmup/replay behavior, and retained snapshots after later batches. Then use a fresh matched baseline/candidate serving comparison on the existing 45-request trace, unchanged pools/admission/scoring/stride. Record exact token lists, ordered keys, capture counts/IDs, load-only latency/throughput and allocations. A single pair is a screen, not significance evidence; only expand if justified and authorized.

Use identical observer settings. Preserve separate index/selection/reduction host scopes, model/serving totals, and sampled stream events; profiler intervals overlap. Drain completed events at the existing end boundary, reporting unfinished/dropped samples. Avoid per-step fences. `profiling.py` event recording and deferred draining are current-stream measurements, not kernel-exclusive GPU time; events and host instrumentation can perturb timings. Check whether time migrates to attention, scoring readback or transfer rather than claiming savings from one shorter scope. Keep a small correctness sample separate from timed load.

## Bounded admission: design challenge only

Thresholds 0 and 1 disable the same native frequency gate. Threshold2 offered 1,692 unique candidates but admitted only 11; first-miss lookup credit and the advancing store cursor explain why this is not an eleven-candidate workload. Threshold0 reached 199 resident blocks but wrote roughly 11 GiB, churned thousands of blocks and slowed serving. Prefetch0 exhausted free GPU blocks. These results support budgets and useful-prefix selection, not unrestricted persistence or demonstrated VRAM savings.

Derive a byte budget from current allocatable tier slots (exclude ready protected blocks, outstanding reservations and incomplete writes), host-memory/pinned-memory headroom, and remaining session fair share after charging existing and in-flight residency. For a preallocated CPU tier, do not double-count its already allocated backing against host free memory; host pressure instead constrains safe expansion. Use global and per-session ledgers, charge deduplicated physical storage once with an explicit sharing rule, and revalidate atomically before reserving.

A candidate write must fit all those limits AND remaining write tokens. Refill write tokens at measured sustainable D2H bandwidth discounted for demand-copy contention, with a bounded burst. Also require estimated queued service + copy time + completion margin to fit a causally available copy deadline. A wait hint is uncertain, not guaranteed future reuse: shorten by hint age/TTL, allow cancellation, and skip if no defensible slack. Account for stores already in flight and reserve demand-copy capacity. No fixed unexplained percentage establishes safety.

Choose complete consecutive native-key blocks from the earliest missing position in a verified chain, after any leading coverage that is actually protected through use. Otherwise start at zero; observed GPU coverage can disappear, as the trial showed. Require exact namespace, position, predecessor and generation validity; use only past reuse, current wait hints and presently available keys. Do not rank an isolated high-score suffix above the missing leading blocks needed to make it usable. Bound the new segment by byte/write/deadline budgets and include restore feasibility: CPU residency without a timely H2D path is not immediately useful reuse.

Staging needs its own live GPU-free-block reserve after demand/running requests, existing staged reservations and in-flight allocations. A configured fraction of the full pool is insufficient when actual free blocks are zero. Reject/cancel speculation before consuming demand headroom; retain fenced ownership until safe release. Recheck prefix coverage and destination mapping at consume time. Do not infer copy benefit from intended-resume timestamps alone.

D2H preservation does not release GPU references. GPU reclamation must use native ownership/refcount rules only after the CPU copy fence completes, and only when the source block is unreferenced; never forcibly free running/shared blocks. Any future claim must measure additional native free capacity or concurrency at fixed latency, separate from CPU bytes written and allocator pool size.

The existing `_chain_admission_identity` includes full-chain digest, session and generation; `_prepare_exact_chain_admission` requires a distinct request and exactly matching bounded observed-key tuple. Growing contexts generally change the digest (and sometimes generation), preventing credit for their shared prefix. Do not fix this by simply removing the digest: discuss a separately validated prefix identity built from native ordered keys, prefix length and compatibility namespace, with causal distinct-access accounting and bounded history. This is a policy change requiring separate approval, not part of capture optimization. Include growing-context, diverging-prefix, stale-hint, duplicate-offer, partial-copy/failure, fairness and zero-free-staging fixtures before any policy trial.

# Private prefix gather candidate — CPU qualified, GPU untested

`sources/prefix-gather` is a file copy of frozen `sources/prefix-mean`. Canonical sources, existing frozen source, harness, and configuration were not changed. The only package change is `semantic_offload/worker.py`; the candidate adds a dedicated test module and an explicitly CUDA-forbidden qualification runner. [Candidate manifest](sources/prefix-gather/pins.json) records baseline identities and every candidate file. Inherited qualification files/results are baseline evidence, not candidate qualification.

## Exact scope

The changed gather is used only when all are true: transported `prefix_mean_prefetch_scoring`, `summary_fields=required`, mean scoring, mean head aggregation, and opportunity schedule. Otherwise the original per-block stack/float code runs unchanged. No new environment variable or public option was added. Missing transport context safely retains the baseline path.

The ordered selected block list is still built in the existing shape-group order and is bounds-checked before any gather. Only adjacent ascending IDs become one basic-slice run; duplicate/reversed/nonadjacent IDs begin another run. For one run, `.to(dtype=float32, copy=True, memory_format=contiguous_format)` produces an owned contiguous FP32 tensor directly. This always copies even FP32 input. For multiple runs, `cat` owns the existing-order blocks and `.float()` retains the baseline value conversion. Grouping, reshaping, FP32 mean reductions, last-duplicate-wins insertion, publication identities, pruning, summary events, rollback, and raw DMA are unchanged.

Expected mechanical saving is fewer tensor-dispatched views, plus removal of the FP16 intermediate stack for a single run. Multiple highly scattered runs save less and may gain little. No latency, throughput, CPU, memory, or scoring-quality benefit has been measured. The helper still executes a short Python loop over IDs; it does not remove required numerical scoring.

## CPU acceptance

`tests/test_prefix_gather.py` covers FP16/FP32 exact ordered FP32 values, contiguous shape/strides, contiguous/reversed/duplicate/multiple-run layouts, owned backing after source overwrite, equal actual grouped reduction inputs, duplicate-key/publication equivalence, invalid IDs before gathering, empty layouts, fallback modes, single-run absence of per-block select/stack/cat, real candidate store failure/exception rollback, stale-publication rejection, and rollback fence-before-retirement.

After the root released the CPU hold, the exact isolated runner passed **478 tests**, with CUDA forbidden and uninitialized, the candidate-import assertion satisfied, and recorded source pins unchanged. No package or test repair was needed. The runner then received an explicit `sys.dont_write_bytecode=True` cleanup improvement and was rerun; [the final CPU receipt](sources/prefix-gather/qualification/gather-cpu-validation.json) records its outcome. Invocation:

```
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python output/prefetch-vram-20261101/sources/prefix-gather/qualification/check_gather_cpu.py
```

The runner loads the qualified runtime overlay, asserts it imported the candidate worker (not canonical), forbids CUDA initialization, sets Torch threads to one, runs the new tests plus the baseline 12-module scoring/publication/prefetch/transfer suite, checks source pins before/after, and writes `qualification/gather-cpu-validation.json`. Passing CPU imports/tests now also provide syntax validation. Inherited `qualification/cpu-validation.json` still describes only the old baseline.

## Recorded block order limits the opportunity

[Read-only layout analysis](sources/prefix-gather/qualification/gather-layout-analysis.json) of each existing short-r0 seed excludes warmup by actual load-window timestamps, deduplicates transfer jobs, and checks duplicate source lists agree. Every seed has the same 66 source-evidenced stores and 1,770 source blocks: 8 wholly ascending contiguous jobs, 30 one-block jobs, 6 wholly descending jobs, and 22 mixed/scattered jobs. There are 606 ascending adjacency edges versus 1,039 descending edges. Thus the candidate represents those lists using 1,164 slices rather than 1,770 individual block views; the 38 single-run jobs cover only 354 blocks (20% of source blocks), where it can additionally avoid the FP16 intermediate. This is a structural opportunity, not a dispatch timing or throughput estimate. The recorded geometry is one source GPU block per key; another geometry's shape grouping can change the selected order.

Descending is indeed common, so expected improvement is bounded and may be small. Negative-step basic tensor slices are unsupported. Reversing descending slices via `flip` or using device index tensors adds allocation/kernel/ownership work and is not included without profiling evidence. The safe alternative for descending/scattered lists is the current ordered `cat` fallback; it retains correctness without claiming coalescing. The candidate has not been broadened.

## GPU acceptance required before a performance comparison

The root remains sole GPU owner. Require a small real CUDA check with actual packed K/V layout for FP16 and FP32: gather and full grouped-reduction inputs bitwise equal to baseline for contiguous and scattered/duplicate/mixed-count cases; output means bitwise equal; overwrite source only after the existing stream/fence contract, verify owned results survive; cover original publication/failure/fence tests. Retain exact native key/order/physical-identity proof in an end-to-end constrained-prefix run, and verify real prefix scalar observations/adoptions still occur. Use same query and prefix inputs for numerical comparison, not different captures.

Only after those pass, freeze updated source and qualification pins and run matched, balanced paired candidate/baseline/ARC controls at the same cache budgets and workload. Keep current prefix component-pair timing separate: it evaluates whether summary gathering is material before this candidate is timed. Any failed candidate is retained as negative evidence; no source-finalization or audit weakening is part of this experiment.

Initial preparation respected the text-only hold. After CPU release, only the qualified CPU tests and stdlib layout inspection ran; no GPU operations or background services were started. Test processes exited. Candidate-only disposable bytecode was removed, and the runner now suppresses its creation. No commits or pushes.

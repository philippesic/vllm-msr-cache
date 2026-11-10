# Why the RTX 5080 result differs from the 2080 Ti result

This is an analysis of existing completed measurements, not another campaign.
No GPU job, profiler, CPU-tool rerun, old-host connection, policy change or
reclassification was performed. The original campaigns/reports/exclusions remain
unchanged. [Side-by-side charts](side-by-side.html),
[mixed-traffic variation and tool timelines](mixed-variance.html) and
[all 42 comparisons with diagnostic evidence](comparison.json) are retained here.

The direct comparison is less of a wholesale reversal than the earlier brief
5080 explanation suggested. Continuum's large lead on document workloads already
existed on the 2080 Ti. The important changes are a much less precise mixed-traffic
result, small native document gains becoming losses, and weaker secondary-model
benefits. All systems execute these fixtures much faster on the 5080 host.

## What actually changed

Percentages are our candidate's geometric throughput difference from the named
baseline; brackets are the original paired whole-process 95% confidence interval.
They are not differences between GPU hardware in isolation.

| Main Qwen=8B workload | Against | 2080 Ti | 5080 |
| --- | --- | --- | --- |
| Parallel repository | Normal cache | +0.1% [-1.8%, +2.0%], n=8 | -2.2% [-4.5%, +0.2%], n=7/8 |
| Parallel repository | Continuum | +0.1% [-1.4%, +1.7%], n=8 | -5.2% [-7.5%, -2.9%], n=7/8 |
| Documents, full CPU cache | Normal cache | +2.0% [+1.1%, +2.9%], n=12 | -1.4% [-3.5%, +0.7%], n=12 |
| Documents, full CPU cache | Continuum | -30.5% [-31.0%, -29.9%], n=12 | -31.5% [-36.0%, -26.6%], n=12 |
| Documents, half CPU cache | Normal cache | +0.7% [+0.4%, +1.0%], n=12 | -2.7% [-3.8%, -1.5%], n=12 |
| Documents, half CPU cache | Continuum | -33.8% [-34.2%, -33.4%], n=12 | -31.7% [-34.4%, -28.9%], n=12 |
| Mixed repository + QA | Normal cache | +12.4% [+10.9%, +14.0%], n=8 | +7.3% [-5.2%, +21.4%], n=8 |
| Mixed repository + QA | Continuum | +2.3% [+0.7%, +4.0%], n=8 | +6.9% [-7.0%, +22.8%], n=8 |
| Single-tool-worker repository | Normal cache | -0.9% [-2.1%, +0.4%], n=12 | -2.4% [-7.7%, +3.2%], n=12 |
| Single-tool-worker repository | Continuum | -0.1% [-1.1%, +0.8%], n=12 | -7.5% [-11.3%, -3.5%], n=12 |

The document/Continuum gap therefore is not a new 5080-specific failure. The mixed
candidate/Continuum point estimate is actually larger on the 5080, but its interval
is much wider. The 2080 Ti's +12.4% native gain lies inside the 5080's interval; these
measurements do not establish that an effect of the old magnitude disappeared.
No formal randomized GPU-only treatment-effect test is claimed.

Mean elapsed serving seconds for the same 80-request mixed batch:

| System | 2080 Ti | 5080 | Old/new time ratio |
| --- | ---: | ---: | ---: |
| Our change | 70.406 | 18.662 | 3.77x |
| Normal cache | 79.162 | 20.028 | 3.95x |
| Continuum | 72.053 | 19.905 | 3.62x |
| Upstream vLLM | 74.593 | 18.636 | 4.00x |

On the two document fixtures, the normal cache is 5.00–5.16x faster on the 5080
host, our candidate 4.83–4.99x, and Continuum 4.68–5.05x. These are descriptive
ratios of mean serving times across different hosts, not a paired GPU-only speedup.
Natural output lengths, CPU and system behavior can differ.

## Strongest explanation for the lost mixed-traffic precision

Both hosts have all eight planned native mixed pairs. No missing or failed B06
group enters this comparison. The 2080 Ti's eight native ratios range from +9.6%
to +15.2%. The 5080 ranges from -10.9% to +36.7%, with six wins and two losses.
The standard deviation of log ratios rises from 0.01654 to 0.14799: 8.95x.
The locked eight repeats therefore produce a much wider interval on the 5080.
We kept the original repeat count and every run.

Saved tool traces identify a real CPU stress test:
`python -m pytest -q -p no:cacheprovider tests/test_more.py::TestConcurrentTee`.
The frozen source creates 100 Python threads; each consumes 100,000 elements from
a shared, lock-protected iterator. All 16 native invocations passed exactly one
test. Its process duration varies from 0.799 to 6.762 seconds. Pytest's own output
also records the variation in test execution, rather than merely server startup.
The source excerpt, sealed tool outputs and exact trace hashes are in comparison.json.

| 5080 saved pair | Normal-cache whole batch s | Our whole batch s | Normal-cache thread test s | Our thread test s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16.952 | 19.020 | 3.128 | 4.196 |
| 2 | 20.386 | 20.575 | 6.605 | 6.578 |
| 3 | 21.919 | 16.036 | 6.762 | 0.831 |
| 4 | 21.100 | 20.418 | 6.593 | 6.580 |
| 5 | 18.648 | 18.023 | 4.307 | 4.600 |
| 6 | 16.969 | 16.577 | 0.799 | 1.522 |
| 7 | 21.488 | 21.459 | 6.565 | 0.833 |
| 8 | 22.764 | 17.192 | 6.649 | 1.333 |

In pair 3, the whole workflow differs by 5.883 s and the thread test by 5.931 s.
In pair 8, those differences are 5.571 s and 5.316 s. The test's session finishes last
in 11 of 16 runs. In pair 3's normal-cache run, it executes 13.244–20.007 s after the
workflow starts, then its session ends 21.915 s into a 21.919 s whole workflow. It is
on the observed completion path, not just an irrelevant overlapping background job.

This is strong evidence that real-tool timing contributes to the apparent large
wins and broad uncertainty. It is not proof that the entire cache benefit is
tool noise. Pair 7 is an explicit counterexample: the test is 5.732 s quicker for
our candidate, but the workflow ties. The candidate there recomputes 25,771
returning prompt tokens versus 17,237, and its summed request decode time is
148.1 s versus 108.4 s. Other model/scheduling work also varies. Request-time sums
overlap across concurrent requests and cannot be subtracted from wall time.

Across eight pairs, thread-test duration difference and whole-workflow duration
difference have descriptive correlation 0.734. This is not a fitted causal model,
outlier filter or corrected throughput result. No time was subtracted from the
published scores. The record does not isolate the OS scheduling/lock-contention
reason the same 100-thread test takes different times.

The measured shorter 5080 workflows make a several-second CPU task a much larger
fraction of total completion time. It is plausible that the changed relative
speeds of model execution and CPU tools expose more scheduling variation. Old
tool timelines are not local, so we cannot prove exactly how much the same test
overlapped model work on the 2080 Ti.

## Why cache retention can lose even when restoration increases

This part is measured 5080 activity, with a bounded interpretation—not a CUDA
profiler's causal attribution.

On Qwen=8B's half-CPU-cache document workload, our candidate averages:

- 3,056 restored returning tokens versus 227 for the normal cache;
- 0.904 GB loaded from CPU versus 0.060 GB;
- 131,920 recomputed returning prompt tokens versus 130,345.5;
- 272 hint events versus zero;
- 43.643 s to complete versus 42.484 s.

There is no average reduction in observed returning-prompt computation despite
much more loading and hint activity. That offers no recomputation saving to pay
for the extra work. Output tokens are almost equal within this5080 comparison.
These counters support the interpretation that retention is not buying useful
enough work avoidance here; they do not identify how much wall time is GPU copy,
Python/HTTP overhead, memory stalls, decode or changed scheduling.

On Qwen 4B's full-cache documents, the candidate does reduce returning-prompt
computation:88,619 versus 110,390 tokens per run. But it loads6.746 GB versus 2.309 GB
and finishes in 48.050 s versus 43.685 s. The prefill request-time sum falls from 15.339
to 13.191 s while the decode request-time sum rises from 98.524 to 109.712 s. Preserving
more cached state is not sufficient to improve end-to-end completion time.

A faster model can change the cost tradeoff between recomputation and preserving/
copying prefixes. That is a plausible explanation, supported by these work counters
and timings, rather than a proven per-kernel cause. Proving the exact split would
require a separately authorized controlled profile; none was run here.

The frozen inputs also do not imply identical cache access histories. For example,
Qwen 4B mixed traffic's normal-cache eviction-affected returns change from 480/480
on the 2080 Ti to 158/480 on the 5080; the candidate changes from 475/480 to 146/480.
With asynchronous real tools and fresh generated continuations, arrival/return
order can change even when resource limits and offered tasks are fixed. This is
direct evidence of changed effective cache pressure, not an inference from GPU VRAM.

## Explanations checked and limits

- All 22 declared resource profiles, frozen schedule/order/repeats/precision match
  the parent transfer. The 5080 was not silently given a bigger KV cache. The base
  Qwen=8B configuration uses 1024 GPU blocks on both hosts; cell-specific CPU-cache
  reductions remain part of the unchanged workload definition.
- Native policy, workload runner and model-source inputs are unchanged. The six
  source-closure differences are the recorded coordinator/analysis/validity/
  transfer/health modules; they are listed in comparison.json.
- The approved LMCache rebuild affects Continuum's separate runtime. ARC and our
  candidate do not use that LMCache extension, so it cannot directly explain their
  native-vs-native shift.
- The three excluded 5080 activity groups affect repository B06 only. They cannot
  explain the complete mixed/document comparisons. They are retained as missing,
  without replacement or imputation.
- Live generated outputs are not identical. Mixed native output totals differ
  by about 1.2–1.3% across hosts and under0.2% between native arms on each host; this
  is far smaller than the mixed timing swings, but output effects are not removed.
  Continuum caps/content differ as already disclosed in both original reports.
- There is no evidence here for a newly broken5080 LMCache kernel after the rebuild,
  or for a sudden loss of the candidate to Continuum across every workload.
- We do not have the 2080 Ti's full request/tool/metric traces on this machine, only
  its42 published per-run comparison inputs, secondary-pressure aggregates,
  transfer and qualification records. No old machine or other project was accessed.
- This is a comparison of two whole hosts. It does not isolate GPU architecture,
  CPU, driver, host memory or filesystem effects. We did not profile copy stalls
  or change the workload to manufacture cleaner cache-only results.

The supported interpretation is: the document competitor gap mostly persists;
the mixed result lost precision because short 5080 workflows are materially
affected by variable real CPU work; and the cache's extra transfers/hints do not
always save enough model computation to improve completion time. Exact attribution
of every native regression remains unresolved.

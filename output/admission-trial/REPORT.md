# Admission trial: threshold 0 versus 2

Threshold 0 removed the admission bottleneck and filled DRAM, but slowed both ARC and prefetch. Heavy write traffic, cache churn and staging pressure accompanied the additional reuse. This trial does not support disabling admission as a product default.

## Matched real serving results

Four fresh GPU serving runs on the same cached dense Qwen3-4B, exact six-agent plus background 45-request trace, 1440 MiB GPU KV pool and 448 MiB DRAM tier. One run per arm, fixed order ARC2/prefetch2/prefetch0/ARC0. Prefetch uses existing opportunity-only scoring, stride8, flat search, unchanged capture; ARC retention in every arm. No production source/runtime change or whole-prefix admission helper. Native inspection confirms threshold0 and1 have the same frequency-gate behavior: no counter table; we tested0.

| Configuration | Peak ready DRAM, MiB | Median release→first token, s | Mean release→first token, s | Output tokens/s |
|---|---:|---:|---:|---:|
| ARC, threshold2 | 24.75 | 2.744 | 4.397 | 82.39 |
| Prefetch, threshold2 | 24.75 | 2.833 | 4.399 | 81.98 |
| ARC, threshold0 | 447.75 | 3.801 | 4.471 | 78.61 |
| Prefetch, threshold0 | 447.75 | 3.880 | 4.888 | 72.25 |

Removing the gate reduced throughput4.58% for ARC and11.86% for prefetch. At threshold0, prefetch was8.09% slower than matched ARC. These are descriptive one-run point estimates, not significance estimates. Closed-loop arrival and cache histories vary. Additional counter events under threshold0 also have real observer cost, included rather than subtracted; component-exclusive cause of slowdown is not isolated.

## Why only eleven blocks?

Both threshold2 arms offered6052 store-key events, covering1692 distinct keys. The frequency gate rejected6031 events; eleven new blocks were accepted and all completed. There was no store capacity failure in these arms. Actual prefix lookup stops at the first missing block, leaving later blocks without lookup credit. The native store cursor still advances when keys fail qualification. CPU fixtures exercised both interactions. Candidate enumeration was not limited to eleven blocks.

## What changed without the gate?

| Metric | ARC2 | Prefetch2 | ARC0 | Prefetch0 |
|---|---:|---:|---:|---:|
| Completed store events | 11 | 11 | 4781 | 4962 |
| Distinct accepted keys | 11 | 11 | 1658 | 1650 |
| Peak ready resident blocks | 11 | 11 | 199 | 199 |
| Cumulative CPU evictions | 0 | 0 | 4582 | 4763 |
| Store-batch prepare calls returning no allocation | 0 | 0 | 470 | 405 |
| D2H traffic, MiB | 24.75 | 24.75 | 10757.25 | 11164.50 |
| Demand H2D, MiB | 38.25 | 27.00 | 639.00 | 247.50 |
| Speculative H2D, MiB | 0 | 13.50 | 0 | 126.00 |
| Executed prompt tokens | 80499 | 80563 | 80163 | 85119 |

The threshold0 arms wrote roughly10.5–10.9GiB into a447.75MiB usable cache over the run: cumulative writes are not concurrent residency or saved VRAM. Repeated offers after refused batches amplify event counts. Store allocation refusal is not a request failure or GPU OOM; every request and submitted transfer succeeded. Protection versus evictability versus batch-size subreason is not isolated.

Prefetch0 consumed126MiB on time, ready897.5ms before intended release. It selected56blocks at exact prefix positions56–111, based on previously observed leading GPU coverage. That leading coverage did not persist until resume; actual GPU hit was0 and complementary demand restoration remained necessary. Real GPU splice ran, native direct adoption did not. No constrained competing-session ranking occurred, so semantic ranking and index value remain untested even with a larger resident bank.

## Memory and useful residency

D2H makes a CPU copy and does not itself free the source GPU block. All eleven threshold2 source blocks were still referenced at store completion. Later reassignment of stored physical IDs was observed in every arm, but ordinary completion/eviction may explain it: causal release due to a DRAM copy is not established.

Sampled mean GPU referenced blocks: ARC2=379.60, prefetch2=380.36, ARC0=362.71, prefetch0=373.53. Sampled means are zero-order-hold estimates between bounded observations and can reflect time spent stalled; they do not demonstrate capacity savings. Peak referenced blocks were579/578/590/639 respectively. Prefetch0 reserved up to56staging blocks and reached zero free GPU blocks, versus49minimum free for ARC0. The fixed GPU pool was unchanged.

Startup-excluded GPU allocator peaks were not successfully isolated: reset occurred during startup. Raw labels and correction are preserved. Measured allocations do not establish reduced VRAM. More concurrency at fixed latency was not tested.

## Verification, evidence and recommendation

180/180 exact requests,13056 outputtokens, native ordered block identities and transfer/fence checks passed.270nativeCPUtests passed without initializing CUDA. All four servers exited; descendant accounting is empty, project lock free, source/runtime baseline unchanged, native custody released. No other job was stopped/resumed. No more GPU runs are scheduled.

Keep the conservative threshold2 setup for now. The next admission design to discuss is bounded, wait-aware preservation of useful consecutive prefixes, with a write budget and staging pressure limit, rather than unrestricted persistence. This experiment implemented diagnostics only; no such policy was implemented or tested.

- [Detailed report](results/admission_trial/REPORT.md)
- [Per-request serving measurements](results/admission_trial/measurements.csv)
- [Admission funnel](results/admission_trial/funnel.csv), [per-request funnel](results/admission_trial/funnel-per-request.csv), [capacity samples](results/admission_trial/capacity-samples.csv)
- [Exactness and transfer validity](results/admission_trial/evidence/validity.json), [CPU evidence](results/admission_trial/evidence/cpu-validation.json)
- [Reproduction instructions](results/admission_trial/REPRODUCE.md), [local verification](VERIFICATION.json), [final handback](results/admission_trial/evidence/handback.json), [storage inventory](STORAGE.json)

Compact archives, task-local source/patches, frozen configuration, compressed raw/logs and prior evidence are preserved outside Git. Missing exclusive DMA/network timing, store-refusal subreasons, true transition-integrated occupancy, repeated-trial noise and a capacity scaling result remain explicit limitations.

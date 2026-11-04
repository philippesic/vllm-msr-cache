# September 2026 results snapshot

This directory keeps the compact evidence needed to understand the project's
pause point. Multi-gigabyte raw runs remain outside Git in the local `output/`
tree. Values below come from retained final reports and were independently
checked where noted.

## E04: threshold-2 admission

**Question.** Does semantic/cost-aware admission improve serving over matched
ARC/LRU, and does the native store threshold help?

**Coverage.** RTX 5080; 420/420 valid cells; 10,500/10,500 request records; five
seeds; four workloads per arm; exact output-token digests and block identities
matched across controls.

| Threshold-2 arm | TTFT ms | ITL ms | Throughput tok/s | Makespan ms | Store MB/cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic/native | 216.89 | 27.16 | 57.17 | 11,246 | 2.96 |
| ARC | 210.18 | 25.83 | 58.72 | 10,631 | 2.98 |
| LRU | 209.55 | 25.81 | 58.78 | 10,660 | 3.02 |

Within the semantic family, threshold 2 versus neutral changed TTFT −22.50%,
ITL −19.78%, throughput +14.93%, makespan −12.00%, and store bytes −99.790%.
Against matched ARC, semantic point estimates were TTFT +3.17%, ITL +4.76%,
throughput −2.70%, and makespan +5.12%. There were no loads, restores, or
prefetches in these cells. The supported positive is a policy-agnostic admission
guard, not a semantic-policy win.

Tracked report: [E04-FINAL-RESULT.md](e04/E04-FINAL-RESULT.md), copied byte-for-byte
from `output/e04-full-grid-repair-20260909-v4/E04-FINAL-RESULT.md`. Frozen extension
revision `01d3eaa5a9bd90095e6f710c3c1ca07f77103f72`; local raw archive SHA-256
`15ffc92056a02fa9abb981289add2c819446b3386ee9cc583db1a94dbbf1b21e`.

## Combined MSR retention and semantic prefetch

**Question.** At matched memory, runtime, arrivals, and generic machinery, does
MSR+semantic prefetch reduce authoritative executed prompt work or attributable
GPU model-runner time per completed request versus ARC/LRU while exact outputs
hold and median TTFT remains within a 3% regression allowance?

**Accepted coverage.** 40 valid A/L/R/M cells and 1,944/1,944 successful exact
requests: 24 profile cells, 12 three-seed reference cells, and four cells on one
genuinely disjoint input. Every DRAM arm used `store_threshold=2`. Output-token
and exact key-chain sequences matched in every accepted comparison.

### Three-seed reference supplement (A26)

Each 48-request trace produced the same work totals: MSR+semantic (`M`) executed
120,174 prompt tokens versus 119,918 for ARC, LRU, and MSR+FIFO (`+0.2135%`).

| M compared with | Executed work | Model-runner time | Median TTFT | Makespan | Throughput | H2D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC | +0.2135% | +3.5626% | +2.5249% | +4.2812% | −4.1051% | +12.4031% |
| LRU | +0.2135% | +3.6567% | +1.8058% | +4.2807% | −4.1029% | +12.4031% |
| MSR+FIFO | +0.2135% | −0.0245% | −0.4213% | −0.2003% | +0.2011% | +12.4031% |

Semantic selection chose the no-return source and wasted all 22,020,096 staged
bytes across the three traces. The traces vary seeds but share an input cohort;
they do not support a broad population claim.

### Disjoint supplement (A27)

On one independently identified 48-request trace, both MSR+semantic and MSR+FIFO
consumed all 7,340,032 speculative bytes. Executed prompt work tied ARC/LRU/FIFO
at 107,334 tokens. MSR+semantic nevertheless had +4.937% model-runner time,
+6.226% median TTFT, +6.800% makespan, and −6.367% throughput versus ARC. With
one disjoint trace, uncertainty intervals are unavailable.

**Decision.** The implementation proved exact activation and useful semantic
consumption, but no repeatable provider benefit. This is a scoped negative for
the tested revisions and inputs, not evidence that MSR can never help.

Tracked report: [FINAL-REPORT.md](a27/FINAL-REPORT.md), copied byte-for-byte from
`output/msr-combined-prefetch-gpu-20260910/analysis-owner/final-all-metrics-a27-v1/FINAL-REPORT.md`,
SHA-256 `8e961ef3fe05ba8261f8e44c15e919ee5cd38280e7b42d57c556a3bbd3309ba3`.
Independent validator: [audit.txt](a27/audit.txt), copied byte-for-byte from
`output/msr-combined-prefetch-gpu-20260910/final-validator-a27-v1/audit.txt`,
SHA-256 `a23676d585022e32f3b4586a64b8457e5204fa8fee46704de4b9655f1a175e6c`
(`PASS`).

## Snapshot provenance

No artifact was regenerated, redacted, or otherwise transformed. The source
files remain in `output/`; this directory contains copies only.

| Tracked artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [E04 report](e04/E04-FINAL-RESULT.md) | 8,009 | `ef2be4894e8e03aa1628a9b6c2026308c334e60b240bb5a4a1b4789f6bb2dcd6` |
| [A27 final report](a27/FINAL-REPORT.md) | 17,989 | `8e961ef3fe05ba8261f8e44c15e919ee5cd38280e7b42d57c556a3bbd3309ba3` |
| [A27 independent audit](a27/audit.txt) | 6,664 | `a23676d585022e32f3b4586a64b8457e5204fa8fee46704de4b9655f1a175e6c` |

## Limitations at the September snapshot

No current evidence supports a production TTL, a useful-capacity frontier,
physical displacement attribution, price-based inference savings, or complete
CPU/RSS/index-memory cost. GPU-to-DRAM and H2D bytes are traffic overhead, not a
provider cache-write price. The tested package remains experimental.

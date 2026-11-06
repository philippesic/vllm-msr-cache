# Selected support and qualification matrix

Every observed cell uses the local2080Ti/WSL2 host, pinned0.26.1 runtime, FP16 KV, TP1 and one full-attention KV group. AWQ quantizes model weights. Raw int8 CPU backing stores unquantized KV bytes. Separate model, memory and execution cells do not qualify their cross-product.

| Selected functional cell | Context / execution | GPU / CPU KV MiB | Classification |
| --- | --- | --- | --- |
| [7B AWQ, short](../../output/wait-qualification-20261105/runtime-probe-repaired-functional-decision.json) | 1024 / eager | 84 / 168 | Functional only: all six waves and operational/allocation checks pass |
| [1.5B, short](../../output/wait-qualification-20261105/runtime-probe-qwen15-functional-decision.json) | 1024 / eager | 84 / 168 | Functional only: all six waves and operational/allocation checks pass |
| [OPT125M MHA](../../output/wait-qualification-20261105/runtime-probe-opt125-complete-functional-decision.json) | 1024 / eager | 83.8125 / 167.625 | Functional only: all six waves and operational/allocation checks pass |
| [7B, bounded audit-off](../../output/wait-qualification-20261105/runtime-probe-minimal-functional-decision.json) | 1024 / eager | 84 / 168 | Functional only: all six waves and operational/allocation checks pass |
| [7B, abandoned advice](../../output/wait-qualification-20261105/runtime-probe-abandon-effective-functional-decision.json) | 1024 / eager | 84 / 168 | Functional only: all six waves and operational/allocation checks pass |
| [7B AWQ, long](../../output/wait-qualification-20261105/runtime-probe-long-functional-decision.json) | 8192 / eager | 448 / 2688 | Functional only: all six waves and operational/allocation checks pass |
| [1.5B, long](../../output/wait-qualification-20261105/runtime-probe-qwen15-long-functional-decision.json) | 8192 / eager | 448 / 2688 | Functional only: all six waves and operational/allocation checks pass |
| [7B AWQ, long graphs](../../output/wait-qualification-20261105/runtime-probe-graphs-functional-decision.json) | 8192 / graphs | 448 / 2688 | Functional only: all six waves and operational/allocation checks pass |

Functional acceptance requires actual CPU restoration (or intentional no-restore after abandonment), a fresh reference that recomputes its entire prompt with zero external hits, exact greedy IDs, strict protocol checks, native abort/drain/recovery, pinned shared backing and owned cleanup. Graphs additionally require actual timed FULL dispatch evidence. This is not direct KV byte comparison, all-position support, production reliability or application speed.

| Selected workflow cell | Completed timing outcome versus plain ARC | Classification |
| --- | --- | --- |
| [stage4-graphs](../../output/wait-qualification-20261105/stage4-graphs-summary.json) | +8.87% batch time saved (+8.08 to +9.65% conditional 95% interval) | passed: complete measurements do not establish useful-answer parity or broad model qualification |
| [stage4-qwen15-performance](../../output/wait-qualification-20261105/stage4-qwen15-performance-summary.json) | +3.06% batch time saved (-5.28 to +10.75% conditional 95% interval) | passed: complete measurements do not establish useful-answer parity or broad model qualification |
| [stage4-qwen15-cohost](../../output/wait-qualification-20261105/stage4-qwen15-cohost-summary.json) | +4.67% batch time saved (-1.18 to +10.17% conditional 95% interval) | passed: complete measurements do not establish useful-answer parity or broad model qualification |
| [stage4-minimal-performance](../../output/wait-qualification-20261105/stage4-minimal-performance-summary.json) | +4.28% batch time saved (-0.75 to +9.07% conditional 95% interval) | passed: complete measurements do not establish useful-answer parity or broad model qualification |

The final machine-readable [matrix](../../output/wait-qualification-20261105/support-matrix.json) preserves every original attempt and replacement. Initial protocol-adapter failure, incomplete OPT snapshot and two abandonment-activity declaration mistakes remain visible. Functional diagnostic spans include extra candidate RPCs and are not matched application timing.

The public and isolated standalone clients pass positive CPU-ready refresh, actual tool execution, pinning, drain and cleanup. Isolation is author-run on the same host, with a separately captured dependency closure and disclosed transitive differences; it does not reproduce benchmark timing or substitute for independent human reproduction.

| Runtime/environment boundary | Classification |
| --- | --- |
| Pinned0.26.1rc1.dev75+gdc1be7903 plus retained native patches | Supported for the specifically observed local functional cells; performance claims remain conditional on their measured workloads |
| Newer published retention runtime0.1.dev1+g7966f717d.precompiled | Unsupported: actual extension import fails and native publication/scheduler/allocation APIs differ |
| Other GPUs, native Linux, BF16, distributed execution, hybrid/sliding-window/multiple KV groups | Untested |

Finite churn/recovery exposure cannot establish an unbounded-session limit or production failure rate. See [regressions](REGRESSIONS.md), [quality](QUALITY.md) and the stage reports.

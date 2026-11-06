# Quick demand/prefetch guard diagnostic

The exact semantic pending/in-flight early-return guard postponed **zero observed requests** in the two direct runs: **0 guard returns across 975 observed method calls (including warmup), 144/144 strict workload proofs**. Prefetch and numerical scoring were active. This rules out this particular guard as the explanation for slowdowns in these two executions; it does not establish that every budget or future execution is unaffected.

| GPU blocks | Actual bind states | Numerical scores | Native adoptions | Semantic guard returns | Separate native transfer deferrals |
|---:|---|---:|---:|---:|---:|
| 1024 | 4 pending, 1 ready | 51 | 1 | 0 | 3 |
| 2048 | 5 ready | 22 | 5 | 0 | 0 |

All direct observation identities were accepted; there are no unfinished guard episodes. The 1,024 case had eight preemptions and five constrained nominations, so this was not a disabled/no-pressure control. Its four pending hints retired for `nomination_deadline_elapsed` before their first recorded real lookup, by 8.82–18.65 seconds. Binding to a pending hint therefore did not imply waiting at the guard. The ordering is measured; no additional queue-cause attribution is made. Existing native pending-transfer deferrals are separate ordinary offload behavior, not this semantic guard.

The retrospective pass covers all eight semantic frontier budgets. Pending/in-flight binding potentials occur below 2,048 blocks: totals 1,434 ms at 768, 1,473 ms at 1,024, 1,174 ms at 1,280 and 420 ms at 1,536. Their largest individual historical potential is 912 ms. These are bind-to-ready/retire bounds, **not measured guard waits**; first-admission joins were unavailable with the original external-ID map. Historical 1,280 remains proof-invalid. At 2,048 only ready bindings occurred; the larger points had none. See `OFFLINE.json` and `RESULTS.csv` rather than reading unmeasured direct cases as zeros.

The new 1,024 execution contained pending binds but no in-flight bind, whereas its historical trace had one in-flight bind. That scheduling exposure remains unconfirmed. No third run or policy ablation was run. There is no clean causal throughput-loss estimate here: instrumentation has overhead, per-request spans can overlap, and returns are nonblocking scheduler decisions rather than whole-GPU stalls. Observed TPS/latency are recorded as diagnostic context only.

The private patch adds two exact-branch observations after reclaim/expiry and an import; source AST and return/native-call controls verify unchanged behavior. Existing ready/retire event timestamps bound eligibility separately from next-native-path polling. **33 CPU controls passed**, including fake/no-hint/native-pending controls and a genuine old strict-failed raw rejection. Actual hook import passed without initializing CUDA. The two-run controller took 6.16 minutes including startup, metrics/export and cleanup; actual workload traffic totalled 3.43 minutes. Offline preparation and reporting are additional.

Evidence and reproduction: `RESULTS.json`, `LOOKUP_TIMING.json`, `OFFLINE.json`, `PLAN.json`, `observer-only.patch`, `REPRODUCE.md`, `FINAL_VALIDATION.json` and `STORAGE.json`. No optimization, source policy/math/ownership changes, model download, commit or push. Prior campaigns and canonical dirty work are preserved.

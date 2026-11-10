# Functional qualification, 2026-11-10

Source checkpoint `6eb7faf`. This record covers functional correctness only. It does
not establish a retention-policy win, throughput gain or answer-quality equivalence.

## What was tested

Raw attempts and logs are retained outside Git at
`/home/philip/.local/share/wait-hint-port-work/` on DESKTOP-T4JKGJB. Each functional
attempt contains its source snapshot, admission record, execution log, observed
inputs and terminal record. These are local files, not remote backups.

| Lane | Retained evidence | Scope |
| --- | --- | --- |
| Portable | `development-logs/wait-port-portable-final.log` | 166 CPU tests and rule proofs; nine groups pass, native and runtime groups skipped |
| Runtime CPU | `development-logs/wait-port-runtime-lane6.log`, `runtime-cpu-06/` | Strict types on 12 adapter/runtime-test files; 12 pinned-runtime CPU tests; CUDA uninitialized |
| Hardware | `hardware-04/`, `hardware-04-wrapper.log`, `hardware-04-sanity.json` | Three hardware tests on RTX 2080 Ti/WSL2 under an external reservation; dense FP16, TP1, TRITON_ATTN, eager, FCFS |
| Setup | `development-logs/wait-port-install.log`, `development-logs/wait-port-lock.log`, `wheels/vllm.whl` | Hash-required install from the locked closure, dependency consistency, archived wheel |

CPU fixtures construct real native ARC, manager, scheduler and RPC objects and fake
only transfer handlers or utility transport. They cannot prove CUDA streams, device
fences or bytes. Independent oracles include an externally calculated digest vector,
hand-worked ARC ordering and victim expectations, known sampled-token and failed-load
cases, and HTTP/schema producer-consumer conformance.

Hardware observations:

- Two CPU slots were filled from GPU blocks 3 and 1. Advice retained the root;
  another store evicted the unhinted key. Restored GPU destinations 6 and 5 exactly
  matched source blocks 3 and 2, including native slot/key order and readiness.
- A completed 96-token transcript registered six full blocks. After resetting only
  GPU prefix state, restoration reused 96 tokens. Resetting both caches produced zero
  cached tokens; restored and fresh output token IDs matched in both advice modes.
  This is a tiny functional case, not general quality equivalence.
- A real API server returned 401 for unauthenticated advice, 400 for a malformed
  schema and 200 with six refreshed keys for valid advice through async RPC. Cleanup
  completed. Optional DeepGEMM import warnings and an upstream resource-tracker
  shutdown warning remain in the logs.

## Failed attempts

All attempt directories remain. `runtime-cpu-03` failed because an added test was
missing from the test inventory; the inventory was updated explicitly. `hardware-03`
failed because an over-restrictive guard rejected the model's native default runner;
the guard was removed and `hardware-04` re-ran all three cases. Earlier portable and
strict-type failures remain in `development-logs/`.

## Limits

- The pinned ARC T2-empty eviction defect is retained in both modes and pinned by a
  regression test.
- Transfer failures were injected only into CPU fake handlers. Real hardware fault
  recovery and power-loss durability are unqualified.
- Other layouts, distributed execution, other models and hosts, and the RTX 5080 are
  unqualified.
- Clients must share the engine host's monotonic clock. API keys define one trusted
  operator, not tenant isolation.
- The verifier owns only its own child session. It is not a scheduler, storage
  service or GPU reservation mechanism.

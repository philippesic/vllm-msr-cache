# Core qualification, 2026-11-10

This record covers the repaired core and engine-method delivery path. Performance,
general answer quality, other models and the RTX 5080 remain unqualified.

## Retained execution

Host: DESKTOP-T4JKGJB, WSL2, RTX 2080 Ti. Raw records are local, outside Git under
`/home/philip/.local/share/vllm-semantic-cache-ext-next/`.

| Execution | Record | Result |
| --- | --- | --- |
| Pinned runtime CPU | `attempts/runtime-cpu-20261111T024418Z/` | 13 tests passed, CUDA uninitialized; strict types passed on 13 adapter/runtime files |
| Hardware | `attempts/hardware-20261111T024424Z/` | Three tests passed in 93 seconds; CUDA initialized |
| Portable | `overhaul-20261110/commit-check.log` | Full acceptance: 27 CPU tests and 106 rule proofs |

Both runtime attempts were executed afresh, not reused. Their retained receipts
were independently read with `tools.runtime_verify --verify`; both passed. Product
file hashes in both snapshots matched the final product source. UTC attempt names
fall on November 11; the local execution date is November 10.

Hardware used Qwen2.5-0.5B-Instruct, dense FP16 TP1, HND, eager TRITON_ATTN. The
model revision and file hashes are in the admission records. Tests exercised exact
GPU/CPU byte transfers across CPU eviction, restored versus fresh generation in
both advice modes, and authenticated HTTP hint delivery through async engine RPC.
Both generation modes restored 96 cached tokens, versus zero after a full reset,
and produced matching restored/fresh output IDs. No active compute process remained
after completion. These are functional cases, not a performance experiment.

## Core behavior exercised

- Same-incarnation registrations require strictly increasing generations; another
  incarnation replaces the previous session.
- Exact replays retain decision, refreshed count and ready-prefix count after
  expiry and changed cache readiness, with no second refresh. Both advice modes
  retain replay results. Conflicting event content refuses.
- Generation replacement, session retirement and reset invalidate old observations.
- Invalid direct construction refuses, including Boolean numeric values, invalid
  clocks, namespaces, settings and oversized or mutable key collections.
- A shared metadata budget refuses observations/events before mutating state or
  refreshing ARC. Replacement/reset reclaim charges. This bounds retained metadata
  admission, not process RSS or temporary request allocations.

## Repairs and limits

Initial portable checks found test typing/import issues and isolated proof-fixture
coupling; these were repaired without adding exceptions or weakening rules. A mypy
internal assertion occurred while runtime and portable checks shared its cache.
A fresh cache passed; the failed cache and two failed full-check logs are retained
under `overhaul-20261110/`. Normal-cache type checking passed after replacement.

PyTorch deprecation warnings and upstream shutdown warnings remain in raw logs.
Hardware fault recovery, distributed execution, other layouts/models and general
quality equivalence are not established. The pinned upstream ARC eviction defect
remains in both modes.

Actions CI was removed by request. Local checks and the pre-commit hook remain.
There is no remote, so this checkpoint and evidence are local only.

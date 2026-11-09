# Stage 6 replication handoff

2026-11-08 update: Philip separately authorized [repository recovery on 2080 Ti](repository-recovery-03/README.md).
The current addendum coordinator explicitly selects the cell/host pinned serving
interpreter; historical exported sources retain the old inheritance behavior.
Use the repaired coordinator for future new studies, with a new source identity;
do not patch sealed bundles in place. This does not authorize 5080 execution.

No 5080 jobs have been started. This handoff preserves the 2080 Ti measurements
and prepares separately gated replication; it is not a 5080 performance or
production qualification. Read [measured outcomes and limitations](RESULTS.md)
before interpreting speed. The repository addendum section is closed and has
no repeated four-system pressure estimate. Do not reset its exhausted/closed
study state or rerun a failed gate under the original authority.

The primary transfer is the `relocated-bundle` directory under
`output/stage6-2080ti-20261106/portability-final-01/`. Its exact acquisition
source and permitted derived analysis closure are separate. The addendum transfer
is `output/stage6-2080ti-20261106/addendum-portability-final-01/relocated-bundle/`:
it includes its own source, seven frozen input cells, schedule/limits, all attempts,
objective/source audits and comparisons. `original-protocol.lock.json` is an
unchanged historical receipt, including original host paths; imported bindings
are separate and never relabel the original acquisition. Preserve private-host
files as private. Model weights and native runtime environments are reacquired
at pinned identities, not embedded in either bundle.

Copy both sealed bundle directories, the completion audit packet, and this
repository checkpoint to the new host using the configured Tailscale SSH alias
(LAN fallback only if needed). Keep SSH host-key/private-key authentication.
The primary proof also retains `relocated-evidence/` for offline analysis; copy
it when the full original raw measurements are needed. The final review packet is
`output/stage6-2080ti-20261106/completion-audit-packet-02/`; its seal and final
proof links are in [completion.json](completion.json). Verify seals after copy
with the exported verifier, before using inputs. Directory names below are
explicit new-host variables, not paths to this WSL host.

The primary bundle was sealed before proving its own relocation. Its embedded
pre-transfer report can therefore retain `complete: false`. Preserve that sealed
snapshot; the adjacent final completion and transfer receipts close the audit.
They do not waive the new host's hardware-preflight requirement.

Run primary replication from **its exported source**, not the addendum source
or mutable checkout. Keep the exact pinned base and selected policy.

```bash
BUNDLE=/absolute/transfer/primary-relocated-bundle
RUNTIMES=/absolute/new-cache/stage6-runtimes
CACHE=/absolute/new-cache/stage6-assets
HOST=/absolute/private/stage6-5080.json
OUT=/absolute/new-output/stage6-5080
cd "$BUNDLE/source"
python3 -m benchmarks.serving_suite.runtime_setup --out "$RUNTIMES"
PY="$RUNTIMES/local-env/bin/python"
"$PY" -m benchmarks.serving_suite.transfer_setup --bundle "$BUNDLE" --runtimes "$RUNTIMES" --cache "$CACHE" --out "$HOST" --id rtx5080
"$PY" -m benchmarks.serving_suite.campaign import --bundle "$BUNDLE" --host "$HOST" --out "$OUT"
make bench-tests PY="$PY"
make check-light PY="$PY"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase preflight
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase screen
"$PY" -m benchmarks.serving_suite.campaign qualify --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign activate-import --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase confirm
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase overhead
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase operational
```

These commands describe future gated work, not authorization to execute it in
this scheduled turn. Stop on capability, resource geometry or workload-purpose
failure and preserve the receipt. Hardware import creates a new linked identity;
`activate-import` preserves frozen order/repeats. Do not `freeze` or replan the
matrix, silently replace workloads/models, or reinterpret GPU-resident cells as
CPU-pressure evidence. The current preliminary PR can disclose the repository
comparison gap; closing it requires separately authorized future measurements.

The addendum's GPU-free import proof binds all 78 scheduled command configurations
to independently rebuilt runtime02, verifies input/source/model identities and
checks unloaded/enabled QA connector selection. It retains hardware preflight
required and execution unauthorized. This is command/input import validation,
not a new runnable campaign or native 5080 execution. Preserve that boundary:
ordinary primary `campaign import` is not an addendum resume command. The frozen
addendum driver is specific to its completed primary parent and persisted budgets.
A future hardware addendum needs a distinct declared parent and study identity;
no new addendum measurements are authorized by this handoff.

The three-model support scope is dense full-attention GQA, two families, TP1,
FP16/eager on the tested host. Hybrid/sliding/MHA/MQA, other engines, multi-GPU
and actual 5080 execution remain untested. Short-QA 3% overhead remains unresolved;
misleading hints measurably delay unhinted traffic. Source and capped-output
review is completed but does not establish answer-quality equivalence. No policy
change was made to remove unfavorable results. Stage 7's integration/release
writing and actual new-hardware qualification remain separate work.

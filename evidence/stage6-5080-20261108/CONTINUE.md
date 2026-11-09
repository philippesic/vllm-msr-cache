# Active Stage 6 RTX 5080 continuation

This is the same authorized job, not a new study request. Continue through
all primary phases and final audits; a scheduled check is not task completion.
The original user prompt authorizes a stop on a real capability, resource
geometry or workload-purpose failure, with a preserved/pushed receipt.

Work only in `/home/philip/vllm-semantic-cache-ext`. Branch:
`stage6-5080-20261108`, user-owned origin `philippesic/vllm-msr-cache`.
Read the original user prompt and the 2080 Ti RESULTS/HANDOFF before claims.
Do not work on Binary Eagle/shaderlod, modify copied seals, invent an
addendum, reset old studies, freeze/replan or substitute models/workloads.

Health schedule: `scheduled-task:fbd7ddce-7c9a-45b1-b122-09c729422576`,
every 15 minutes, bound to this thread. Keep enabled while active. Its first
due time is `2026-11-09T05:05:42.986Z`. During confirm the existing bounded
`benchmarks.research_checks.stage6_healthcheck` is suitable; before activation
inspect process/phase receipts directly because that checker requires a lock.
Do not keep an agent turn polling unchanged setup/serving logs.

Paths:

```bash
REPO=/home/philip/vllm-semantic-cache-ext
BASE=$REPO/output/stage6-5080-20261108
BUNDLE=$BASE/primary-execution-bundle-01
RUNTIMES=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01
PY=$RUNTIMES/local-env/bin/python
CACHE=/home/philip/.cache/semantic-offload/stage6-5080-assets
HOST=$BASE/host-replica.private.json
OUT=$BASE/primary-01
export PATH=/usr/lib/wsl/lib:$PATH
# Intended new Windows model-cache location; create only this task's directory.
export HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface
```

`runtime_setup` completed successfully in session 72310, with final
`published-identities.json` and all three GPU-free imports. Retain this cache.
Pinned runtime logs are under `$RUNTIMES`; summary is `$BASE/runtime-setup.log`.

Model reacquisition completed successfully in session 74455 from the original
primary export. All three pinned model contents verified; the initial private
host file is `$BASE/host.private.json`. Log: `$BASE/transfer-setup-03.log`.

Offline binding/import completed in session 71664. All host/cell interpreters
explicitly bind `$PY`; all four transfer/execution seals verify. See
`import.json` here and the raw `transfer-bind-replica-01.log` / `import-01.log`.

Native preflight completed in session 39853: all four cases passed, including
restore/fresh-reference correctness for every model and the Qwen3-8B registry
probe. All four attempt seals and 223 files verify, with clean GPU release.
See `preflight.json` here and `$BASE/preflight-summary.json` for full proofs.
Do not repeat preflight or call the confirmation-specific health checker
before protocol activation.

**Active operation:** safe campaign storage placement is running in session
53061, log `$BASE/storage-placement.log`, through the maintained
`benchmarks.research_checks.stage6_5080_storage` helper. It owns both GPU
reservations while copying/verifying. Do not start serving or duplicate it.
On success `$BASE/storage-placement.private.json` contains full file hashes
and preserved campaign identity; the original project campaign path becomes
an alias to the canonical Windows destination below. Set future `OUT` to:

`/mnt/c/Users/philip/AppData/Local/vllm-semantic-cache-ext-stage6-5080/campaigns/primary-01`

The helper verifies the old/copy/alias content, attempt seals and real Windows
writer-lock contention before removing its recoverable duplicate. Private host
files remain mode 600 in `$BASE/private-bindings/primary-01`, linked into the
Windows campaign. Inspect the helper result before proceeding to full screen.

**Storage placement before screen/confirmation:** the 2080 Ti primary campaign
retained 14,182,204,461 bytes, exceeding this WSL volume's remaining ~7.1 GiB.
The bounded four-case preflight passed. Its completed generated campaign is
being placed on the dedicated Windows
campaign volume before screen, keeping the original project path as an alias
so frozen absolute input strings remain valid. Destination parent:
`/mnt/c/Users/philip/AppData/Local/vllm-semantic-cache-ext-stage6-5080/campaigns`.
It has ~28.7 GiB free and a verified explicit ACL allowing only Philip, SYSTEM
and Administrators. The exact private ACL/storage receipt is
`$BASE/storage-plan.private.json` (placement is active).

Never move an active campaign. First verify coordinator/workers exited, GPU
clients absent and both reservations free. Copy to a new destination, preserve
all attempts/input strings/identity/order/repeats, and verify content and seals.
Keep root `host.private.json` and `hardware.private.json` in a permanent private
WSL directory (mode 600), with file links from the Windows campaign, rather
than exposing private host files on DrvFS. Switch the original project campaign
directory to an alias only after verifying the copy; verify campaign integrity
again through the alias. Preserve the previous tree until copy recoverability
is established. This is storage placement, not a new import, identity, matrix,
source revision or retry. Set future serving `OUT` to the canonical Windows
campaign path so bounded health process matching finds the coordinator.
Earlier setup attempts/logs are retained (one verifier rejection of generated
caches; one stopped cwd correction). All original seals verify again.

The separate `$BUNDLE` is a sealed copy derived from the original primary:
its coordinator is from checkpoint 1319f09, adding health-check source
attestation, with a new source identity. No other transfer/protocol fields
changed. Read `source-revision.json` here and the full raw receipt in `$BASE`.
It is not an addendum. `bench-tests` and `check-light` passed from this new
execution source too: 240 tests, 31 tests and compileall (`*-02.log`).

Recheck disk before continuing. Initial WSL free space was 23,943,786,496
bytes; the Windows filesystem had 60,804,325,376 bytes free. Sealed model
weights total 29,540,039,872 bytes. Existing uv cache permits hardlink reuse;
runtime reconstruction leaves about 7.3 GiB in WSL, and completed model
downloads leave about 29 GiB on Windows. Do not delete unrelated data or fill
a volume.

The commands below retain the entire authorized sequence. Offline rebinding
and import/preflight completed, and storage placement is active; do not repeat
those steps.
Use the pinned interpreter throughout:

```bash
cd "$BUNDLE/source"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m benchmarks.serving_suite.transfer_setup --bundle "$BUNDLE" --runtimes "$RUNTIMES" --cache "$CACHE" --out "$HOST" --id rtx5080 --offline
"$PY" -m benchmarks.serving_suite.campaign import --bundle "$BUNDLE" --host "$HOST" --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase preflight
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase screen
"$PY" -m benchmarks.serving_suite.campaign qualify --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign activate-import --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase confirm
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase overhead
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase operational
```

Run serving phases in persistent execution sessions with receipts and the
coordinator's built-in `gpu_reserved` protection. Verify each phase's receipts,
not only CLI exit (run_phase can retain failed cases yet return normally).
Stop on capability/resource/purpose failure. Do not run later dependent phases
after a failed gate. The primary exported coordinator explicitly launches with
cell profile `runtime_python`, falling back to host `runtime_python`; it does
not inherit the test interpreter. Imported cells must bind `$PY` explicitly.
The handoff's repaired-addendum warning must not cause a substitution of the
addendum source for primary. Preserve exact primary serving inputs. Document
the new imported acquisition identity and any necessary coordinator/source
adaptation separately, never patching the sealed transfer in place.

Avoid writing generated cache/package files into any sealed bundle. For
additional validation use external `PYTHONPYCACHEPREFIX` and pytest
`-o cache_dir=...`; `compileall` writes bytecode despite
`PYTHONDONTWRITEBYTECODE`. Verify seals after any tool runs there.

After acquisition, verify campaign integrity, produce source and capped-output
audits as required by the original prompt, retain adverse/unresolved outcomes,
write 5080 RESULTS and compact evidence, update only the affected STATUS entry,
commit and push the branch. Preserve private host files as private. Release GPU
reservations and owned serving processes; preserve useful caches/raw evidence.

Only when actually complete or failed for good: disable this saved schedule,
POST JSON to the following webhook with this thread and outcome (plus headline
results or failure reason), retain the response receipt, and stop this job:

`https://api2.cursor.sh/automations/webhook/0e2cee94-863f-5a53-a774-5d673448a10f`

Thread link:
<t3-thread://v2/92296584-2a2c-4998-8f7d-4ecb7d7f5ff2/mcp:launch:4ba24d5cb6f0895afb9f1420c78fb52741f61c910c692734f5dd8fa4aba7b875>

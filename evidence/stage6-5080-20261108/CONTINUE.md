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
BUNDLE=$BASE/transfers/primary-relocated-bundle
RUNTIMES=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01
PY=$RUNTIMES/local-env/bin/python
CACHE=/home/philip/.cache/semantic-offload/stage6-5080-assets
HOST=$BASE/host.private.json
OUT=$BASE/primary-01
export PATH=/usr/lib/wsl/lib:$PATH
# Intended new Windows model-cache location; create only this task's directory.
export HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface
```

`runtime_setup` is already running in persistent exec session 72310 from
`$BUNDLE/source`, logging to `$BASE/runtime-setup.log`. Its process birth
identity is in `$BASE/preparation-progress.private.json`. Inspect the existing
session/process and step logs under `$RUNTIMES`; do not start a duplicate or
delete this useful cache. Final `published-identities.json` plus clean exit
is needed, not just the existence of `local-env`. Setup logs preserve failures.
`bench-tests` and `check-light` have already passed from the exact exported
primary source with `$PY`: 240 tests, 31 tests, compileall. Logs are in `$BASE`.

Recheck disk before continuing. Initial WSL free space was 23,943,786,496
bytes; the Windows filesystem had 60,804,325,376 bytes free. Sealed model
weights total 29,540,039,872 bytes. Existing uv cache permits hardlink reuse;
models are not present locally. Do not delete unrelated data or fill a volume.

After successful setup, use the exact exported source and pinned interpreter:

```bash
cd "$BUNDLE/source"
"$PY" -m benchmarks.serving_suite.transfer_setup --bundle "$BUNDLE" --runtimes "$RUNTIMES" --cache "$CACHE" --out "$HOST" --id rtx5080
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

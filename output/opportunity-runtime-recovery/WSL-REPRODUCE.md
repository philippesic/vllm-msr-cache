This is a new screen using a reconstructed environment. It does not reproduce the historical runtime Git tree or an old campaign result.

Run from this task directory. Obtain a fresh whole-session Windows-authority `benchmark_quiet` grant with the installed devicecheckout procedure first. Maintain its finite keeper, use broker-mediated launches, and update the task-local broker wrapper to the new owner's credential file. Do not reuse an expired reservation. Hold the nonblocking project GPU flock throughout each arm and preserve historical pauses.

The retained `.venv`, `source`, `runtime`, `incoming` and evidence manifests describe the tested environment. `evidence/runtime-stage.json` pins the source archive and wheel; `evidence/recovered-runtime-files.json` pins all recovered payloads; `evidence/runtime-overlay.json` pins additions. Recovery preserved original Python files; absent package resources and compiled libraries came from the named prerelease wheel. Missing requirements files were not reconstructed as an alleged exact Git tree.

Environment construction used `uv venv --python 3.12`, the pinned prerelease vLLM wheel, Torch2.13.0+cu130, and an editable no-dependency install of `source`. Full installed versions are in `evidence/environment-freeze.txt`. `stage_recovered_runtime.py` and `complete_runtime_overlay.py` record the original construction steps; they mutate a new staging directory and should not be blindly rerun over this retained checkout. The former's original Windows transfer path may have been cleaned after preservation in WSL.

Runtime imports require source BEFORE runtime in PYTHONPATH, because both contain a benchmarks package. The server driver sets this itself. The executable is `.venv/bin/python`, and the compiled extension is `vllm._C_stable_libtorch`.

Validated screen commands (replace `TASK` with this directory; all workloads are broker-mediated):

```bash
python3 broker_run.py --key validate-new -- "$TASK/.venv/bin/python" -B "$TASK/validate_runtime.py"
python3 broker_run.py --key freeze-new -- "$TASK/.venv/bin/python" -B "$TASK/opportunity_screen.py" freeze
python3 broker_run.py --key readiness-new -- "$TASK/.venv/bin/python" -B "$TASK/opportunity_screen.py" smoke
python3 broker_run.py --key pairs-new -- "$TASK/.venv/bin/python" -B "$TASK/opportunity_screen.py" run
python3 broker_run.py --key analyze-new -- "$TASK/.venv/bin/python" -B "$TASK/analyze_screen.py"
```

Use a new results directory for a reproduction; existing results must be preserved. Freeze writes `evidence/frozen-screen.json` and pins driver, connector, token inputs, geometry and schedule before GPU execution. Readiness timings are excluded from paired results. The first readiness attempt did not admit DRAM; its evidence remains. The next attempt lacked the development reset endpoint and ran no requests. The successful correction uses256 fixed source requests with GPU-only cache reset before each, preserving external DRAM. Native threshold2 and first-miss lookup behavior progressively admit all128 blocks. Six fixed distinct pressure requests then remove the GPU prefix.

Both arms receive the same privileged exact-target pending hint at wait origin. Demand releases on its own timer without waiting for RPC acknowledgment. The sole config switch `wait_hint_diagnostic_no_dispatch` suppresses speculative movement for the matched demand control. Its patch and test are `evidence/extension-diagnostic.patch`. Staging, binding, destination copies and fences in the prefetch arm use the existing implementation.

The screen uses a cached Qwen2.5-1.5B-Instruct snapshot, BF16/TP1/block16,640 total GPU blocks,448MiB DRAM, native ARC threshold2,128-block staging budget, batch-invariant eager generation, three fixed paired seeds with alternating arm order, plus a zero-lead control. Geometry, exact token inputs and release schedule are authoritative in the frozen JSON. Background512 tokens releases100ms before resume and resume generates128 tokens. No semantic scoring or per-step synchronized CUDA timing is enabled. Source DRAM refcount protection during actual movement differs inherently; soft retention is off in both arms, and no competing DRAM admissions are intended during the gap.

Seal and later execute the exact restoration argv through `broker_run.py --restore --prepare` / `--restore`, with `.venv/bin/python -B restore_screen.py`. Verify all activity descendant trees empty, release the reservation with the restoration receipt, stop only the captured keeper, and cancel the device-offer subscription. Retain raw data and recovery evidence outside Git; use storage-steward for owned disposable artifacts. Do not resume historical campaigns.

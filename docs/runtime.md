# Runtime

Portable development uses CPython 3.12.3 and uv 0.13.6 with `uv.lock`, and installs
no vLLM, torch or CUDA. The separate runtime installs one exact vLLM nightly wheel,
torch 2.13.0 and transformers 5.22.1 from the hash-locked `runtime/requirements.txt`.
Install it under ignored `artifacts/venvs/`, and do not modify an environment while another
run uses it.

The admitted environments are `artifacts/venvs/runtime` (benchmark runtime with the
Gemma overrides and patch), `artifacts/venvs/wait-hint`, `artifacts/venvs/lmcache` and
`artifacts/venvs/harness`. Receipts before 2026-11-13 record their former paths outside
the repository. Do not modify an environment while a run uses it.

```sh
uv venv --python 3.12.3 artifacts/venvs/wait-hint
uv pip install --python artifacts/venvs/wait-hint/bin/python --require-hashes -r runtime/requirements.txt
uv pip install --python artifacts/venvs/wait-hint/bin/python --no-deps -e .
uv pip check --python artifacts/venvs/wait-hint/bin/python
uv run --locked python tools/check.py --runtime-python artifacts/venvs/wait-hint/bin/python
uv run --locked python tools/check.py --runtime-python artifacts/venvs/wait-hint/bin/python --test tests/runtime/test_arc.py
```

The first command is the runtime lane. It type-checks the adapter and runtime tests
against that environment, snapshots inputs, runs real pinned CPU integration with
CUDA hidden, and prints one verdict line with the record location. Records go to
`artifacts/runs/` unless `--out` names another new directory under `artifacts/`. It covers real ARC, manager and scheduler
construction, engine RPC and HTTP routing, submission/completion failures and
computed-token guards. CPU transfer handlers are fakes; real workers need CUDA
streams and host registration. The second command is fast partial feedback for one
runtime test file. After adding runtime tests, run `--update-tests` with
`--runtime-python`. The GPU inventory is separate and never included in normal
discovery.

## Version pin and engine integration

The supported Python and vLLM versions live in `wait_hint_offload/runtime_pin.py`.
A portable test fails if `runtime/requirements*.txt` or the Python pins disagree with
it. The connector refuses to construct on another vLLM version, another Python or
optimized Python.

The adapter subclasses private vLLM offloading classes, so it works only with the
pinned version. If a vLLM upgrade is wanted, it is a separate task Philip approves:
change the lock and the pin together, run the runtime lane, fix what breaks, and
requalify both advice modes on hardware.

When runtime setup, supported configuration or qualification commands change, update
this guide and affected setup examples in the same change. Keep the pin, lock files
and actual compatibility guards consistent. Update tests for changed assumptions and
report which runtime/hardware paths were exercised. New support must not inherit an
old qualification claim; update current limitations and relevant result explanations
through the result correction procedure without rewriting original execution records.

The pinned ARC policy has a characterized fallback limitation: it can refuse
eviction from an empty preferred queue even when the other queue has an evictable
block. This can reduce store opportunities in both advice modes.
`tests/runtime/test_arc.py` retains the case; the adapter does not patch the native
policy. Account for it when interpreting results, and treat any repair or repin as
a separate runtime change requiring qualification.

Hints reach the scheduler through vLLM's existing utility RPC, which looks up engine
methods by name. When the scheduler-side connector starts inside the engine process,
it adds one method, `EngineCore.publish_kv_connector_wait_hint`, which forwards to the
connector. The endpoint calls it with `engine_core.call_utility_async`. No vLLM files
are patched. See the [upstream attribution](../runtime/NOTICE.md).

Functional attempts have no automatic retries; each repair and retest creates another
retained record. The verifier enforces a finite timeout and cleans up only the process
session it created.

## Serving configuration

The supported target is dense full-attention FP16, TP1, HND, eager execution with
TRITON_ATTN. The benchmark implementation additionally accepts FP16 full/sliding
cache groups and aligned recurrent checkpoints, with at least one full-attention group.
Both modern benchmark models have initial restored-continuation probes; these do not
qualify broad hybrid correctness or performance. Gemma4 starts on the 2080 Ti only
with the hash-pinned `gemma-turing` compatibility patch. Its first native-ARC sustained
check found a continuation-token mismatch against a fresh run and stopped early; that
failed cell is retained. The GPU byte case later showed CPU-restored Gemma KV is
byte-identical to the stored blocks and the restored return equals a GPU-resident
return at the same boundary, so the mismatch reflects different cached boundaries,
not a faulty restore. Sustained operating checks now use that same-boundary
reference; a new Gemma operating check has not been run. See
[benchmark qualification](benchmarks.md).
BF16 on Turing, speculative decoding, LoRA and parallel execution remain
unsupported. Use an ordinary vLLM serving
command with `--dtype float16`, `--enforce-eager`, `--attention-backend TRITON_ATTN`,
prefix caching, an explicit API key, and this `--kv-transfer-config`:

```json
{
  "kv_connector": "WaitHintConnector",
  "kv_connector_module_path": "wait_hint_offload.vllm_adapter.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "namespace": {
      "model_revision": "immutable-model-identity",
      "tokenizer_revision": "immutable-tokenizer-identity",
      "adapter_namespace": "none",
      "layout_namespace": "fp16-dense-tp1-hnd-block16"
    },
    "cpu_bytes_to_use": 4194304,
    "advice_enabled": true
  }
}
```

Install the project in the runtime environment and set `VLLM_PLUGINS=wait_hint` to
expose `/v1/wait-hints`. On WSL hosts also set `VLLM_WSL2_ENABLE_PIN_MEMORY=1`; the
hardware lane sets it for its own runs. Namespace labels do not attest model
bytes; attempt receipts hash the actual model and tokenizer files. The hint protocol
is in [architecture](architecture.md). The advice-disabled control changes only
`advice_enabled`.

## GPU gate

GPU runs need Philip's explicit authorization. There is no GPU reservation system.
Before starting, check with `nvidia-smi` that no other process is using the GPU and
that there is disk headroom for the output.

```sh
artifacts/venvs/wait-hint/bin/python -m tools.runtime_verify --gpu --model /local/model --timeout 600
```

This tests FP16 store, CPU eviction and restore byte equality, restored versus fresh
generation in both modes, and authenticated HTTP-to-engine advice.
`python -m tools.runtime_verify --verify artifacts/runs/attempt` validates a retained
receipt, frozen source, model dependencies and artifact hashes without rerunning it.
Without arguments the verifier exits 2 and asks for an environment and output.
Unknown schemas, failed attempts, missing observations and changed inputs refuse.

Current GPU evidence covers only the RTX 2080 Ti/WSL2 cases in [FINDINGS](../FINDINGS.md).
The RTX 5080/sm_120 is a separate, unqualified gate. Adding owned C++/CUDA source
requires build and diagnostic recipes plus sm_75/sm_120 acceptance in
`tools/native.json`.

# Supported runtime

Portable development uses CPython 3.12.3 and uv 0.13.6 with `uv.lock`. It installs
no vLLM/torch/CUDA dependency. The separate runtime uses the exact historical
vLLM wheel `0.26.1rc1.dev75+gdc1be7903`, torch 2.13.0 and transformers 5.22.1.
`runtime/requirements.txt` includes the hash-locked split closure. Install it in
an external environment; do not mutate an environment while another attempt uses it.

```sh
uv venv --python 3.12.3 /external/venvs/wait-hint
uv pip install --python /external/venvs/wait-hint/bin/python --require-hashes -r runtime/requirements.txt
uv pip install --python /external/venvs/wait-hint/bin/python --no-deps -e .
uv pip check --python /external/venvs/wait-hint/bin/python
uv run --locked python tools/check.py --runtime-python /external/venvs/wait-hint/bin/python --out /external/new-cpu-attempt
```

The runtime lane explicitly type-checks the adapter and runtime tests against
that environment, then snapshots inputs and executes real pinned CPU integration
with CUDA hidden. It includes real ARC/manager/scheduler construction, RPC/HTTP
routing, submission/completion failures and computed-token guards. CPU transfer
handlers are fakes; real workers need CUDA streams and host registration.
After adding runtime cases, explicitly update both inventories with
`uv run --locked python tools/check.py --update-tests --runtime-python /external/venvs/wait-hint/bin/python`.
Collection also hides CUDA. The GPU inventory is separate and is never included
in normal directory discovery.

## Attested launcher

`tools.run_runtime` builds an owned temporary package overlay; installed packages
stay intact. It verifies wheel/interpreter and private-seam hashes, inserts only
the RPC methods in `runtime/rpc.json`, and records their resulting identities.
The connector requires the attested overlay and rechecks its private/modified
files at construction. Unpatched, drifting, optimized and unsupported runtimes
refuse with a repair instruction. See `runtime/compatibility.json` and the
[upstream attribution](../runtime/NOTICE.md).

Run any serving command through the same runtime Python:

```sh
/external/venvs/wait-hint/bin/python -m tools.run_runtime --gpu --timeout 600 /external/venvs/wait-hint/bin/python -m vllm.entrypoints.openai.api_server --help
```

The `--gpu` flag requests device visibility; it grants no authority or ownership.
Without it, the launcher hides CUDA. The launcher has a finite timeout and
boot/birth-checked owned-session cleanup. Functional attempts have zero automatic
retries; each repair/retest creates another retained attempt. A general durable
research coordinator, retry service and notification system are deferred.

## Serving configuration

The supported initial target is dense full-attention FP16, TP1, HND, eager
execution with TRITON_ATTN. BF16 on Turing, sliding/hybrid attention, speculative
decoding, LoRA and parallel execution refuse or remain unqualified as specified
by construction. The pinned model's native default runner is exercised; this does
not qualify every model/runner combination. Use an ordinary vLLM serving command with `--dtype float16`,
`--enforce-eager`, `--attention-backend TRITON_ATTN`, prefix caching, an explicit
API key, and this `--kv-transfer-config` object:

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

Install the project's entry point in the runtime environment and explicitly set
`VLLM_PLUGINS=wait_hint` to expose `/v1/wait-hints`. The launcher sets
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` for this WSL host. Namespace labels do not attest
model bytes; attempt receipts separately hash the actual model/tokenizer files.
See [the registration/hint protocol](source-port.md).

The advice-disabled control changes only `advice_enabled`. Both modes use the
computed-token bound, full-transcript registration and narrow failure recovery.
The pin's known ARC T2-empty fallback defect is intentionally retained in both
modes and characterized. It is not silently fixed or claimed neutral for performance.

## Hardware gate and retained evidence

After explicit authorization and external ownership, invoke the runtime Python
with `-m tools.runtime_verify --gpu --model /local/model --timeout 600 --out /external/new-hardware-attempt`.
This tests FP16 store/CPU eviction/restore byte equality, restored-versus-fresh
generation in both modes, and real authenticated HTTP-to-engine advice activation.
Keep the external reservation for the whole invocation. This repo supplies no
reservation mechanism; on this host the existing historical wrapper is used from
outside the repo, without passing its file descriptors to children. Check ambient
GPU work and disk headroom again after acquiring ownership. Advisory exclusion
covers cooperating users, not arbitrary external jobs.

`python -m tools.runtime_verify --verify /external/attempt` validates its receipt,
frozen source, model dependencies and artifact identities without rerunning it.
Without arguments, the verifier exits 2 and requests the explicit environment/output.
Unknown schemas, failed attempts, missing observations and changed inputs refuse.
Original evidence remains intact; there is no migration or automatic evidence expiry.

Current evidence qualifies only the named RTX 2080 Ti/WSL2 and cached Qwen2.5-0.5B
functional cases in [FINDINGS](../FINDINGS.md). RTX 5080/sm_120 is a separate hardware
gate. There are no throughput, quality, production, power-loss durability or general
agent-productivity claims. CPU warnings from optional upstream Torch/DeepGEMM paths
are retained in logs, rather than treated as selected mechanism failures.

## Historical closures

These remain provenance, not alternative supported installations:

| Historical closure | Native base | transformers |
| --- | --- | --- |
| Stage 6 native/candidate | Same nightly plus broader historical patches | 5.5.3 |
| Root development | Same nightly plus broader historical patches | 5.22.1 |
| Continuum | vLLM 0.10.2 / LMCache 0.3.6 / torch 2.8.0+cu128 | 4.55.2 |
| Retention PR #59714 experiment | vLLM 0.30.1rc1.dev530 plus mapped-host-pointer patch | 5.22.1 |

Do not collapse these into one tested environment or relabel historical results.
[Decision 003](decisions/003-runtime.md) preserves provenance. No owned C++/CUDA
source was added; the portable native-source group remains skipped. Adding owned
native code still requires the qualified build/diagnostic recipes and separate
sm_75/sm_120 acceptance in `tools/native.json`.

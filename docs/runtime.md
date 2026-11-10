# Runtime

Portable development uses CPython 3.12.3 and uv 0.13.6 with `uv.lock`, and installs
no vLLM, torch or CUDA. The separate runtime pins vLLM wheel
`0.26.1rc1.dev75+gdc1be7903`, torch 2.13.0 and transformers 5.22.1 through the
hash-locked `runtime/requirements.txt`. Install it outside the repository, and do
not modify an environment while another run uses it.

On DESKTOP-T4JKGJB the installed runtime is
`/home/philip/.local/share/venvs/wait-hint-offload`. Retained attempts live under
`/home/philip/.local/share/wait-hint-port-work/`.

```sh
uv venv --python 3.12.3 /external/venvs/wait-hint
uv pip install --python /external/venvs/wait-hint/bin/python --require-hashes -r runtime/requirements.txt
uv pip install --python /external/venvs/wait-hint/bin/python --no-deps -e .
uv pip check --python /external/venvs/wait-hint/bin/python
uv run --locked python tools/check.py --runtime-python /external/venvs/wait-hint/bin/python --out /external/new-cpu-attempt
```

The runtime lane type-checks the adapter and runtime tests against that environment,
then snapshots inputs and runs real pinned CPU integration with CUDA hidden. It
covers real ARC, manager and scheduler construction, RPC/HTTP routing,
submission/completion failures and computed-token guards. CPU transfer handlers are
fakes; real workers need CUDA streams and host registration. After adding runtime
tests, update both inventories with
`uv run --locked python tools/check.py --update-tests --runtime-python /external/venvs/wait-hint/bin/python`.
The GPU inventory is separate and never included in normal discovery.

## Launcher

`tools.run_runtime` builds a temporary package overlay and leaves installed packages
intact. It verifies the wheel, interpreter and private-seam hashes in
`runtime/compatibility.json`, inserts only the RPC methods in `runtime/rpc.json`,
and records their resulting hashes. The connector requires the overlay and rechecks
its files at construction. Unpatched, drifting, optimized and unsupported runtimes
refuse with a repair instruction. See the [upstream attribution](../runtime/NOTICE.md).

Run any serving command through the runtime Python:

```sh
/external/venvs/wait-hint/bin/python -m tools.run_runtime --gpu --timeout 600 /external/venvs/wait-hint/bin/python -m vllm.entrypoints.openai.api_server --help
```

`--gpu` makes the device visible; it grants no authority. Without it, the launcher
hides CUDA. The launcher enforces a finite timeout and cleans up only the process
session it created. Functional attempts have no automatic retries; each repair and
retest creates another retained attempt.

## Serving configuration

The supported target is dense full-attention FP16, TP1, HND, eager execution with
TRITON_ATTN. BF16 on Turing, sliding/hybrid attention, speculative decoding, LoRA
and parallel execution refuse or are unqualified. Use an ordinary vLLM serving
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

Install the project's entry point in the runtime environment and set
`VLLM_PLUGINS=wait_hint` to expose `/v1/wait-hints`. The launcher sets
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` for WSL hosts. Namespace labels do not attest model
bytes; attempt receipts hash the actual model and tokenizer files. The hint protocol
is in [architecture](architecture.md). The advice-disabled control changes only
`advice_enabled`.

## GPU gate

GPU runs need Philip's explicit authorization naming the host, device and window,
plus an external reservation held for the whole invocation. This repository has
no reservation mechanism; on DESKTOP-T4JKGJB use the existing wrapper from the
older `vllm-semantic-cache-ext` checkout, from outside this repo, without passing
its file descriptors to children. Check for other GPU work and disk headroom after
acquiring it.

```sh
/external/venvs/wait-hint/bin/python -m tools.runtime_verify --gpu --model /local/model --timeout 600 --out /external/new-hardware-attempt
```

This tests FP16 store, CPU eviction and restore byte equality, restored versus fresh
generation in both modes, and authenticated HTTP-to-engine advice.
`python -m tools.runtime_verify --verify /external/attempt` validates a retained
receipt, frozen source, model dependencies and artifact hashes without rerunning it.
Without arguments the verifier exits 2 and asks for an environment and output.
Unknown schemas, failed attempts, missing observations and changed inputs refuse.

Current GPU evidence covers only the RTX 2080 Ti/WSL2 cases in [FINDINGS](../FINDINGS.md).
The RTX 5080/sm_120 is a separate, unqualified gate. Adding owned C++/CUDA source
requires build and diagnostic recipes plus sm_75/sm_120 acceptance in
`tools/native.json`.

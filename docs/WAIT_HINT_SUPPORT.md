# Wait-hint integration and qualification scope

The implementation advises exact-prefix CPU ARC retention while an application
workflow waits for an actual tool or for its returning request's first token.
Hints refresh already-ready CPU keys within their ARC queues; they do not count
as demand hits, change ARC's learned balance, reserve GPU pages, or perform
speculative transfers in the CPU-retention-only profile.

Current qualification is tracked in [the staged evidence](../evidence/wait-qualification-20261105/PROTOCOL.md)
and [current status](STATUS.md). Stage 1 passed on the local pinned configuration.
Selected functional cells now pass for 7B AWQ,1.5B unquantized weights and OPT125M
on the pinned local host, including long 7B/1.5B and long 7B graph checks. This is
functional coverage, not their cross-product or a model-wide speed claim. The
selected Stage 4 workflow measurements are complete; the [matrix](../evidence/wait-qualification-20261105/SUPPORT_MATRIX.md) distinguishes resolved gains, neutral bounds and inconclusive cells. Loading a model or
acknowledging a hint is insufficient.

## Supported shape and runtime requirements

The completed-transcript path requires one full-attention KV group, tensor
parallel size one, a single native offload block per chunk, unquantized FP16/BF16
KV, and native prefix-publication support. The tested RTX 2080 Ti requires FP16
and `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. AWQ refers to model weights, not quantized KV.
Hybrid/sliding-window/chunked/noncausal attention, multiple KV groups, distributed
execution, other KV dtypes, and all vLLM-supported architectures are not implied.

Use the exact source/wheel setup in [vendor/VLLM.md](../vendor/VLLM.md).
The maintained development launcher requires
`0.26.1rc1.dev75+gdc1be7903` plus this repository's native patches. A broad package
dependency declaration is not a supported-version matrix. An actual newer-runtime import/constructor probe fails because required APIs changed; it remains unsupported. No other runtime or hardware is qualified by the local result.

## Minimal client and configuration

[wait_hint_client.py](../examples/wait_hint_client.py) is a two-turn example that
executes an actual fixed SHA-256 subprocess, publishes known-prefix advice, and
returns with the observed transcript. It never executes model-generated shell.
[wait_hint_config.json](../examples/wait_hint_config.json) is the selected minimal
CPU-only profile. Both the [public client run](../output/wait-qualification-20261105/example-client/result.json)
and [isolated source/environment run](../output/wait-qualification-20261105/isolated-example-client/result.json)
passed actual tool execution, positive ready-key refresh, pinning, drain and cleanup. Do not infer performance from this tiny example.

The server must use the pinned launcher, `--served-model-name suite-model`,
FP16, block size 16, the JSON transfer configuration, and
`VLLM_PLUGINS=semantic_wait_hint`. Keep the server on the same host as this
example: timestamps use the shared host monotonic clock. Cross-host clock
translation is not provided. With an existing correctly configured server:

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
$PY examples/wait_hint_client.py --url http://127.0.0.1:8000 \
  --model /path/to/local/model/snapshot \
  --config examples/wait_hint_config.json
```

The namespace identifies the actual model revision, tokenizer revision, adapter
and KV layout; update it when those inputs change. Tokenize with the same local
tokenizer and chat template as the server. The completed prefix is the previous
prompt plus output IDs excluding the last token, rounded down to complete pages.
The application identifies only observed materialized tokens. It publishes a
current wait estimate, never a recorded future completion time. Repeated updates
stop when the returning request yields its first token.

## Diagnostics and off switch

An acknowledgment with `reason=cpu_prefix_recency_refreshed` and a positive
`ready_cpu_prefix_keys` establishes queue activity. Zero ready keys is a valid
no-op. Neither proves useful work saved. Inspect actual CPU restores, recomputed
prompt tokens, batch completion and returning-request latency against a no-hint
control. `observed_prefix_unavailable`, `namespace_mismatch`,
`causal_clock_mismatch`, and `session_registry_capacity` identify distinct causes
of inactive advice; do not silently reinterpret them as successful retention.

To disable the addition, stop application hint production and set
`prefetch_budget_fraction=0`. Keep ordinary ARC CPU offloading enabled. Disabling
`wait_hint_cpu_retention_only` alone is not the off switch: another prefetch path
could become eligible if a nonzero budget remains. No-hint ARC is the primary
measured control in the qualification campaigns.

The unaudited profile bounds the observed-prefix and result registries and caps
new session registration at 4,096 identities by default. It preserves generation
tombstones to reject delayed old generations. At capacity, new advice is refused
while normal serving continues; a cache reset/restart clears the registry. The
audited research profile retains generation history for evidence. Unlimited
session churn and long-running production reliability are not qualified by a
finite local soak. Keep the profile and exposure of each reliability result
explicit.

# Passive demand/prefetch guard diagnostic

Reuse the existing Qwen2.5-7B-Instruct AWQ cache and public corpus. Nothing is installed into the canonical source or serving environment. `PLAN.json` pins the private source/hook/runtime and exact two command vectors (1,024 and 2,048 GPU blocks). CPU cache remains 4,096 blocks / 3.5 GiB; workload, scoring, kernel, waits, ownership and returns are unchanged. The only connector changes are the import and two passive calls in `observer-only.patch`.

From this directory, with the recorded host environment:

```sh
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B offline.py
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B -m pytest -q -p no:cacheprovider test_probe.py test_frontier.py
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B frontier.py --plan PLAN.json
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B analyze_probe.py
```

Do not overwrite existing run directories. A fresh reproduction must copy this overlay to a new directory, regenerate output paths and dependency pins, and preserve model/corpus/source identities. The strict controller stops on any failed completed-response proof. It does not accept `summary.valid` alone.

`OFFLINE.json` examines all eight existing semantic curve points. Bind-state intervals are potential bounds, not observed guard calls. New raw probe data are at `postload_export.scheduler.demand_guard_probe`. The exact early-return observation occurs after reclaim and expiry. Existing ready/retire event timestamps close eligibility bounds; the next normal native-path timestamp and polling gap are separate. The separate native pending-transfer observer remains active. The helper never waits, synchronizes GPU work, alters a lease or changes a return value. Spans are per-request postponement bounds; their sum is not GPU idle time or throughput loss. Both instrumented runs are diagnostic-only, with no causal policy ablation or clean throughput comparison.

The optional existing-timestamp join is reproducible with `python -B lookup_timing.py` after the two diagnostics; it never launches a model. It explicitly joins native and external request identities.

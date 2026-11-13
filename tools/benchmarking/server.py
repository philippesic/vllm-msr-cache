from __future__ import annotations

import json
import socket
from pathlib import Path


def server_command(
    python: Path,
    model: Path,
    system: str,
    port: int,
    backend: str = "TRITON_ATTN",
    cpu_bytes: int | None = None,
    mechanism: str = "recency",
    gpu_blocks: int | None = None,
    max_model_len: int = 8192,
) -> tuple[list[str], dict[str, str]]:
    gemma = (
        json.loads((model / "config.json").read_text()).get("model_type")
        == "gemma4_unified"
    )
    command = [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        "qualification",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--enforce-eager",
        "--attention-backend",
        backend,
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.89" if gemma else "0.85",
        "--max-num-batched-tokens",
        "512" if gemma else "2048",
        "--enable-prefix-caching",
        "--language-model-only",
        "--api-key",
        "local-benchmark",
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
    ]
    if gpu_blocks is not None:
        command.extend(["--num-gpu-blocks-override", str(gpu_blocks)])
    pool = cpu_bytes or (2048 if gemma else 256) * 1024 * 1024
    namespace = {
        "model_revision": model.name,
        "tokenizer_revision": model.name,
        "adapter_namespace": "none",
        "layout_namespace": "native-fp16-tp1",
    }
    if system == "lmcache":
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cache_port = listener.getsockname()[1]
        command.extend(
            [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "LMCacheMPConnector",
                        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "lmcache.mp.host": "tcp://127.0.0.1",
                            "lmcache.mp.port": cache_port,
                            "lmcache.mp.autostart": True,
                            "lmcache.mp.mp_transfer_mode": "lmcache_driven",
                            "lmcache.mp.autostart.server_args": (
                                f"--l1-size-gb {pool / 1024**3:g} --chunk-size "
                            )
                            + ("256" if gemma else "528")
                            + " --separate-object-groups --eviction-policy LRU --disable-observability",
                        },
                    }
                ),
            ]
        )
    elif system != "gpu-only":
        command.extend(
            [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "WaitHintConnector",
                        "kv_role": "kv_both",
                        "kv_load_failure_policy": "recompute",
                        "kv_connector_module_path": "wait_hint_offload.vllm_adapter.connector",
                        "kv_connector_extra_config": {
                            "namespace": namespace,
                            "cpu_bytes_to_use": pool,
                            "advice_enabled": system == "wait-hint",
                            "advice_mechanism": mechanism,
                        },
                    }
                ),
            ]
        )
    return command, namespace

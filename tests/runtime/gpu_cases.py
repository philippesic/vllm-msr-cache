from __future__ import annotations

import asyncio
import gc
import json
import time
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.inputs import TokensPrompt
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy

from tests.runtime.kv_probe import KVProbe
from tests.runtime.support import native_fixture
from tools.config import hardware_model
from wait_hint_offload.contracts import Hint, Namespace
from wait_hint_offload.vllm_adapter.arc import ARCPrefix
from wait_hint_offload.vllm_adapter.engine_rpc import PUBLISH_METHOD
from wait_hint_offload.vllm_adapter.transfers import TransferWorker


def test_real_fp16_transfer_order_and_byte_equality() -> None:
    spec, _, cache = native_fixture()
    values = torch.arange(8 * 32, dtype=torch.float16, device="cuda").reshape(8, 32)
    expected = values[[3, 2]].clone()
    manager = CPUOffloadingManager(num_blocks=2, cache_policy="arc")
    context = ReqContext("byte-case")
    first, second, third = (OffloadKey(value) for value in (b"a", b"b", b"c"))
    prepared = manager.prepare_store([first, second], context)
    assert prepared is not None
    assert isinstance(prepared.store_spec, CPULoadStoreSpec)
    assert isinstance(manager._policy, ARCCachePolicy)
    assert list(prepared.store_spec.block_ids) == [0, 1]
    byte_rows = values.view(torch.int8)
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(byte_rows, 64)], [[CanonicalKVCacheRef(0, 64)]]
    )
    worker = TransferWorker(spec, cache)
    worker.worker = spec.get_worker(caches)
    try:
        source = GPULoadStoreSpec([3, 1], [2], [0])
        slots = CPULoadStoreSpec([0, 1])
        store = OffloadingConnectorMetadata(
            {}, {1: TransferJob("store", source, slots)}, {1}
        )
        worker.handle_preemptions(store)
        worker.worker.wait({1})
        worker.get_finished(set())
        stored = worker.build_connector_worker_meta()
        assert (
            stored is not None
            and stored.completed_jobs == {1: 1}
            and not stored.failed_jobs
        )
        assert manager.lookup(first, context) is LookupResult.HIT_PENDING
        manager.complete_store([first, second], context)
        assert ARCPrefix(manager._policy).promote_prefix((bytes(first),)) == 1
        replacement = manager.prepare_store([third], context)
        assert replacement is not None and replacement.evicted_keys == [second]
        assert isinstance(replacement.store_spec, CPULoadStoreSpec)
        assert list(replacement.store_spec.block_ids) == [1]
        worker.handle_preemptions(
            OffloadingConnectorMetadata(
                {},
                {
                    3: TransferJob(
                        "evict", GPULoadStoreSpec([2], [1], [0]), replacement.store_spec
                    )
                },
                {3},
            )
        )
        worker.worker.wait({3})
        worker.get_finished(set())
        replaced = worker.build_connector_worker_meta()
        assert replaced is not None and replaced.completed_jobs == {3: 1}
        manager.complete_store([third], context)
        assert manager.lookup(second, context) is LookupResult.MISS
        load_spec = manager.prepare_load([first, third], context)
        assert isinstance(load_spec, CPULoadStoreSpec)
        assert list(load_spec.block_ids) == [0, 1]
        values[[3, 1, 2]] = -1
        destination = GPULoadStoreSpec([6, 5], [2], [0])
        worker.start_kv_transfers(
            OffloadingConnectorMetadata(
                {2: TransferJob("load", load_spec, destination)}, {}
            )
        )
        worker.worker.wait({2})
        assert worker.get_finished(set()) == (set(), {"load"})
        loaded = worker.build_connector_worker_meta()
        assert (
            loaded is not None
            and loaded.completed_jobs == {2: 1}
            and not loaded.failed_jobs
        )
        torch.cuda.synchronize()
        assert torch.equal(values[[6, 5]].view(torch.int8), expected.view(torch.int8))
        manager.complete_load([first, third], context)
        assert not worker.get_block_ids_with_load_errors()
        print(
            json.dumps(
                {
                    "mechanism": "native FP16 GPU-CPU-GPU",
                    "blocks": 2,
                    "initial_store_source_order": [3, 1],
                    "restored_source_order": [3, 2],
                    "cpu_slots": [0, 1],
                    "destination_order": [6, 5],
                    "byte_equality": True,
                    "cpu_eviction": "unhinted second key replaced; hinted root restored",
                }
            )
        )
    finally:
        worker.shutdown()
        del worker
        gc.collect()
        torch.cuda.empty_cache()


def test_serving_restore_matches_fresh_in_both_advice_modes() -> None:
    model = hardware_model()
    namespace = Namespace(str(model), str(model), "none", "fp16-dense-tp1-hnd-block16")
    tokens = list(range(1000, 1096))
    outputs = []
    for enabled in (False, True):
        llm = LLM(
            model=str(model),
            dtype="float16",
            enforce_eager=True,
            max_model_len=256,
            max_num_seqs=1,
            gpu_memory_utilization=0.4,
            block_size=16,
            num_gpu_blocks_override=16,
            enable_prefix_caching=True,
            attention_config={"backend": "TRITON_ATTN"},
            kv_transfer_config=KVTransferConfig(
                kv_connector="WaitHintConnector",
                kv_role="kv_both",
                kv_load_failure_policy="recompute",
                kv_connector_module_path="wait_hint_offload.vllm_adapter.connector",
                kv_connector_extra_config={
                    "namespace": namespace.mapping(),
                    "cpu_bytes_to_use": 4 * 1024 * 1024,
                    "advice_enabled": enabled,
                },
            ),
        )
        try:
            registration = {
                "schema": "wait-registration-v2",
                "scope": "hardware",
                "session": "conversation",
                "incarnation": "instance",
                "generation": 1,
            }
            params = SamplingParams(
                temperature=0,
                max_tokens=1,
                extra_args={"kv_transfer_params": {"wait_hint": registration}},
            )
            initial = llm.generate(
                [TokensPrompt(prompt_token_ids=tokens)], params, use_tqdm=False
            )[0]
            assert initial.kv_transfer_params is not None
            observed = initial.kv_transfer_params["wait_hint"]
            assert observed["registered"] and observed["boundary"] == 96

            async def readiness(engine: Any) -> dict[str, Any]:
                deadline = time.monotonic() + 10
                event = 0
                while time.monotonic() < deadline:
                    now = time.monotonic_ns()
                    hint = Hint(
                        "hardware",
                        "conversation",
                        "instance",
                        1,
                        str(event),
                        observed["prefix"],
                        96,
                        namespace,
                        observed["observed_ns"],
                        now,
                        now + 1_000_000_000,
                        observed["clock_domain"],
                    )
                    result: dict[str, Any] = engine.call_utility(
                        PUBLISH_METHOD, hint.mapping()
                    )
                    if result["ready_cpu_prefix_keys"] == 6:
                        return result
                    event += 1
                    await asyncio.sleep(0.01)
                raise TimeoutError("CPU store readiness not observed")

            acknowledgement = asyncio.run(readiness(llm.llm_engine.engine_core))
            assert acknowledgement["decision"] == (
                "refreshed" if enabled else "disabled"
            )
            assert acknowledgement["refreshed_keys"] == (6 if enabled else 0)
            assert llm.reset_prefix_cache(reset_connector=False)
            restored = llm.generate(
                [TokensPrompt(prompt_token_ids=tokens)],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )[0]
            assert (
                restored.num_cached_tokens is not None
                and restored.num_cached_tokens >= 80
            )
            assert llm.reset_prefix_cache(reset_connector=True)
            fresh = llm.generate(
                [TokensPrompt(prompt_token_ids=tokens)],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )[0]
            assert fresh.num_cached_tokens == 0
            assert list(restored.outputs[0].token_ids) == list(
                fresh.outputs[0].token_ids
            )
            outputs.append(list(fresh.outputs[0].token_ids))
            print(
                json.dumps(
                    {
                        "advice_enabled": enabled,
                        "scheduling_policy": "fcfs",
                        "acknowledgement": acknowledgement,
                        "restored_cached_tokens": restored.num_cached_tokens,
                        "fresh_cached_tokens": fresh.num_cached_tokens,
                        "restored_matches_fresh": True,
                        "output_ids": outputs[-1],
                    }
                )
            )
        finally:
            llm.llm_engine.engine_core.shutdown()
            del llm
            gc.collect()
            torch.cuda.empty_cache()
    assert outputs[0] == outputs[1]


def test_real_authenticated_http_registration_and_hint(tmp_path: Any) -> None:
    import signal
    import socket
    import subprocess
    import sys
    import uuid

    import httpx

    from tools.config import server_environment
    from tools.process import process_identity, signal_owned, stop_session

    model = hardware_model()
    namespace = Namespace(str(model), str(model), "none", "fp16-dense-tp1-hnd-block16")
    key = uuid.uuid4().hex
    name = "functional-" + uuid.uuid4().hex
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    configuration = {
        "kv_connector": "WaitHintConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_module_path": "wait_hint_offload.vllm_adapter.connector",
        "kv_connector_extra_config": {
            "namespace": namespace.mapping(),
            "cpu_bytes_to_use": 4 * 1024 * 1024,
            "advice_enabled": True,
            "max_sessions": 2,
            "max_events": 64,
        },
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        key,
        "--dtype",
        "float16",
        "--enforce-eager",
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.4",
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "16",
        "--enable-prefix-caching",
        "--attention-backend",
        "TRITON_ATTN",
        "--kv-transfer-config",
        json.dumps(configuration),
    ]
    previous_hint: dict[str, Any] | None = None
    for restart in range(2):
        log_path = tmp_path / f"server-{restart}.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(
                command,
                env=server_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            owner = process_identity(child.pid)
            assert owner is not None
            try:

                async def scenario() -> None:
                    nonlocal previous_hint
                    headers = {"Authorization": "Bearer " + key}
                    async with httpx.AsyncClient(
                        base_url=f"http://127.0.0.1:{port}", timeout=10
                    ) as client:
                        deadline = time.monotonic() + 120
                        while True:
                            if child.poll() is not None:
                                raise RuntimeError("server startup failed")
                            try:
                                models = await client.get("/v1/models", headers=headers)
                                if (
                                    models.status_code == 200
                                    and models.json()["data"][0]["id"] == name
                                ):
                                    break
                            except httpx.TransportError:
                                pass
                            if time.monotonic() >= deadline:
                                raise TimeoutError("server readiness")
                            await asyncio.sleep(0.1)
                        registration = {
                            "schema": "wait-registration-v2",
                            "scope": "http",
                            "session": "conversation",
                            "incarnation": "instance",
                            "generation": 1,
                        }
                        response = await client.post(
                            "/v1/completions",
                            headers=headers,
                            json={
                                "model": name,
                                "prompt": list(range(1000, 1096)),
                                "max_tokens": 1,
                                "temperature": 0,
                                "kv_transfer_params": {"wait_hint": registration},
                            },
                        )
                        assert response.status_code == 200, response.text
                        observed = response.json()["kv_transfer_params"]["wait_hint"]
                        assert observed["registered"] and observed["boundary"] == 96
                        now = time.monotonic_ns()
                        hint = Hint(
                            "http",
                            "conversation",
                            "instance",
                            1,
                            "event",
                            observed["prefix"],
                            96,
                            namespace,
                            observed["observed_ns"],
                            now,
                            now + 10_000_000_000,
                            observed["clock_domain"],
                        )
                        if previous_hint is not None:
                            stale = await client.post(
                                "/v1/wait-hints", headers=headers, json=previous_hint
                            )
                            assert (
                                stale.status_code == 200
                                and stale.json()["decision"] == "clock_or_deadline"
                            )
                            print(
                                json.dumps(
                                    {"actual_engine_restart_stale_hint_refused": True}
                                )
                            )
                        previous_hint = hint.mapping()
                        denied = await client.post(
                            "/v1/wait-hints", json=hint.mapping()
                        )
                        assert denied.status_code == 401
                        invalid = await client.post(
                            "/v1/wait-hints",
                            headers=headers,
                            json={**hint.mapping(), "schema": "unknown"},
                        )
                        assert invalid.status_code == 400
                        deadline = time.monotonic() + 5
                        event = 0
                        while True:
                            result = await client.post(
                                "/v1/wait-hints",
                                headers=headers,
                                json={**hint.mapping(), "event": str(event)},
                            )
                            assert result.status_code == 200, result.text
                            if result.json()["refreshed_keys"] == 6:
                                break
                            if time.monotonic() >= deadline:
                                raise TimeoutError("HTTP advice activation")
                            event += 1
                            await asyncio.sleep(0.01)
                        print(
                            json.dumps(
                                {
                                    "mechanism": "HTTP to async engine RPC to scheduler ARC",
                                    "authenticated": True,
                                    "unauthenticated_status": denied.status_code,
                                    "malformed_status": invalid.status_code,
                                    "acknowledgement": result.json(),
                                }
                            )
                        )
                        from tests.runtime.http_scenarios import lifecycle

                        print(
                            json.dumps(
                                await lifecycle(
                                    client, headers, name, namespace.mapping()
                                )
                            )
                        )

                asyncio.run(scenario())
            finally:
                if child.poll() is None:
                    signal_owned(child.pid, owner, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        stop_session(child.pid, owner)
                        child.wait()
                stop_session(child.pid, owner)
                print(log_path.read_text())


def test_large_head_attention_matches_float_reference() -> None:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    generator = torch.Generator(device="cuda").manual_seed(7103)
    for head_size in (256, 512):
        keys = torch.randn(
            (2, 64, 1, head_size),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        values = torch.randn(
            keys.shape, generator=generator, device="cuda", dtype=torch.float16
        )
        for count in (1, 64):
            queries = torch.randn(
                (count, 8, head_size),
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            for window in (-1, 32):
                output = torch.empty_like(queries)
                unified_attention(
                    queries,
                    keys,
                    values,
                    output,
                    torch.tensor([0, count], device="cuda", dtype=torch.int32),
                    count,
                    torch.tensor([128], device="cuda", dtype=torch.int32),
                    128,
                    head_size**-0.5,
                    True,
                    (window, window),
                    torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
                    0.0,
                    None,
                    None,
                    None,
                )
                scores = (
                    torch.einsum(
                        "qhd,khd->qhk",
                        queries.float(),
                        keys.reshape(128, 1, head_size).float(),
                    )
                    * head_size**-0.5
                )
                positions = torch.arange(128, device="cuda")
                query_positions = torch.arange(128 - count, 128, device="cuda")
                visible = positions[None, :] <= query_positions[:, None]
                if window > 0:
                    visible &= positions[None, :] >= query_positions[:, None] - window
                scores.masked_fill_(~visible[:, None, :], float("-inf"))
                expected = torch.einsum(
                    "qhk,khd->qhd",
                    scores.softmax(-1),
                    values.reshape(128, 1, head_size).float(),
                )
                torch.testing.assert_close(
                    output.float(), expected, atol=0.004, rtol=0.004
                )


def generate_once(llm: LLM, prompt: list[int], params: SamplingParams) -> Any:
    return llm.generate(
        [TokensPrompt(prompt_token_ids=prompt)], params, use_tqdm=False
    )[0]


def test_restored_kv_bytes_match_stored_and_resident_boundary_output() -> None:
    model = hardware_model()
    gemma = json.loads((model / "config.json").read_text()).get("model_type") == (
        "gemma4_unified"
    )
    namespace = Namespace(str(model), str(model), "none", "native-fp16-tp1")
    prompt = list(range(5000, 5000 + 64 * 24 + 1))
    options: dict[str, Any] = {"language_model_only": True} if gemma else {}
    llm = LLM(
        model=str(model),
        dtype="float16",
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=512 if gemma else 2048,
        gpu_memory_utilization=0.89 if gemma else 0.4,
        enable_prefix_caching=True,
        attention_config={"backend": "TRITON_ATTN"},
        kv_transfer_config=KVTransferConfig(
            kv_connector="WaitHintConnector",
            kv_role="kv_both",
            kv_load_failure_policy="recompute",
            kv_connector_module_path="wait_hint_offload.vllm_adapter.connector",
            kv_connector_extra_config={
                "namespace": namespace.mapping(),
                "cpu_bytes_to_use": 1024 * 1024 * 1024,
                "advice_enabled": False,
            },
        ),
        worker_extension_cls="tests.runtime.kv_probe.KVProbe",
        **options,
    )
    params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    try:
        fresh = generate_once(llm, prompt, params)
        resident = generate_once(llm, prompt, params)
        generate_once(llm, prompt, params)
        generate_once(llm, prompt, params)
        llm.collective_rpc(KVProbe.snapshot_kv.__name__)
        assert llm.reset_prefix_cache(reset_connector=False)
        llm.collective_rpc(KVProbe.clear_kv.__name__)
        restored = generate_once(llm, prompt, params)
        comparison: dict[str, Any] = llm.collective_rpc(KVProbe.compare_kv.__name__)[0]
        boundary = 64 * 24
        assert resident.num_cached_tokens == restored.num_cached_tokens == boundary
        assert comparison["written_blocks_identical_to_a_stored_block"] > 0
        same_boundary = list(restored.outputs[0].token_ids) == list(
            resident.outputs[0].token_ids
        )
        print(
            json.dumps(
                {
                    "model": model.name,
                    "boundary_tokens": boundary,
                    "fresh_cached_tokens": fresh.num_cached_tokens,
                    "resident_cached_tokens": resident.num_cached_tokens,
                    "restored_cached_tokens": restored.num_cached_tokens,
                    **comparison,
                    "restored_matches_resident_at_boundary": same_boundary,
                    "restored_matches_fresh": list(restored.outputs[0].token_ids)
                    == list(fresh.outputs[0].token_ids),
                    "fresh": list(fresh.outputs[0].token_ids),
                    "resident": list(resident.outputs[0].token_ids),
                    "restored": list(restored.outputs[0].token_ids),
                }
            )
        )
        assert same_boundary
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

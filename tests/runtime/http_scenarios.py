from __future__ import annotations

import time
from typing import Any

import httpx


async def lifecycle(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    model: str,
    namespace: dict[str, str],
) -> dict[str, Any]:
    acknowledgements = []
    for cycle in range(3):
        reset = await client.post(
            "/reset_prefix_cache?reset_external=true", headers=headers
        )
        assert reset.status_code == 200 and reset.json()["success"]
        registration = {
            "schema": "wait-registration-v2",
            "scope": "lifecycle",
            "session": "returning",
            "incarnation": str(cycle),
            "generation": 1,
        }

        async def generate(session: str) -> dict[str, Any]:
            response = await client.post(
                "/v1/completions",
                headers=headers,
                json={
                    "model": model,
                    "prompt": list(range(1000, 1096)),
                    "max_tokens": 1,
                    "temperature": 0,
                    "kv_transfer_params": {
                        "wait_hint": {**registration, "session": session}
                    },
                },
            )
            assert response.status_code == 200, response.text
            observed: dict[str, Any] = response.json()["kv_transfer_params"][
                "wait_hint"
            ]
            assert observed["registered"]
            return observed

        observed = await generate("returning")
        now = time.monotonic_ns()
        hint = {
            **{key: value for key, value in registration.items() if key != "schema"},
            "schema": "wait-hint-v2",
            "event": "0",
            "namespace": namespace,
            **{
                key: observed[key]
                for key in ("prefix", "boundary", "observed_ns", "clock_domain")
            },
            "published_ns": now,
            "deadline_ns": now + 30_000_000_000,
        }

        async def publish(payload: dict[str, Any]) -> dict[str, Any]:
            response = await client.post(
                "/v1/wait-hints", headers=headers, json=payload
            )
            assert response.status_code == 200, response.text
            acknowledgement: dict[str, Any] = response.json()
            state = acknowledgement["metadata"]
            assert state["sessions"] <= 2 and state["events"] <= 128
            assert 0 <= state["charged_bytes"] <= state["metadata_bytes_limit"]
            return acknowledgement

        first = await publish(hint)
        replay = await publish(hint)
        assert (
            not first["replayed"]
            and replay["replayed"]
            and replay["newly_refreshed_keys"] == 0
        )
        assert replay["metadata"] == first["metadata"]
        assert (await publish({**hint, "published_ns": now + 1}))[
            "decision"
        ] == "event_conflict"
        assert (
            await publish(
                {
                    **hint,
                    "event": "expired",
                    "published_ns": observed["observed_ns"],
                    "deadline_ns": observed["observed_ns"] + 1,
                }
            )
        )["decision"] == "clock_or_deadline"
        assert (
            await publish(
                {**hint, "event": "other-clock", "clock_domain": "stopped-engine"}
            )
        )["decision"] == "clock_or_deadline"
        assert (await publish({**hint, "event": "wrong-prefix", "prefix": "wrong"}))[
            "decision"
        ] == "identity_mismatch"
        for event in range(1, 64):
            accepted = await publish({**hint, "event": str(event)})
            assert accepted["decision"] != "event_capacity"
        exhausted = await publish({**hint, "event": "overflow"})
        assert (
            exhausted["decision"] == "event_capacity"
            and exhausted["metadata"]["events"] == 64
        )
        await generate("competitor-one")
        retired = await generate("competitor-two")
        assert (
            retired["metadata"]["sessions"] == 2 and retired["metadata"]["events"] == 0
        )
        assert (await publish(hint))["decision"] == "unobserved"
        reset = await client.post(
            "/reset_prefix_cache?reset_external=true", headers=headers
        )
        assert reset.status_code == 200 and reset.json()["success"]
        cleared = await publish(hint)
        assert cleared["decision"] == "clock_or_deadline"
        assert (
            cleared["metadata"]["sessions"]
            == cleared["metadata"]["events"]
            == cleared["metadata"]["charged_bytes"]
            == 0
        )
        acknowledgements.append(
            {
                "exhausted": exhausted,
                "retired": retired["metadata"],
                "cleared": cleared["metadata"],
            }
        )
    async with client.stream(
        "POST",
        "/v1/completions",
        headers=headers,
        json={
            "model": model,
            "prompt": list(range(1000, 1096)),
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        async for _chunk in response.aiter_bytes():
            break
    recovered = await client.post(
        "/v1/completions",
        headers=headers,
        json={
            "model": model,
            "prompt": list(range(1000, 1096)),
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    assert recovered.status_code == 200 and recovered.json()["choices"]
    return {
        "cycles": acknowledgements,
        "disconnect_then_generate": True,
        "limits": "bounded two-session/64-event configuration; not a default-capacity throughput or lifetime claim",
    }

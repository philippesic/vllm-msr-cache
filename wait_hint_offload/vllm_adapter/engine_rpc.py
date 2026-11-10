from __future__ import annotations

from typing import Any

from vllm.v1.engine.core import EngineCore

PUBLISH_METHOD = "publish_kv_connector_wait_hint"


def publish(core: EngineCore, hint: dict[str, Any]) -> dict[str, Any]:
    connector = core.scheduler.get_kv_connector()
    publish_hint = getattr(connector, "publish_wait_hint", None)
    if not callable(publish_hint):
        return {"status": "unsupported", "reason": "wait-hint connector not configured"}
    result: dict[str, Any] = publish_hint(hint)
    return result


def install() -> None:
    existing = getattr(EngineCore, PUBLISH_METHOD, None)
    if existing is None:
        setattr(EngineCore, PUBLISH_METHOD, publish)
    elif existing is not publish:
        raise RuntimeError(f"EngineCore.{PUBLISH_METHOD} already defined by vLLM")

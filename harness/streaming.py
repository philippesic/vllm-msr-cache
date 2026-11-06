"""Exact-token SSE accounting shared by serving benchmarks."""
import hashlib
import json
import time


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


async def consume_campaign_sse(lines) -> dict[str, object]:
    """Consume one completion stream with exact-token and terminal proof."""
    output_tokens: list[int] = []
    token_receipt_ns: list[int] = []
    chunk_receipt_ns: list[int] = []
    finish_reasons: list[str] = []
    done_seen = False
    first_token_ns: int | None = None
    async for raw_line in lines:
        line = raw_line.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        raw = line.removeprefix("data:").strip()
        if done_seen:
            raise ValueError("stream data arrived after DONE")
        if raw == "[DONE]":
            done_seen = True
            continue
        chunk = json.loads(raw)
        received_ns = time.monotonic_ns()
        chunk_receipt_ns.append(received_ns)
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise TypeError("stream chunk choices must be a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise TypeError("stream choice must be an object")
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if finish_reason not in {"stop", "length"}:
                    raise ValueError("finish_reason must be stop or length")
                finish_reasons.append(finish_reason)
            token_ids = choice.get("token_ids")
            text_value = choice.get("text")
            if token_ids is None:
                if text_value:
                    raise ValueError("stream text arrived without exact token_ids")
                continue
            if not isinstance(token_ids, list) or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in token_ids
            ):
                raise ValueError("malformed stream token_ids")
            if token_ids and first_token_ns is None:
                first_token_ns = received_ns
            output_tokens.extend(token_ids)
            token_receipt_ns.extend(received_ns for _ in token_ids)
    if len(finish_reasons) > 1:
        raise ValueError("stream contains multiple terminal finish reasons")
    terminal_complete = done_seen and len(finish_reasons) == 1
    return {
        "output_token_ids": output_tokens,
        "output_token_digest": _digest(output_tokens),
        "token_receipt_ns": token_receipt_ns,
        "chunk_receipt_ns": chunk_receipt_ns,
        "finish_reasons": finish_reasons,
        "done_seen": done_seen,
        "terminal_complete": terminal_complete,
        "first_token_ns": first_token_ns,
    }


from __future__ import annotations

import json
from typing import Any

import pytest

from tools.benchmarking.w1 import canonical_messages, episode, partition, repository


class FixtureTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize and not enable_thinking
        text = "".join(
            message["role"]
            + ":"
            + message["content"]
            + json.dumps(message.get("tool_calls", []))
            + "|"
            for message in messages
        )
        return text + ("assistant:" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(map(ord, text))


def test_w1_canonical_tool_content_and_repository_partition() -> None:
    source = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "natural"},
                {"type": "text", "text": " prefix"},
            ],
        },
        {
            "role": "assistant",
            "content": "reason",
            "thought": "reason",
            "action": "bash pwd",
            "tool_calls": [
                {
                    "id": "a",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": [{"type": "text", "text": "observation"}],
            "tool_call_ids": ["a"],
        },
    ]
    result = canonical_messages(source)
    assert result[0]["content"] == "natural prefix"
    assert result[1]["content"] == "reason" and "thought" not in result[1]
    assert result[1]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert result[2]["tool_call_id"] == "a" and result[2]["content"] == "observation"
    assert repository("org__repo.with.dot.abcdef1234.task") == "org/repo.with.dot"
    assert partition("org/repo.with.dot") == partition(
        repository("org__repo.with.dot.abcdef1234.other")
    )
    with pytest.raises(ValueError, match="ambiguous_tool_result"):
        canonical_messages(
            [{"role": "tool", "content": "result", "tool_call_ids": ["a", "b"]}]
        )


def test_w1_causal_current_turn_eligibility_includes_truncated_last_action() -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "natural prefix"}]
    for index in range(9):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "answer",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "bash" if index != 1 else "submit",
                                "arguments": {},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "observation"},
            ]
        )
    observed = episode(messages, FixtureTokenizer())
    assert len(observed["turns"]) == 8 and observed["ending"] == "turn_limit"
    assert observed["turns"][0]["post_response_eligible"]
    assert not observed["turns"][1]["post_response_eligible"]
    assert observed["turns"][-1]["post_response_eligible"]
    assert observed["turns"][1]["previous_request_lcp_tokens"] > 0
    two_turns = episode(messages[:5], FixtureTokenizer())
    assert two_turns["turns"] == observed["turns"][:2]


def test_w1_output_cap_and_first_context_overflow_are_chronological() -> None:
    messages = [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "a" * 1000},
        {"role": "tool", "content": "observed"},
        {"role": "assistant", "content": "b"},
        {"role": "tool", "content": "x" * 6200},
        {"role": "assistant", "content": "would overflow"},
        {"role": "user", "content": "later short"},
        {"role": "assistant", "content": "never selected"},
    ]
    observed = episode(messages, FixtureTokenizer())
    assert len(observed["turns"]) == 2 and observed["ending"] == "context_limit"
    assert observed["turns"][0]["requested_output_tokens"] == 256
    assert observed["turns"][0]["recorded_output_tokens"] > 1000
    assert observed["turns"][1]["requested_output_tokens"] < 256

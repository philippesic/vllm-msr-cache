import json

from benchmarks.research_checks.hybrid_source_audit import audit_span_rows, audit_transcripts


def span(start, end, inp, output, conversation="a"):
    return dict(start_time=f"2026-01-01T00:00:{start:02d}+00:00",
                end_time=f"2026-01-01T00:00:{end:02d}+00:00",
                status={"code": 1}, attributes={
                    "gen_ai.operation.name": "chat", "gen_ai.conversation.id": conversation,
                    "gen_ai.input.messages": json.dumps(inp),
                    "gen_ai.output.messages": json.dumps(output),
                    "gen_ai.request.model": "test-model",
                })


def test_valid_message_continuation_still_requires_timing_and_tokenizer_proof():
    user = [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
    answer = [{"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}]
    a = span(0, 1, user, [{**answer[0], "finish_reason": "stop"}])
    b = span(3, 4, user + answer + user, answer)
    result = audit_span_rows([{"spans": [b, a]}])
    assert result["counts"]["structurally_usable_sequential_pairs"] == 1
    assert result["counts"]["sessions_without_observation_end"] == 1
    assert result["qualification"] == "not_established"


def test_parallel_and_missing_output_do_not_become_return_labels():
    user, answer = [{"role": "user"}], [{"role": "assistant"}]
    a = span(0, 5, user, answer)
    b = span(2, 6, user + answer, None)
    c = span(8, 9, user + answer + user, answer)
    result = audit_span_rows([{"spans": [a, b, c]}])["counts"]
    assert result["overlapping_calls"] == 1
    assert result["missing_output"] == 1
    assert result["structurally_usable_sequential_pairs"] == 0


def test_missing_identity_and_different_conversations_never_pair():
    user = [{"role": "user"}]
    result = audit_span_rows([{"spans": [span(0, 1, user, None, None),
                                         span(2, 3, user, None, "b")]}])["counts"]
    assert result.get("adjacent_same_conversation_calls", 0) == 0
    assert result["missing_conversation_id"] == 1


def test_zero_duration_or_failed_calls_do_not_qualify_sequential_reuse():
    user, answer = [{"role": "user"}], [{"role": "assistant"}]
    a, b = span(0, 0, user, answer), span(2, 3, user + answer, answer)
    result = audit_span_rows([{"spans": [a, b]}])["counts"]
    assert result["zero_duration_calls"] == 1
    assert result["structurally_usable_sequential_pairs"] == 0
    a["end_time"] = "2026-01-01T00:00:01+00:00"
    a["status"]["code"] = 2
    result = audit_span_rows([{"spans": [a, b]}])["counts"]
    assert result["failed_calls"] == 1
    assert result["structurally_usable_sequential_pairs"] == 0


def test_transcript_tool_identity_and_terminal_censoring(tmp_path):
    path = tmp_path / "fixture_claude_code_session.jsonl"
    rows = [
        dict(type="assistant", timestamp="2026-01-01T00:00:00Z", message={"content": [
            {"type": "tool_use", "id": "tool-1"}]}),
        dict(type="user", timestamp="2026-01-01T00:00:02Z", message={"content": [
            {"type": "tool_result", "tool_use_id": "tool-1"}]}),
        dict(type="assistant", timestamp="2026-01-01T00:00:03Z", message={"content": []}),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    result = audit_transcripts([path])
    assert result["counts"]["linked_tool_result_pairs"] == 1
    assert result["counts"]["censored_terminal_assistant"] == 1
    assert result["qualification"] == "not_established"

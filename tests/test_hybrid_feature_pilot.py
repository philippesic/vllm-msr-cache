from benchmarks.research_checks.hybrid_feature_pilot import completed_example, split_for, task_group


def test_features_stop_at_completed_response_and_never_read_next_tool_result():
    user = {"role": "user", "content": "fix this"}
    answer = {"role": "assistant", "content": "running tests", "tool_calls": [{"id": "t"}]}
    previous = dict(session_id="swebench__repo__task__model", model="m", input=[user], output_length=7, pre_gap=.1)
    following = dict(previous, input=[user, answer, {"role": "tool", "content": "future secret"}], pre_gap=.4)
    a = completed_example(previous, following, 3)
    following["input"][-1]["content"] = "entirely different future tool output" * 100
    following["pre_gap"] = 99.
    b = completed_example(previous, following, 3)
    assert a["messages"] == b["messages"] == [user, answer]
    assert a["cheap"] == b["cheap"]
    assert a["target"] == 1 and b["target"] == 0


def test_model_variants_of_same_task_are_not_split_across_train_and_test():
    a, b = "swebench__repo__task__claude__run2", "swebench__repo__task__deepseek"
    assert task_group(a) == task_group(b)
    assert split_for(a) == split_for(b)


def test_missing_output_context_reset_and_cross_session_pairs_are_rejected():
    a = dict(session_id="a", model="m", input=[{"role": "user", "content": "hello"}], output_length=1, pre_gap=0.)
    assert completed_example(a, dict(a, input=a["input"] + [{"role": "tool"}]), 1) is None
    assert completed_example(a, dict(a, input=[{"role": "user", "content": "reset"}]), 1) is None
    assert completed_example(a, dict(a, session_id="b"), 1) is None

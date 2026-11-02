from app.models.state import history_reducer, sub_answers_reducer


class TestHistoryReducer:
    def test_appends_without_compaction_marker(self):
        left = [{"role": "user", "content": "hi"}]
        right = [{"role": "assistant", "content": "hello"}]
        result = history_reducer(left, right)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_replaces_when_compaction_marker_present(self):
        left = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        right = [{"role": "system", "content": "[summary]", "compacted": True}]
        result = history_reducer(left, right)
        assert result == [{"role": "system", "content": "[summary]", "compacted": True}]

    def test_empty_lists(self):
        assert history_reducer([], []) == []
        assert history_reducer([], [{"role": "user", "content": "hi"}]) == [
            {"role": "user", "content": "hi"}
        ]

    def test_multiple_compacted_messages_replace(self):
        left = [{"role": "user", "content": "old"}]
        right = [
            {"role": "system", "content": "[summary]", "compacted": True},
            {"role": "assistant", "content": "new"},
        ]
        result = history_reducer(left, right)
        assert result == right


class TestSubAnswersReducer:
    def test_accumulates_within_turn(self):
        left = [{"agent": "store_agent", "response": "a", "iteration": 1}]
        right = [{"agent": "policy_agent", "response": "b", "iteration": 1}]
        assert sub_answers_reducer(left, right) == left + right

    def test_reset_marker_drops_previous_turn_entries(self):
        left = [
            {"agent": "store_agent", "response": "旧设备回答", "iteration": 1},
        ]
        right = [{"_reset": True}]
        assert sub_answers_reducer(left, right) == []

    def test_reset_marker_keeps_same_step_entries(self):
        left = [{"agent": "store_agent", "response": "旧回答", "iteration": 1}]
        right = [{"_reset": True}, {"agent": "store_agent", "response": "新回答", "iteration": 1}]
        assert sub_answers_reducer(left, right) == [
            {"agent": "store_agent", "response": "新回答", "iteration": 1}
        ]

    def test_no_reset_with_empty_right_keeps_left(self):
        left = [{"agent": "store_agent", "response": "a", "iteration": 1}]
        assert sub_answers_reducer(left, []) == left

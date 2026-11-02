from app.graph.parallel import build_parallel_sends, plan_dispatch
from app.models.state import make_agent_state


class TestPlanDispatch:
    def test_done_when_all_completed(self):
        mode, targets = plan_dispatch(
            agent_queue=["store_agent", "policy_agent"],
            intent_queue=["STORE", "POLICY"],
            completed=["store_agent", "policy_agent"],
        )
        assert mode == "done"
        assert targets == []

    def test_parallel_when_all_independent(self):
        mode, targets = plan_dispatch(
            agent_queue=["store_agent", "policy_agent"],
            intent_queue=["STORE", "POLICY"],
            completed=[],
        )
        assert mode == "parallel"
        assert targets == ["store_agent", "policy_agent"]

    def test_serial_when_dependent(self):
        mode, targets = plan_dispatch(
            agent_queue=["material_agent", "supply_agent"],
            intent_queue=["MATERIAL", "SUPPLY"],
            completed=[],
        )
        assert mode == "serial"
        assert targets == ["material_agent"]

    def test_partial_completion_triggers_parallel_for_remaining(self):
        mode, targets = plan_dispatch(
            agent_queue=["store_agent", "policy_agent", "supply_agent"],
            intent_queue=["STORE", "POLICY", "SUPPLY"],
            completed=["store_agent"],
        )
        assert mode == "parallel"
        assert targets == ["policy_agent", "supply_agent"]

    def test_serial_advances_after_first_completed(self):
        mode, targets = plan_dispatch(
            agent_queue=["material_agent", "supply_agent"],
            intent_queue=["MATERIAL", "SUPPLY"],
            completed=["material_agent"],
        )
        assert mode == "serial"
        assert targets == ["supply_agent"]


class TestBuildParallelSends:
    def test_builds_sends_for_each_agent(self):
        state = make_agent_state(question="test", thread_id="t1")
        sends = build_parallel_sends(["store_agent", "policy_agent"], state)
        assert len(sends) == 2
        assert sends[0].node == "store_agent"
        assert sends[1].node == "policy_agent"
        assert sends[0].arg["question"] == "test"
        assert sends[1].arg["question"] == "test"

    def test_questions_go_into_branch_key_not_question(self):
        state = make_agent_state(question="full question", thread_id="t1")
        sends = build_parallel_sends(
            ["store_agent", "policy_agent"], state, ["查设备", "说报损流程"]
        )
        assert sends[0].arg["question"] == "full question"
        assert sends[0].arg["_branch_question"] == "查设备"
        assert sends[1].arg["_branch_question"] == "说报损流程"

    def test_empty_list(self):
        state = make_agent_state(question="test", thread_id="t1")
        sends = build_parallel_sends([], state)
        assert sends == []

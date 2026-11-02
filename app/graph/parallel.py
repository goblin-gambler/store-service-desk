from langgraph.types import Send

from app.intent.multi_intent import are_independent
from app.models.state import AgentState


def plan_dispatch(
    agent_queue: list[str], intent_queue: list[str], completed: list[str]
) -> tuple[str, list[str]]:
    completed_set = set(completed)
    remaining = [
        (intent, agent)
        for intent, agent in zip(intent_queue, agent_queue, strict=True)
        if agent not in completed_set
    ]

    if not remaining:
        return "done", []

    all_independent = True
    for i in range(len(remaining)):
        for j in range(i + 1, len(remaining)):
            if not are_independent(remaining[i][0], remaining[j][0]):
                all_independent = False
                break
        if not all_independent:
            break

    if all_independent and len(remaining) > 1:
        return "parallel", [agent for _, agent in remaining]

    return "serial", [remaining[0][1]]


def build_parallel_sends(
    agent_names: list[str],
    state: AgentState,
    questions: list[str] | None = None,
) -> list[Send]:
    """为并行分支构造 Send。

    Send 的第二个参数是该分支节点的输入，必须带上当前状态；
    传空字典会让并发 Agent 拿不到 question / user_id，只能返回"无法识别身份"。
    `questions` 与 agent_names 一一对应。子问题放在 `_branch_question` 而不是
    直接改 `question`：父图通道里的 `question` 是整句原问，分支节点只把子问题
    覆盖到传给 Agent 的副本上（见 ``build_agent_subgraph``），这样分支之间不会
    争抢同一个通道值。
    """
    snapshot = dict(state)
    sends: list[Send] = []
    for idx, name in enumerate(agent_names):
        branch_state = dict(snapshot)
        if questions and idx < len(questions) and questions[idx]:
            branch_state["_branch_question"] = questions[idx]
        sends.append(Send(name, branch_state))
    return sends

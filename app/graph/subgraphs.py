import json
import logging
from typing import cast

from app.agents.base import BaseAgent
from app.context.masking import mask_observation
from app.models.state import AgentState
from app.observability.metrics import (
    record_agent_context_reduction,
    record_agent_context_tokens,
)

logger = logging.getLogger(__name__)


def _filter_state(state: AgentState, allowed_keys: list[str]) -> AgentState:
    """Return a copy of ``state`` containing only ``allowed_keys``."""
    return cast(AgentState, {k: v for k, v in state.items() if k in allowed_keys})


def _estimate_state_tokens(state: AgentState) -> int:
    """Estimate token count for a serialized state dictionary."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        serialized = json.dumps(state, ensure_ascii=False, default=str)
        return len(encoder.encode(serialized))
    except (ImportError, OSError):
        serialized = json.dumps(state, ensure_ascii=False, default=str)
        return len(serialized) // 4


def get_agent_tools(agent_name: str) -> list[dict[str, str]]:
    """Return tool definitions scoped to ``agent_name``.

    Args:
        agent_name: Name of the expert agent.

    Returns:
        List of tool metadata dicts with ``name`` and ``description`` keys.
    """
    from app.graph.workflow import get_agent_tool_scope

    tool_names = get_agent_tool_scope(agent_name)
    return [{"name": name, "description": f"Tool '{name}' for {agent_name}"} for name in tool_names]


def build_agent_subgraph(agent: BaseAgent, allowed_keys: list[str] | None = None):
    """Build the isolated graph node for ``agent``.

    历史实现返回一个单节点的 ``StateGraph`` 子图。编译后的子图作为父图节点时，
    会把**整个终态**回写给父图：``history`` / ``sub_answers``（``operator.add``
    通道）被重复累加，并行扇出（Send）时多个分支同时回写 ``question``，直接抛
    ``InvalidUpdateError: At key 'question': Can receive only one value per step``。
    该子图本来就只有一个节点，因此这里直接返回节点函数：上下文隔离边界和指标
    采集保持不变，只回写真正的增量。

    Args:
        agent: The expert agent to wrap.
        allowed_keys: Optional list of state keys the agent is allowed to see.
            When provided, the full ``AgentState`` is filtered before being
            passed to ``agent.process()``. Isolation happens at the node
            boundary so agents cannot access data produced by other experts.
    """

    async def agent_node(state: AgentState) -> dict:
        full_tokens = _estimate_state_tokens(state)
        isolated_state = _filter_state(state, allowed_keys) if allowed_keys else state
        # 并行多意图分支的子问题：只覆盖传给 Agent 的副本，不改父图通道。
        branch_question = state.get("_branch_question")
        if branch_question:
            isolated_state = {**isolated_state, "question": branch_question}
        filtered_tokens = _estimate_state_tokens(isolated_state)
        reduction_pct = (
            round(((full_tokens - filtered_tokens) / full_tokens) * 100, 2)
            if full_tokens > 0
            else 0.0
        )
        logger.info(
            "[ContextIsolation] agent=%s full_tokens=%s filtered_tokens=%s reduction=%s%%",
            agent.name,
            full_tokens,
            filtered_tokens,
            reduction_pct,
        )
        record_agent_context_tokens(tokens=filtered_tokens, agent_name=agent.name)
        if full_tokens > 0:
            reduction_ratio = (full_tokens - filtered_tokens) / full_tokens
            record_agent_context_reduction(agent_name=agent.name, ratio=round(reduction_ratio, 4))

        result = await agent.process(isolated_state)
        updated_state = result.get("updated_state") or {}
        # sub_answers 是给观测/记忆用的，需要脱敏；但父图的 state 必须拿到
        # 真实结果，否则 evaluator 看不到 retrieval_result，会把所有回答判成
        # “无检索结果”+低置信度并强制转人工。
        state_keys = set(AgentState.__annotations__)
        passthrough = {k: v for k, v in updated_state.items() if k in state_keys}
        masked_state = mask_observation(updated_state) if updated_state else {}
        updates: dict = {
            "current_agent": agent.name,
            "sub_answers": [
                {
                    "agent": agent.name,
                    "response": result.get("response", ""),
                    "updated_state": masked_state,
                    "iteration": state.get("iteration_count", 0),
                }
            ],
        }
        updates.update(passthrough)
        if result.get("response"):
            updates["answer"] = result["response"]
        return updates

    # 保留 ainvoke 便于单测/脚本直接调用（与编译子图的调用方式一致）。
    agent_node.ainvoke = agent_node  # type: ignore[attr-defined]
    return agent_node

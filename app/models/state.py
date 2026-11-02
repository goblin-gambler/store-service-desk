# app/models/state.py
import operator
from typing import Annotated, Any, NotRequired, TypedDict


def history_reducer(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer for AgentState.history that supports compaction.

    When the right side contains a compaction marker (``compacted: True``),
    it replaces the left side entirely instead of being concatenated.
    """
    if any(msg.get("compacted") for msg in right):
        return right
    return left + right


def _last_value(left: str | None, right: str | None) -> str | None:
    return right


def sub_answers_reducer(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer for ``AgentState.sub_answers``.

    ``sub_answers`` must accumulate across parallel ``Send`` branches within a
    turn, so it uses additive semantics. It is also checkpointed, which means
    the previous turn's entries survive into the next request. Since
    ``iteration_count`` restarts at 0 on every request, old entries would be
    mistaken for the current turn's results: the supervisor skips agents it
    thinks already ran and the synthesis node replays the old answer.

    A reset marker at request start clears the accumulated entries, mirroring
    the ``compacted`` marker used by :func:`history_reducer`.
    """
    if any(sa.get("_reset") for sa in right):
        return [sa for sa in right if not sa.get("_reset")]
    return left + right


class AgentProcessResult(TypedDict):
    """Agent.process() 的统一返回类型"""

    response: str
    updated_state: NotRequired[dict[str, Any]]


class AgentState(TypedDict):
    """Agent 状态定义（连锁零售门店服务台）"""

    question: str
    user_id: int
    thread_id: str

    current_agent: Annotated[str | None, _last_value]
    next_agent: Annotated[str | None, _last_value]
    iteration_count: Annotated[int, _last_value]
    retry_requested: Annotated[bool, _last_value]

    history: Annotated[list[dict], history_reducer]

    retrieval_result: Annotated[dict[str, Any] | None, _last_value]

    # 领域数据（每个专家 Agent 一份）
    store_data: Annotated[dict[str, Any] | None, _last_value]
    supply_data: Annotated[dict[str, Any] | None, _last_value]
    settlement_data: Annotated[dict[str, Any] | None, _last_value]
    material_data: Annotated[dict[str, Any] | None, _last_value]
    staff_data: Annotated[dict[str, Any] | None, _last_value]
    product_data: Annotated[dict[str, Any] | None, _last_value]

    audit_level: Annotated[str | None, _last_value]
    audit_log_id: Annotated[int | None, _last_value]
    audit_reason: Annotated[str | None, _last_value]

    confidence_score: Annotated[float | None, _last_value]
    confidence_signals: Annotated[dict | None, _last_value]

    needs_human_transfer: Annotated[bool, _last_value]
    transfer_reason: Annotated[str | None, _last_value]

    messages: Annotated[list[dict[str, Any]], operator.add]
    answer: Annotated[str, _last_value]

    # 报损流程
    loss_flow_active: Annotated[bool | None, _last_value]
    loss_data: Annotated[dict[str, Any] | None, _last_value]

    # Intent & clarification fields produced by the graph
    intent_result: Annotated[dict[str, Any] | None, _last_value]
    slots: Annotated[dict[str, Any] | None, _last_value]
    awaiting_clarification: Annotated[bool | None, _last_value]
    clarification_state: Annotated[dict[str, Any] | None, _last_value]

    execution_mode: Annotated[str | None, _last_value]
    sub_answers: Annotated[list[dict[str, Any]], sub_answers_reducer]
    supervisor_reasoning: Annotated[str | None, _last_value]
    synthesized_answer: Annotated[str | None, _last_value]
    pending_agent_results: Annotated[list[str] | None, _last_value]
    pending_agent_questions: Annotated[list[str] | None, _last_value]
    completed_agents: Annotated[list[str] | None, _last_value]
    memory_context: Annotated[dict[str, Any] | None, _last_value]
    experiment_variant_id: Annotated[int | None, _last_value]
    memory_context_config: Annotated[dict[str, Any] | None, _last_value]

    context_tokens: Annotated[int | None, _last_value]
    context_utilization: Annotated[float | None, _last_value]

    variant_llm_model: Annotated[str | None, _last_value]
    variant_retriever_top_k: Annotated[int | None, _last_value]
    variant_reranker_enabled: Annotated[bool | None, _last_value]


def make_agent_state(
    *,
    question: str,
    user_id: int = 1,
    thread_id: str = "default",
    current_agent: str | None = None,
    next_agent: str | None = None,
    iteration_count: int = 0,
    retry_requested: bool = False,
    history: list[dict] | None = None,
    retrieval_result: dict[str, Any] | None = None,
    store_data: dict[str, Any] | None = None,
    supply_data: dict[str, Any] | None = None,
    settlement_data: dict[str, Any] | None = None,
    material_data: dict[str, Any] | None = None,
    staff_data: dict[str, Any] | None = None,
    product_data: dict[str, Any] | None = None,
    audit_level: str | None = None,
    audit_log_id: int | None = None,
    audit_reason: str | None = None,
    confidence_score: float | None = None,
    confidence_signals: dict | None = None,
    needs_human_transfer: bool = False,
    transfer_reason: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    answer: str = "",
    loss_flow_active: bool | None = None,
    loss_data: dict[str, Any] | None = None,
    intent_result: dict[str, Any] | None = None,
    slots: dict[str, Any] | None = None,
    awaiting_clarification: bool | None = None,
    clarification_state: dict[str, Any] | None = None,
    execution_mode: str | None = None,
    sub_answers: list[dict[str, Any]] | None = None,
    supervisor_reasoning: str | None = None,
    synthesized_answer: str | None = None,
    pending_agent_results: list[str] | None = None,
    pending_agent_questions: list[str] | None = None,
    completed_agents: list[str] | None = None,
    memory_context: dict[str, Any] | None = None,
    experiment_variant_id: int | None = None,
    memory_context_config: dict[str, Any] | None = None,
    context_tokens: int | None = None,
    context_utilization: float | None = None,
    variant_llm_model: str | None = None,
    variant_retriever_top_k: int | None = None,
    variant_reranker_enabled: bool | None = None,
) -> AgentState:
    return {
        "question": question,
        "user_id": user_id,
        "thread_id": thread_id,
        "history": history if history is not None else [],
        "current_agent": current_agent,
        "next_agent": next_agent,
        "iteration_count": iteration_count,
        "retry_requested": retry_requested,
        "retrieval_result": retrieval_result,
        "store_data": store_data,
        "supply_data": supply_data,
        "settlement_data": settlement_data,
        "material_data": material_data,
        "staff_data": staff_data,
        "product_data": product_data,
        "audit_level": audit_level,
        "audit_log_id": audit_log_id,
        "audit_reason": audit_reason,
        "confidence_score": confidence_score,
        "confidence_signals": confidence_signals,
        "messages": messages if messages is not None else [],
        "answer": answer,
        "loss_flow_active": loss_flow_active,
        "loss_data": loss_data,
        "needs_human_transfer": needs_human_transfer,
        "transfer_reason": transfer_reason,
        "intent_result": intent_result,
        "slots": slots,
        "awaiting_clarification": awaiting_clarification,
        "clarification_state": clarification_state,
        "execution_mode": execution_mode,
        "sub_answers": sub_answers if sub_answers is not None else [],
        "supervisor_reasoning": supervisor_reasoning,
        "synthesized_answer": synthesized_answer,
        "pending_agent_results": pending_agent_results,
        "pending_agent_questions": pending_agent_questions,
        "completed_agents": completed_agents,
        "memory_context": memory_context,
        "experiment_variant_id": experiment_variant_id,
        "memory_context_config": memory_context_config,
        "context_tokens": context_tokens,
        "context_utilization": context_utilization,
        "variant_llm_model": variant_llm_model,
        "variant_retriever_top_k": variant_retriever_top_k,
        "variant_reranker_enabled": variant_reranker_enabled,
    }

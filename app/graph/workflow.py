import logging

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    build_complaint_node,
    build_decider_node,
    build_evaluator_node,
    build_material_node,
    build_memory_node,
    build_policy_node,
    build_product_node,
    build_router_node,
    build_settlement_node,
    build_staff_node,
    build_store_node,
    build_supervisor_node,
    build_supply_node,
    build_synthesis_node,
)
from app.graph.subgraphs import build_agent_subgraph
from app.models.state import AgentState

logger = logging.getLogger(__name__)


_COMMON_ALLOWED_KEYS = [
    "question",
    "user_id",
    "thread_id",
    "history",
    "memory_context",
    "memory_context_config",
    "intent_result",
    "slots",
    "iteration_count",
    "experiment_variant_id",
    "variant_llm_model",
    "variant_retriever_top_k",
    "variant_reranker_enabled",
    "context_tokens",
    "context_utilization",
]

_AGENT_ALLOWED_KEYS: dict[str, list[str]] = {
    "policy_agent": [*_COMMON_ALLOWED_KEYS, "retrieval_result", "answer"],
    "store_agent": [
        *_COMMON_ALLOWED_KEYS,
        "store_data",
        "loss_data",
        "loss_flow_active",
        "retrieval_result",
        "audit_level",
        "audit_log_id",
        "audit_reason",
    ],
    "supply_agent": [*_COMMON_ALLOWED_KEYS, "supply_data", "retrieval_result"],
    "settlement_agent": [*_COMMON_ALLOWED_KEYS, "settlement_data", "retrieval_result"],
    "material_agent": [*_COMMON_ALLOWED_KEYS, "material_data", "retrieval_result"],
    "staff_agent": [*_COMMON_ALLOWED_KEYS, "staff_data", "retrieval_result"],
    "product": [*_COMMON_ALLOWED_KEYS, "product_data", "retrieval_result"],
    "complaint": [
        *_COMMON_ALLOWED_KEYS,
        "retrieval_result",
        "audit_level",
        "audit_log_id",
        "audit_reason",
    ],
}

_AGENT_TOOL_SCOPES: dict[str, list[str]] = {
    "policy_agent": ["retrieve_knowledge"],
    "store_agent": ["store", "submit_loss_report", "check_loss_status"],
    "supply_agent": ["supply"],
    "settlement_agent": ["settlement"],
    "material_agent": ["material"],
    "staff_agent": ["staff"],
    "product": ["product"],
    "complaint": ["create_ticket", "escalate_ticket", "get_ticket_status"],
}


def create_workflow(
    router_agent,
    policy_agent,
    store_agent,
    supply_agent,
    settlement_agent,
    material_agent,
    staff_agent,
    evaluator,
    supervisor_agent=None,
    product_agent=None,
    complaint_agent=None,
    llm=None,
    structured_manager=None,
    vector_manager=None,
):
    workflow = StateGraph(AgentState)  # type: ignore

    workflow.add_node(
        "router_node",
        build_router_node(router_agent),  # type: ignore
        metadata={"tags": ["router_node", "internal"]},
    )

    if supervisor_agent is not None and llm is not None:
        workflow.add_node(
            "supervisor_node",
            build_supervisor_node(supervisor_agent),  # type: ignore
            metadata={"tags": ["supervisor_node", "internal"]},
        )
        workflow.add_node(
            "synthesis_node",
            build_synthesis_node(llm),  # type: ignore
            metadata={"tags": ["synthesis_node", "internal"]},
        )

    if supervisor_agent is not None:
        workflow.add_node(
            "policy_agent",
            build_agent_subgraph(
                policy_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("policy_agent")
            ),
            metadata={"tags": ["policy_agent", "user_visible"]},
        )
        workflow.add_node(
            "store_agent",
            build_agent_subgraph(store_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("store_agent")),
            metadata={"tags": ["store_agent", "user_visible"]},
        )
        workflow.add_node(
            "supply_agent",
            build_agent_subgraph(supply_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("supply_agent")),
            metadata={"tags": ["supply_agent", "user_visible"]},
        )
        workflow.add_node(
            "settlement_agent",
            build_agent_subgraph(
                settlement_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("settlement_agent")
            ),
            metadata={"tags": ["settlement_agent", "user_visible"]},
        )
        workflow.add_node(
            "material_agent",
            build_agent_subgraph(
                material_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("material_agent")
            ),
            metadata={"tags": ["material_agent", "user_visible"]},
        )
        workflow.add_node(
            "staff_agent",
            build_agent_subgraph(staff_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("staff_agent")),
            metadata={"tags": ["staff_agent", "user_visible"]},
        )
        if product_agent is not None:
            workflow.add_node(
                "product",
                build_agent_subgraph(
                    product_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("product")
                ),
                metadata={"tags": ["product", "user_visible"]},
            )
        if complaint_agent is not None:
            workflow.add_node(
                "complaint",
                build_agent_subgraph(
                    complaint_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("complaint")
                ),
                metadata={"tags": ["complaint", "user_visible"]},
            )
    else:
        workflow.add_node(
            "policy_agent",
            build_policy_node(policy_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("policy_agent")),  # type: ignore
            metadata={"tags": ["policy_agent", "user_visible"]},
        )
        workflow.add_node(
            "store_agent",
            build_store_node(store_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("store_agent")),  # type: ignore
            metadata={"tags": ["store_agent", "user_visible"]},
        )
        workflow.add_node(
            "supply_agent",
            build_supply_node(
                supply_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("supply_agent")
            ),  # type: ignore
            metadata={"tags": ["supply_agent", "user_visible"]},
        )
        workflow.add_node(
            "settlement_agent",
            build_settlement_node(
                settlement_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("settlement_agent")
            ),  # type: ignore
            metadata={"tags": ["settlement_agent", "user_visible"]},
        )
        workflow.add_node(
            "material_agent",
            build_material_node(
                material_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("material_agent")
            ),  # type: ignore
            metadata={"tags": ["material_agent", "user_visible"]},
        )
        workflow.add_node(
            "staff_agent",
            build_staff_node(staff_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("staff_agent")),  # type: ignore
            metadata={"tags": ["staff_agent", "user_visible"]},
        )
        if product_agent is not None:
            workflow.add_node(
                "product",
                build_product_node(product_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("product")),  # type: ignore
                metadata={"tags": ["product", "user_visible"]},
            )
        if complaint_agent is not None:
            workflow.add_node(
                "complaint",
                build_complaint_node(
                    complaint_agent, allowed_keys=_AGENT_ALLOWED_KEYS.get("complaint")
                ),  # type: ignore
                metadata={"tags": ["complaint", "user_visible"]},
            )

    workflow.add_node(
        "memory_node",
        build_memory_node(
            structured_manager,
            vector_manager,
            use_supervisor=supervisor_agent is not None,
        ),  # type: ignore
        metadata={"tags": ["memory_node", "internal"]},
    )
    workflow.add_node(
        "evaluator_node",
        build_evaluator_node(evaluator),  # type: ignore
        metadata={"tags": ["evaluator_node", "confidence_eval", "internal"]},
    )
    workflow.add_node(
        "decider_node",
        build_decider_node(vector_manager),  # type: ignore
        metadata={"tags": ["decider_node", "internal"]},
    )

    workflow.add_edge(START, "router_node")
    for agent_node in (
        "policy_agent",
        "store_agent",
        "supply_agent",
        "settlement_agent",
        "material_agent",
        "staff_agent",
        "product",
        "complaint",
    ):
        if agent_node in workflow.nodes:
            workflow.add_edge(
                agent_node,
                "synthesis_node" if supervisor_agent is not None else "evaluator_node",
            )

    workflow.add_edge("evaluator_node", "decider_node")
    workflow.add_edge("decider_node", END)

    return workflow


def get_agent_tool_scope(agent_name: str) -> list[str]:
    """Return the list of tool names relevant to ``agent_name``."""
    return _AGENT_TOOL_SCOPES.get(agent_name, [])


async def compile_app_graph(
    router_agent,
    policy_agent,
    store_agent,
    supply_agent,
    settlement_agent,
    material_agent,
    staff_agent,
    evaluator,
    checkpointer,
    supervisor_agent=None,
    product_agent=None,
    complaint_agent=None,
    llm=None,
    structured_manager=None,
    vector_manager=None,
):
    logger.info("Compiling LangGraph multi-agent workflow (store service desk)...")

    workflow = create_workflow(
        router_agent,
        policy_agent,
        store_agent,
        supply_agent,
        settlement_agent,
        material_agent,
        staff_agent,
        evaluator,
        supervisor_agent=supervisor_agent,
        product_agent=product_agent,
        complaint_agent=complaint_agent,
        llm=llm,
        structured_manager=structured_manager,
        vector_manager=vector_manager,
    )
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled successfully")
    return compiled

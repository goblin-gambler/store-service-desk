import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.base import BaseAgent
from app.graph.parallel import plan_dispatch
from app.models.state import AgentProcessResult, AgentState

logger = logging.getLogger(__name__)

_INTENT_TO_AGENT: dict[str, str] = {
    "STORE": "store_agent",
    "LOSS": "store_agent",
    "POLICY": "policy_agent",
    "SUPPLY": "supply_agent",
    "SETTLEMENT": "settlement_agent",
    "PRODUCT": "product",
    "MATERIAL": "material_agent",
    "COMPLAINT": "complaint",
    "STAFF": "staff_agent",
    "OTHER": "policy_agent",
}


class SupervisorAgent(BaseAgent):
    """Supervisor agent responsible for routing and execution-mode planning.

    The Supervisor operates on the **full** ``AgentState`` because it needs
    visibility of all intent results, completed agents, and pending work to
    make routing decisions. Context isolation for individual expert agents is
    enforced at the **subgraph boundary** (see ``app/graph/subgraphs.py``),
    not inside this class.
    """

    def __init__(self, llm: BaseChatModel):
        super().__init__(name="supervisor", llm=llm, system_prompt=None)

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override
        from app.agents.config_loader import is_agent_enabled

        intent_result = state.get("intent_result") or {}
        primary = intent_result.get("primary_intent")
        slots = state.get("slots") or {}
        pending = slots.get("pending_intents", [])
        iteration = state.get("iteration_count", 0)
        completed = {
            sa["agent"] for sa in state.get("sub_answers", []) if sa.get("iteration") == iteration
        }

        from app.agents.config_loader import get_target_agent_for_intent

        primary_str = (
            primary.value
            if hasattr(primary, "value")
            else str(primary)
            if primary is not None
            else "OTHER"
        )
        primary_agent = await get_target_agent_for_intent(
            primary_str, fallback=_INTENT_TO_AGENT.get(primary_str, "policy_agent")
        )
        if not await is_agent_enabled(primary_agent):
            logger.info("Agent %s is disabled, falling back to policy_agent", primary_agent)
            primary_agent = "policy_agent"
            if not await is_agent_enabled(primary_agent):
                logger.error("Fallback policy_agent is also disabled; forcing human transfer")
                return {
                    "response": "",
                    "updated_state": {
                        "next_agent": None,
                        "execution_mode": "serial",
                        "pending_agent_results": [],
                        "supervisor_reasoning": "Target and fallback agents disabled",
                        "needs_human_transfer": True,
                        "transfer_reason": "all_routing_targets_disabled",
                    },
                }

        plan_agents = [primary_agent]
        plan_intents = [primary_str]
        # 多意图时主 Agent 只拿自己那一段，否则它会抢答其它意图（例如把
        # "报损流程" 一起答了），也会让检索 / 评分被无关内容稀释。
        primary_question = state.get("question", "")
        if pending:
            primary_question = intent_result.get("raw_query") or primary_question
        plan_questions = [primary_question]
        seen = {primary_agent}
        for p in pending:
            pname = p.get("primary_intent")
            if not pname:
                continue
            pagent = await get_target_agent_for_intent(
                str(pname), fallback=_INTENT_TO_AGENT.get(str(pname), "policy_agent")
            )
            if not await is_agent_enabled(pagent):
                logger.info("Agent %s is disabled, skipping in plan", pagent)
                continue
            if pagent not in seen:
                plan_agents.append(pagent)
                plan_intents.append(str(pname))
                sub_question = p.get("query")
                if not sub_question:
                    # 旧的意图缓存可能没有 query 字段，提醒排查而不是静默串味。
                    logger.warning(
                        "pending_intent %s 缺少 query，回退整句问题；请清理意图缓存",
                        pname,
                    )
                    sub_question = state.get("question", "")
                plan_questions.append(sub_question)
                seen.add(pagent)

        mode, targets = plan_dispatch(plan_agents, plan_intents, list(completed))
        logger.info(
            "Supervisor routing primary_intent=%s mode=%s targets=%s completed=%s",
            primary_str,
            mode,
            targets,
            completed,
        )

        if mode == "done":
            return {
                "response": "",
                "updated_state": {
                    "next_agent": None,
                    "execution_mode": "serial",
                    "pending_agent_results": [],
                    "pending_agent_questions": [],
                    "supervisor_reasoning": "All agents completed",
                },
            }

        reasoning = f"Dispatch {mode}: {', '.join(targets)}"
        return {
            "response": "",
            "updated_state": {
                "next_agent": targets[0] if targets else None,
                "execution_mode": mode,
                "pending_agent_results": plan_agents,
                "pending_agent_questions": plan_questions,
                "supervisor_reasoning": reasoning,
            },
        }

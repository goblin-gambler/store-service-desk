import pytest
import pytest_asyncio
from sqlalchemy import text

from app.agents.supervisor import _INTENT_TO_AGENT, SupervisorAgent
from app.core.database import async_session_maker
from app.intent.multi_intent import are_independent
from app.models.state import make_agent_state
from tests._llm import DeterministicChatModel


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _clear_routing_rules():
    async with async_session_maker() as session:
        conn = await session.connection()
        await conn.execute(text("DELETE FROM routing_rules"))
        await session.commit()
    yield


@pytest.fixture
def supervisor():
    return SupervisorAgent(llm=DeterministicChatModel())


@pytest.mark.asyncio
async def test_supervisor_single_intent(supervisor):
    state = make_agent_state(
        question="运费怎么算",
        intent_result={"primary_intent": "POLICY"},
        slots={},
    )
    result = await supervisor.process(state)
    assert result["response"] == ""
    updated = result["updated_state"]
    assert updated["next_agent"] == "policy_agent"
    assert updated["execution_mode"] == "serial"
    assert "policy_agent" in updated["pending_agent_results"]


@pytest.mark.asyncio
async def test_supervisor_store_intent(supervisor):
    state = make_agent_state(
        question="帮我查下张江店的设备",
        intent_result={"primary_intent": "STORE"},
        slots={},
    )
    result = await supervisor.process(state)
    updated = result["updated_state"]
    assert updated["next_agent"] == "store_agent"
    assert "store_agent" in updated["pending_agent_results"]


@pytest.mark.asyncio
async def test_supervisor_product_intent(supervisor):
    state = make_agent_state(
        question="这个配方怎么做",
        intent_result={"primary_intent": "PRODUCT"},
        slots={},
    )
    result = await supervisor.process(state)
    updated = result["updated_state"]
    assert updated["next_agent"] == "product"
    assert "product" in updated["pending_agent_results"]


@pytest.mark.asyncio
async def test_supervisor_parallel_independent_intents(supervisor):
    state = make_agent_state(
        question="查下张江店的设备情况，顺便讲讲报损流程",
        intent_result={
            "primary_intent": "STORE",
            "raw_query": "查下张江店的设备情况，",
        },
        slots={
            "pending_intents": [
                {"primary_intent": "POLICY", "query": "讲讲报损流程"},
            ]
        },
    )
    result = await supervisor.process(state)
    updated = result["updated_state"]
    assert updated["execution_mode"] == "parallel"
    assert "store_agent" in updated["pending_agent_results"]
    assert "policy_agent" in updated["pending_agent_results"]
    # 每个并行分支拿到自己的子问题，而不是整句原问
    assert updated["pending_agent_questions"] == [
        "查下张江店的设备情况，",
        "讲讲报损流程",
    ]


@pytest.mark.asyncio
async def test_supervisor_missing_sub_query_falls_back_to_full_question(supervisor):
    """旧的意图缓存没有 query 字段时，退化为整句问题而不是崩溃。"""
    state = make_agent_state(
        question="查下张江店的设备情况，顺便讲讲报损流程",
        intent_result={"primary_intent": "STORE"},
        slots={"pending_intents": [{"primary_intent": "POLICY"}]},
    )
    result = await supervisor.process(state)
    updated = result["updated_state"]
    assert updated["pending_agent_questions"] == [
        "查下张江店的设备情况，顺便讲讲报损流程",
        "查下张江店的设备情况，顺便讲讲报损流程",
    ]


@pytest.mark.asyncio
async def test_supervisor_serial_dependent_intents(supervisor):
    state = make_agent_state(
        question="物料不够了，顺便问下配送到哪了",
        intent_result={"primary_intent": "MATERIAL"},
        slots={
            "pending_intents": [
                {"primary_intent": "SUPPLY"},
            ]
        },
    )
    result = await supervisor.process(state)
    updated = result["updated_state"]
    assert updated["execution_mode"] == "serial"
    assert updated["next_agent"] == "material_agent"
    assert "material_agent" in updated["pending_agent_results"]
    assert "supply_agent" in updated["pending_agent_results"]


def test_intent_mappings_complete():
    assert _INTENT_TO_AGENT["STORE"] == "store_agent"
    assert _INTENT_TO_AGENT["LOSS"] == "store_agent"
    assert _INTENT_TO_AGENT["POLICY"] == "policy_agent"
    assert _INTENT_TO_AGENT["PRODUCT"] == "product"
    assert _INTENT_TO_AGENT["MATERIAL"] == "material_agent"
    assert _INTENT_TO_AGENT["SUPPLY"] == "supply_agent"


def test_are_independent():
    assert are_independent("STORE", "POLICY") is True
    assert are_independent("SUPPLY", "POLICY") is True
    assert are_independent("MATERIAL", "SUPPLY") is False
    assert are_independent("LOSS", "STORE") is False


@pytest.mark.asyncio
async def test_supervisor_routes_with_enum_value(supervisor):
    """Regression: str(IntentCategory) returns 'IntentCategory.X'.
    Ensure routing works when primary_intent is an enum instance."""
    from app.intent.models import IntentCategory

    state = make_agent_state(
        question="查询门店设备",
        intent_result={"primary_intent": IntentCategory.STORE},
        slots={},
    )
    result = await supervisor.process(state)
    updated = result["updated_state"]
    assert updated["next_agent"] == "store_agent"
    assert are_independent("STORE", "STORE") is False

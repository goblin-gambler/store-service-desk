from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.base import BaseAgent
from app.intent.few_shot_loader import load_agent_examples
from app.models.state import AgentProcessResult, AgentState
from app.tools.registry import ToolRegistry

PRODUCT_SYSTEM_PROMPT = """你是连锁零售门店的「商品与配方」专家。

职责：
1. 根据问题检索门店商品目录，回答在售状态、零售价、规格、保质期
2. 回答配方、过敏原、储存条件等问题，必须基于目录中的商品描述
3. 对已下架商品说明下架状态，并建议替代品（仅当目录中确有替代品）

规则：
1. 严禁编造不存在的商品、配方或价格
2. 配方和过敏原涉及食品安全，目录没有明确写出的内容必须说"目录未标注"
3. 语气简洁，面向店员和店长"""


class ProductAgent(BaseAgent):
    def __init__(self, tool_registry: ToolRegistry, llm: BaseChatModel):
        super().__init__(name="product", llm=llm, system_prompt=PRODUCT_SYSTEM_PROMPT)
        self.tool_registry = tool_registry
        self._few_shot_examples = load_agent_examples("product")

    async def process(self, state: AgentState) -> AgentProcessResult:
        await self._load_config()
        override = await self._resolve_experiment_prompt(state)
        if override:
            self._dynamic_system_prompt = override

        tool_result = await self.tool_registry.execute("product", state)
        output = tool_result.output

        if output.get("status") == "not_found":
            response_text = "门店商品目录暂未初始化，请联系总部运营导入商品主数据。"
        elif output.get("direct_answer"):
            response_text = output["direct_answer"]
        else:
            products = output.get("products", [])
            if not products:
                response_text = "没有检索到匹配的商品。可以换一个商品名，或提供商品编码。"
            else:
                response_text = self._format_product_list(products)

        return {
            "response": response_text,
            "updated_state": {"product_data": output},
        }

    def _format_product_list(self, products: list[dict]) -> str:
        lines = ["为您找到以下商品："]
        for product in products:
            status = "在售" if product.get("in_stock") else "已下架"
            lines.append(
                f"• {product.get('name', '未知商品')} "
                f"（零售价 ¥{product.get('price', 'N/A')}，{status}）"
            )
            if product.get("description"):
                lines.append(f"  说明: {product['description'][:80]}")
        return "\n".join(lines)

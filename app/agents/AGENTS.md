# AGENTS.md - Expert Agents

> **IMPORTANT**: Read the root [`AGENTS.md`](../../AGENTS.md) first for repo-wide rules.

## Maintenance Contract

- This file is a living document for the expert agent fleet.
- Update this file in the same PR when adding new agents, changing conventions, or modifying key file mappings.

## Read Order

1. Read the root [`AGENTS.md`](../../AGENTS.md) for repo-wide rules, commands, and routing.
2. Read this file for agent-specific guidance.

## Overview

Expert agent fleet based on `BaseAgent` ABC covering store operations, loss reports, policies, supply, settlement, products, material requests, complaints, and staff. Each agent exposes a unified `async process(state) -> AgentProcessResult` entry point and delegates I/O to dedicated tool layers.

## Key Files

| Role | File | Notes |
|------|------|-------|
| Base class | `@app/agents/base.py` | `BaseAgent` ABC; `process()` → `AgentProcessResult` |
| Store & device | `@app/agents/store.py` + `@app/tools/store_tool.py` | 门店信息、设备状态、报修工单；通过 `LossService` 受理报损 |
| Product | `@app/agents/product.py` + `@app/tools/product_tool.py` | 商品与配方目录语义检索（Qdrant） |
| Supply | `@app/agents/supply.py` + `@app/tools/supply_tool.py` | 补货单、配送进度、冷链温度异常 |
| Settlement | `@app/agents/settlement.py` + `@app/tools/settlement_tool.py` | 日结、对账差异与发票状态 |
| Material | `@app/agents/material.py` + `@app/tools/material_tool.py` | 物料申领单（Redis 草稿态，提交后转正式领用单） |
| Staff | `@app/agents/staff.py` + `@app/tools/staff_tool.py` | 员工档案、门店角色、权限与培训记录 |
| Complaint | `@app/agents/complaint.py` + `@app/tools/complaint_tool.py` | LLM 自动分类 + 投诉工单创建 |
| Policy | `@app/agents/policy.py` | 制度与政策问答，混合 RAG 检索 |
| Supervisor | `@app/agents/supervisor.py` | Serial/parallel dispatch logic |
| Intent router | `@app/agents/router.py` | `IntentRouterAgent`; handles greeting personalization, retry logic, disabled agent fallback, and iteration limits |
| Config hot-reload | `@app/agents/config_loader.py` | Redis-cached routing rules and system prompts (60s TTL) |
| Experiment prompts | `@app/agents/base.py` | `BaseAgent._resolve_experiment_prompt()` resolves experiment variant prompts from database |
| Evaluator | `@app/agents/evaluator.py` | Agent response quality evaluation using `calculate_confidence_signals` from `@app/confidence/signals.py` |

## Commands

```bash
# Run agent module tests
uv run python -m pytest tests/agents/
```

## Code Style

General Python rules are defined in the root `AGENTS.md`. Agent-specific conventions:

- **Type hints**: Mandatory on all `BaseAgent` subclass methods and `AgentProcessResult` fields.
- **Docstrings**: Google-style docstrings for all public agent classes and `process()` methods.
- **No sync I/O**: Agents must not perform synchronous blocking calls; all tool/service access is `async`.
- **Config isolation**: Agent-specific prompt templates and routing rules live in `config_loader.py`; never hardcode prompts inside `process()`.

## Testing Patterns

- Use `make_agent_state()` from `@app/models/state.py` to construct agent state.
- Mock LLM calls and tool invocations; verify `AgentProcessResult` structure.
- Cover normal flows and edge cases (missing slots, permission checks).
- New agent → new test file under `tests/agents/`.

## Conventions

- **Unified entry**: All `BaseAgent` subclasses must implement `async process(self, state) -> AgentProcessResult`.
- **Hot-reload**: Each agent calls `await self._load_config()` inside `process()` to pick up config changes without restart.
- **Memory injection priority**: summaries → facts/profile → preferences → vector messages.
- **User isolation**: All loss-report, store, and memory queries must filter by `user_id`. Never return cross-user data.
- **Return contract**: `AgentProcessResult` must include `response` (string); optionally carry `updated_state`.

## Routing

The `SupervisorAgent` (see `@app/agents/supervisor.py`) routes intents to agents via `_INTENT_TO_AGENT`:

| Intent | Target Agent | Description |
|--------|--------------|-------------|
| `STORE` | `store_agent` | 门店信息、设备状态与报修 |
| `LOSS` | `store_agent` | 报损申请、审批与进度查询 |
| `POLICY` | `policy_agent` | 制度与政策问答（RAG） |
| `SUPPLY` | `supply_agent` | 补货单、配送与冷链异常 |
| `SETTLEMENT` | `settlement_agent` | 日结、对账与发票 |
| `PRODUCT` | `product` | 商品与配方检索 |
| `MATERIAL` | `material_agent` | 物料申领 |
| `COMPLAINT` | `complaint` | 投诉受理与工单 |
| `STAFF` | `staff_agent` | 员工档案与权限 |
| `OTHER` | `policy_agent` | 兜底意图 |

The `IntentRouterAgent` (`@app/agents/router.py`) produces these intent strings from user input.

## Anti-Patterns

- **Cross-layer coupling**: `supervisor.py` should minimize direct imports from `@app/graph/parallel.py`; prefer dependency injection for dispatch planning.
- **Direct DB access in agents**: Agents should not bypass the tool/service layer to access the database directly.
- **Stale AGENTS.md**: Adding a new agent without updating this file and the corresponding test suite.

## Related Files

- `@app/safety/AGENTS.md` — Output moderation pipeline integrated into agent responses.
- `@app/tools/AGENTS.md` — Tool layer consumed by agents.

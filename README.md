# 连锁零售门店服务台（Store Service Desk）

一个面向连锁零售门店的 AI 服务台：门店员工用自然语言提问，系统自动识别意图、调度专家 Agent 查询门店/设备/供货/结算数据或检索制度库，并在涉及报损、物料超额、食品安全等高风险动作时按金额与类型走人工审批。

> **二次开发声明**：本项目基于开源项目 ecommerce-smart-agent 的架构二次开发，保留其 LangGraph 多 Agent 编排、RAG、置信度评估、人工兜底、可观测性等框架能力，将业务域从"电商客服（订单/退款/物流/购物车）"整体替换为"连锁零售门店服务台（门店/报损/供货/物料/结算/排班）"。原始电商项目未被修改。
>
> 仓库中仍保留少量上游素材（如 `data/return_policy.md`、`data/shipping_policy.md`、`app/tasks/refund_tasks.py` 中的退款任务名），属于待清理的历史遗留，不影响本系统运行。

---

## 一、能做什么

| 能力 | 说明 |
|---|---|
| 门店与设备查询 | 门店信息、营业状态、设备在线率、故障设备清单 |
| 报修工单 | 创建/查询设备报修工单，Redis 保存上下文，支持多轮补充 |
| **报损分级审批** | 单笔 < 200 元自动核销；200–2000 元店长审批；> 2000 元区域督导审批；10 万元以上总部专项。**分级由代码判定，不交给模型** |
| 供货与到货 | 配送单查询、到货差异、缺货品种 |
| 日结与对账 | 日结金额、差额、差异超 ±500 元自动提示区域督导介入 |
| 物料申领 | 按配方/数量自动估价，校验月度额度，超额拦截并提示走督导追加审批 |
| 员工信息 | 排班、请假规则、加班规则、员工档案 |
| 政策咨询（RAG） | 制度库混合检索 + Self-RAG 文档评分 + 引用校验，答案必须带 `[来源: X]` |
| 商品/配方查询 | 商品属性、配方、保质期、过敏原（Qdrant `store_catalog`） |
| 食品安全与客诉 | 食品安全事件自动建单、分级上报，生成客诉工单 |
| 人工兜底 | 置信度不足或高风险请求转人工，B 端审批队列支持通过/驳回并回写会话 |
| 运营后台 | Agent 配置热重载、路由规则、审计日志、A/B 实验、Token 用量 |

## 二、架构

```
浏览器 (/app 门店端, /admin 运营后台)
        │  SSE
        ▼
FastAPI (:8100)
  ├─ 鉴权：JWT，thread_id 命名空间 {user_id}__{thread}
  ├─ 意图识别：规则正则 + LLM function calling 兜底
  └─ LangGraph 状态机
        router ──► memory ──► supervisor
                                   │  （按意图分发，可并行）
        ┌──────────┬──────────┬────┴─────┬──────────┬──────────┐
     policy     store      supply    settlement  material    staff …
        └──────────┴──────────┴────┬─────┴──────────┴──────────┘
                                   ▼
                              synthesis ──► evaluator ──► decider
                                                              │
                                            置信度低 / 高风险 ──┴──► 人工审批队列
```

- **状态**：`AgentState` 显式保存 `store_data / loss_data / supply_data / settlement_data / material_data / staff_data / retrieval_result / audit_*` 等槽位；每个 Agent 只允许写自己的键（`_AGENT_ALLOWED_KEYS`）。
- **工具作用域**：每个 Agent 只能调用授权工具（`_AGENT_TOOL_SCOPES`），例如 `store_agent` 只有 `store / submit_loss_report / check_loss_status`。
- **持久化**：PostgreSQL（业务数据 + 审计）、Redis（LangGraph checkpointer / 工单草稿 / 缓存，DB 0）、Qdrant（`store_knowledge` 制度库、`store_catalog` 商品库、`conversation_memory` 记忆）。

完整的请求生命周期（接入 → 意图/多意图 → 记忆 → Supervisor 并行/串行 → 专家 Agent → 合成 → 评估/决策 → SSE/Celery → 报损审批）见 [门店服务台端到端流程](./docs/explanation/architecture/system-flows/store-service-desk-end-to-end.md)；本轮修复与测试治理记录见 [二次开发交付总结](./docs/explanation/architecture/store-service-desk-delivery-summary.md)。

### 电商 → 门店 领域映射

| 原电商域 | 门店服务台域 | 对应实现 |
|---|---|---|
| 订单查询 | 门店/设备查询 | `store_tool.py` |
| 退款/售后 | **报损申请与分级审批** | `loss_service.py` + `store.py` |
| 物流跟踪 | 供货配送与到货 | `supply_tool.py` |
| 支付/账单 | 日结与对账 | `settlement_tool.py` |
| 购物车 | 物料申领 | `material_tool.py` |
| 账户/会员 | 员工档案与排班 | `staff_tool.py` |
| 商品咨询 | 商品配方/保质期/过敏原 | `product_tool.py` |
| 投诉 | 食品安全与客诉 | `complaint_tool.py` |
| 活动政策 RAG | 门店制度 RAG | `data/knowledge/*.md` |

## 三、快速开始

前置依赖：PostgreSQL（本机 `D:\pgsql16`，5432）、Redis（6379）、Qdrant（6335）、TEI embedding（8081）。数据库名 `store_service_desk`。

```bash
cd /d/store-service-desk

# 1) 依赖（已有 .venv）
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2) 环境变量：复制模板后填写，切勿提交 .env
cp .env.example .env

# 3) 建表 + 种子数据
.venv/Scripts/python.exe scripts/init_db.py
.venv/Scripts/python.exe scripts/seed_data.py

# 4) 制度库与商品库入库（需 TEI / Qdrant 已启动）
.venv/Scripts/python.exe scripts/etl_qdrant.py
.venv/Scripts/python.exe scripts/seed_product_catalog.py

# 5) 启动后端（Windows Git Bash）
PYTHONUTF8=1 PYTHONPATH=. .venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8100

# 6) 启动 Celery（记忆抽取、审计通知、知识同步、可观测入库）
PYTHONUTF8=1 PYTHONPATH=. .venv/Scripts/python.exe -m celery -A app.celery_app worker \
  --loglevel=info --concurrency=2 --pool=solo -Q store_service_desk

# 7) 前端（产物由 FastAPI 直接托管）
cd frontend && pnpm install && pnpm run build
```

访问：门店端 <http://127.0.0.1:8100/app> ｜ 运营后台 <http://127.0.0.1:8100/admin>

### 演示账号

| 账号 | 密码 | 角色 |
|---|---|---|
| `store1001` | `store123` | 上海张江店 |
| `store1002` | `store123` | 北京中关村店 |
| `store1003` / `store1004` | `store123` | 其余两家门店 |
| `admin` | `admin123` | 运营后台（审批） |

> 演示密码仅用于本地环境，部署前必须修改。

### 5 分钟演示脚本

1. `store1001` 登录 → 问"张江店今天设备都正常吗" → 门店 + 故障设备清单
2. 问"我店里打碎了一箱牛奶，报损 350 元" → 自动落 ST1001、生成报损单、状态"待店长审批"
3. 问"报损 2500 元怎么办" → 系统提示区域督导审批，并提交后进入审批队列
4. 问"我的报损审批到哪了" → 列出最近报损单与状态
5. `admin` 登录后台 → 待办审批 → 通过 → 回问第 4 步，状态变为"已通过（区域督导审批）"
6. 问"报损金额多少需要区域督导审批" → 制度原文 + `[来源: OPS-LOSS-2026]` + 置信度 ≥ 0.8

## 四、目录结构

```
app/
  agents/        专家 Agent（policy/store/supply/settlement/material/staff/product/complaint）
  graph/         LangGraph 节点、状态机、边
  intent/        意图分类（规则 + LLM）
  tools/         领域工具与工具注册表
  services/      业务服务（loss_service 报损、admin_service 审批、knowledge 等）
  retrieval/     混合检索 + rerank
  memory/        向量记忆与上下文预算
  confidence/    置信度信号融合
  safety/        内容安全与 PII
  models/        SQLModel 表与 AgentState
  api/           REST/SSE 接口
  tasks/         Celery 任务
data/
  knowledge/     门店制度库（6 篇）
  store_mock.json 门店与设备模拟数据
  products.json  商品与配方
frontend/        React + Vite（customer / admin 两套入口）
scripts/         init_db / seed_data / etl_qdrant / seed_product_catalog / clear_user_memory
data/golden_store_v1.jsonl  门店域评测集（25 条，覆盖 9 类意图）
```

评测：

```bash
.venv/Scripts/python.exe scripts/run_evaluation.py      # 使用 data/golden_store_v1.jsonl
.venv/Scripts/python.exe -m app.evaluation.run --dataset data/golden_store_v1.jsonl
```

清理某个测试账号被污染的记忆（向量 + 结构化事实 + 摘要）：

```bash
.venv/Scripts/python.exe scripts/clear_user_memory.py --user-id 1 --dry-run
.venv/Scripts/python.exe scripts/clear_user_memory.py --user-id 1 --yes
```

## 五、安全与合规

- `.env`、密钥、数据库口令**不得提交到仓库**，也不得出现在文档与日志中；对外演示请使用脱敏数据。
- 涉及金额的决策（报损分级、物料超额）全部在代码中判定并留痕，模型只负责抽取与解释。
- 员工与门店信息属于内部数据，生产部署需接入企业 SSO 并开启审计日志留存。

## 六、常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `Cannot create index on db != 0` | Redis checkpointer 只支持 DB 0，检查 `REDIS_DB` |
| 启动报 `No module named uvicorn` | 用了系统 Python，改用项目 `.venv/Scripts/python.exe` |
| 提问后长时间无响应 | 检查 TEI(8081)/Qdrant(6335) 是否在线，`/health` 三项是否 connected |
| 重复问同一句答案不变 | 同一 `thread_id` 会命中检查点，换新 thread 或清理 Redis 键 |
| Windows 中文乱码 | 终端与进程都设 `PYTHONUTF8=1` |

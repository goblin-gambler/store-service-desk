# 连锁零售门店服务台（Store Service Desk）

面向连锁零售门店的 AI 服务台。门店员工用自然语言提问，系统完成意图识别、专家 Agent 调度、业务数据查询与制度检索；涉及报损、物料超额、食品安全等高风险动作时，按金额与类型进入人工审批队列，决策全程留痕。

- 后端：FastAPI + SQLModel + PostgreSQL + Redis Stack + Qdrant
- 编排：LangGraph 状态机 + LangChain，多 Agent 串行/并行调度
- 模型：OpenAI 兼容接口（默认 Qwen），Embedding 1024 维，Qwen Reranker
- 异步：Celery Worker + Beat
- 前端：React 19 + TypeScript + Vite + Tailwind + Radix UI
- 观测：Prometheus + Grafana + Loki + Tempo + Mimir + OpenTelemetry

---

## 一、功能与实现

| 能力 | 实现要点 |
|---|---|
| 门店与设备查询 | `store` 工具读取 `data/store_mock.json`；支持门店编号 `ST\d{4}`、城市/店名指代、设备在线率与故障清单 |
| 设备报修工单 | 工单号 `RP{YYYYMMDD}{4位hex}`；Redis 保存 `repair_ticket:{sn}`（TTL 30 天）与门店工单列表，支持多轮查询进度 |
| **报损分级审批** | 金额阈值写死在 `LossService.resolve_approval_level`：`< 200` 自动核销，`200–2000` 店长审批，`> 2000` 区域督导审批，`> 10 万` 转线下专项；非自动单写入 `audit_logs` 并投递 `store.notify_admin_audit` |
| 供货与到货 | 配送单 `DL\d{8,}` 查询、到货差异、冷链温度异常 |
| 日结与对账 | 按日期查询应收/实收/差额；差额超阈值提示督导介入 |
| 物料申领 | `QUERY / ADD / REMOVE / MODIFY` 四类动作，按单价自动估价，月度额度 `5000 元`，草稿存 Redis `material_request:{user_id}`（TTL 24 小时），超额拦截 |
| 员工与排班 | 员工号 `E\d{4}`；排班、请假、加班规则；涉及总部权限的操作显式提示审批 |
| 制度问答（RAG） | 混合检索 + RRF + Rerank + Self-RAG 文档评分 + 引用校验，答案强制携带 `[来源: X]` |
| 商品与配方 | Qdrant `store_catalog` 向量检索：价格、配方、规格、保质期、储存条件、过敏原、在售状态 |
| 食品安全与客诉 | 规则快速分类（食品安全/环境/供货/服务/其他）→ LLM 兜底 → 写入 `complaint_tickets`；高风险自动升级 |
| 人工兜底 | 置信度不足、高风险、重复重试均转人工；B 端审批队列支持通过/驳回并回写会话 |
| 运营后台 | Agent 配置热更新与版本回滚、路由规则、A/B 实验、知识库同步、审计日志、Token 用量、质量看板 |

### 报损审批链路

```
用户: "打碎一箱牛奶，报损 350 元"
  │
  ├─ 金额抽取 extract_amount()       → 350.0
  ├─ 原因分类 classify_loss_reason() → DAMAGE（破损）
  ├─ 分级判定 resolve_approval_level → STORE_MANAGER（200–2000）
  ├─ 建单 LossReport(status=PENDING)
  ├─ 写审计 AuditLog(trigger_type=RISK, risk_level=MEDIUM)
  └─ Celery store.notify_admin_audit
        │
        ▼
运营后台 /admin → process_admin_decision()
  ├─ SELECT ... FOR UPDATE 锁审计单，防并发重复审批
  ├─ 更新 LossReport(status=APPROVED/REJECTED)
  ├─ 写 MessageCard(AUDIT_CARD)
  └─ WebSocket 推送状态变更到门店会话
```

金额与层级完全由代码判定，模型只负责从自然语言中抽取金额、原因和门店，不参与审批决策。

---

## 二、整体架构

```
浏览器：门店端 /app  ·  运营后台 /admin
        │  SSE / WebSocket / REST
        ▼
FastAPI（app/main.py）
  ├─ 中间件：Correlation-ID、Trace-ID、CORS、SlowAPI 限流、入站 PII 过滤
  ├─ /api/v1/chat        → LangGraph 状态机（SSE 流式）
  ├─ /api/v1/admin/*     → 审批、Agent 配置、知识库、实验、告警、指标
  └─ /api/v1/ws/*        → 会话与管理员实时推送
        │
        ▼
LangGraph
  router ─► memory ─► supervisor ─┬─► policy        ┐
                                  ├─► store         │ 串行或并行
                                  ├─► supply        │
                                  ├─► settlement    │
                                  ├─► material      │
                                  ├─► staff         │
                                  ├─► product       │
                                  └─► complaint     ┘
                        │
                        ▼
                  synthesis ─► evaluator ─► decider
                                                │
                          置信度低 / 高风险 ────┴──► 人工审批 / 转人工
        │
        ├─ 工具层 ToolRegistry：门店、报损、供货、结算、物料、员工、商品、客诉
        ├─ RAG：store_knowledge（制度）、store_catalog（商品）
        ├─ Memory：conversation_memory（向量）+ 结构化事实/摘要
        └─ 存储：PostgreSQL · Redis Stack · Qdrant
        │
        ▼
Celery Worker + Beat（记忆抽取、审计通知、知识同步、告警、评测、影子流量）
```

### 请求生命周期

1. **接入**：`POST /api/v1/chat` 校验 JWT，按用户与全局限流；入站 PII 过滤；生成/透传 `X-Correlation-ID`、`X-Trace-ID`。
2. **意图**：规则正则优先（命中置信度 0.8），未命中走 LLM function calling（阈值 0.7，5 秒超时），结果缓存 300 秒；多意图拆分为主/次意图，槽位不足进入澄清。
3. **记忆**：`memory` 节点在单个会话内顺序读取用户画像、偏好、事实和摘要，并行检索 Qdrant 历史消息；按 Token 预算裁剪后注入上下文。
4. **调度**：`supervisor` 根据意图映射与依赖关系决定串行或并行；并行时用 LangGraph `Send` 分发，每个分支只拿到允许的 State 切片。
5. **专家执行**：Agent 只能调用授权工具，结果写回自己的状态槽位，不污染其他 Agent。
6. **合成**：`synthesis` 合并本轮子答案；多答案时由 LLM 统一口径并做输出审核。
7. **评估**：`evaluator` 融合检索、LLM、情绪信号得到置信度；低于重试阈值回到 router，否则交给 `decider`。
8. **决策**：确定自动回复 / 人工审批 / 转人工；写入历史、触发记忆抽取、记录可观测数据。
9. **异步**：会话结束后 Celery 写入观测日志、抽取事实、同步向量记忆；影子流量按采样率旁路执行。

---

## 三、LangGraph 编排实现

`app/graph/workflow.py` 使用 `StateGraph(AgentState)` 构建，节点通过返回 `Command(goto=...)` 决定下一跳，不依赖集中式条件边。

### 状态与隔离

- `AgentState` 显式定义业务槽位：`store_data`、`loss_data`、`supply_data`、`settlement_data`、`material_data`、`staff_data`、`product_data`、`retrieval_result`、`audit_*`、`confidence_*`、`sub_answers` 等。
- 每个 Agent 有白名单 `_AGENT_ALLOWED_KEYS`，子图入口过滤 State，出口只合并白名单内的键；观测结果统一掩码后再写入 `sub_answers`。
- 工具作用域 `_AGENT_TOOL_SCOPES` 独立于状态白名单，双重限制 Agent 能力边界。

| Agent | 意图 | 授权工具 |
|---|---|---|
| `policy_agent` | `POLICY` / `OTHER` | `retrieve_knowledge` |
| `store_agent` | `STORE` / `LOSS` | `store`、`submit_loss_report`、`check_loss_status` |
| `supply_agent` | `SUPPLY` | `supply` |
| `settlement_agent` | `SETTLEMENT` | `settlement` |
| `material_agent` | `MATERIAL` | `material` |
| `staff_agent` | `STAFF` | `staff` |
| `product` | `PRODUCT` | `product` |
| `complaint` | `COMPLAINT` | `create_ticket`、`escalate_ticket`、`get_ticket_status` |

### 并行与串行

`app/graph/parallel.py` 的 `plan_dispatch()` 检查剩余 Agent 两两是否独立：全部独立则返回 `parallel` 并构造 `Send` 分支，否则按队列串行。独立组合例如 `STORE/POLICY`、`SUPPLY/POLICY`、`STORE/PRODUCT`；存在依赖的组合（如 `LOSS/STORE`、`MATERIAL/SUPPLY`）强制串行，保证前序结果可被后序使用。

### 检查点

`app/graph/checkpointer.py` 在 `AsyncRedisSaver` 之上做压缩层：

- TTL 30 天；每 10 次写入保留一个完整快照，其余写 zlib 差分。
- 键结构：`ckpt_opt:*`、`ckpt_index:{thread}:{ns}`、`checkpoint_latest_opt:*`。
- 读取以基础 saver 为准，优化层仅用于加速重建；支持 `aprune(keep_latest)` 清理。

---

## 四、意图识别

`app/intent/` 采用「规则 + LLM」混合策略：

- 意图枚举：`STORE`、`LOSS`、`POLICY`、`SUPPLY`、`SETTLEMENT`、`PRODUCT`、`MATERIAL`、`COMPLAINT`、`STAFF`、`OTHER`。
- 规则命中直接返回，置信度 0.8；未命中调用 function calling，低于 0.5 触发澄清。
- 多意图：按「顺便 / 还有 / 另外 / 以及 / 然后」等分隔，最多 2 个意图；结合独立/依赖矩阵判断并行可行性，可选 LLM 复核并落 `multi_intent_decision_logs`。
- 话题切换：显式切换词、置信度骤降或意图不兼容时重置上下文。
- 槽位校验：P0 槽位缺失才阻断；澄清最多 3 轮，支持拒绝、可选、推断、跳过、升级五种降级路径。
- 安全过滤：敏感词、Prompt 注入、对抗样本、代码片段规则层 + LLM 语义层，命中直接拒绝并记录指标。

---

## 五、RAG 实现

### 检索管线

```
query
 ├─ 多查询改写（可选，N=3，Redis 缓存 1h）
 ├─ dense 检索 top 15（text-embedding-v3 / 1024 维 / cosine）
 ├─ sparse 检索 top 15（fastembed BM25，IDF）
 ├─ RRF 融合（k=60）→ top 10
 ├─ Qwen Reranker（/reranks）→ top 5
 └─ Redis 检索缓存（TTL 600s）
```

`app/retrieval/client.py` 建集合时使用命名向量：dense `text-embedding-v3`（1024，cosine）+ sparse `Modifier.IDF`。Reranker 不可用时由 `create_retriever` 自动降级。

### Self-RAG 与引用校验

`PolicyAgent` 在检索后执行三步：

1. 文档相关性评分，阈值 `0.3`，过滤无关片段。
2. 检索充分性判断，阈值 `0.6`，不足则改写/多查询补检。
3. 生成后并行做引用校验与自我反思；答案必须包含至少一条 `[来源: X]`，否则标记幻觉风险。

制度库为 `data/knowledge/*.md` 六篇：报损管理、排班请假、物料申领、设备报修、食品安全与客诉、日结对账。入库脚本 `scripts/etl_qdrant.py` 完成加载、切分、dense/sparse 向量化与批量写入。

---

## 六、记忆系统

`app/memory/` 分三层：

| 层 | 存储 | 内容 |
|---|---|---|
| 结构化记忆 | PostgreSQL | `user_profiles`、`user_preferences`、`user_facts`、`interaction_summaries` |
| 向量记忆 | Qdrant `conversation_memory` | 历史消息与摘要，payload 含 `user_id/thread_id/message_role/intent/timestamp` |
| 会话状态 | Redis Checkpoint | LangGraph 全量状态，按 `{user_id}__{thread}` 命名空间隔离 |

- 检索按 `user_id + message_role` 过滤，相似度阈值 0.5，去重后取 top 10。
- Token 预算 `MemoryTokenBudget` 默认 2048，优先级：用户画像 > 偏好 > 结构化事实 > 历史摘要 > 相关历史消息。
- 压缩条件：历史超过 20 轮、上下文利用率超过 0.75，或会话自然结束；压缩后用一条 `[会话摘要]` 替换历史并保留 `original_turn_count`。
- 事实抽取异步执行，跳过 PII，置信度低于 0.7 丢弃。

---

## 七、置信度与安全

### 置信度融合

`app/confidence/evaluator.py`：

```
overall = rag_score × 0.3 + llm_score × 0.5 + emotion_score × 0.2
```

- 情绪信号：负面词 ≥ 3 或紧急词 ≥ 2 显著降分；正面词加分；无信号取 0.7。
- RAG 信号：相似度映射后按 `max × 0.4 + avg × 0.3 + coverage × 0.3` 计算。
- 转人工阈值 0.7；低于 0.3 且未超过最大重试次数时回到 router 重试。
- 审计等级：`≥ 0.8` 放行、`≥ 0.5` 自动处理、`< 0.5` 人工。

### 安全链路

- 入站：Unicode/零宽字符清理、注入标记剔除、敏感词与对抗模式检测、超长截断。
- 出站：四层审核（规则 → 模式 → 语义 → LLM Judge）。风险分取各层最大值，每个非零层再累加 0.05，封顶 1.0；高风险直接替换为安全提示。
- PII：信用卡（Luhn 校验）、手机号、身份证（校验位）、护照、邮箱、密码、银行卡等正则识别；日志只记录类型与数量，不落原文；保留期 90 天，支持按用户删除向量与结构化记忆。
- 观测掩码：Agent 间传递的观测结果超过 500 字符时压缩为摘要 + 引用 ID。

---

## 八、人工兜底与审批

- `decider` 判定需要人工时写入 `AuditLog`，通过 WebSocket 推送到管理员房间 `admins`。
- 审批接口 `POST /api/v1/admin/resume/{audit_log_id}` 使用行锁更新审计单，联动更新报损单，写入 `MessageCard` 并回推会话。
- 风险分 ≥ 0.5 的会话自动创建 `review_tickets`，带 SLA 截止时间、风险因子、坐席分配与准确率统计。
- 所有金额与权限判定均在服务层完成，模型输出不直接改变业务状态。

---

## 九、API 一览

统一前缀 `/api/v1`，认证使用 JWT Bearer。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | 主对话接口，SSE 流式返回；60/min 全局 + 10/min/用户限流 |
| POST | `/feedback` | 消息反馈（赞/踩、分类、评论），进入在线评测 |
| POST | `/login` / `/register` | 登录 / 注册，5/min |
| GET | `/me` | 当前用户信息 |
| GET | `/status/{thread_id}` | 会话处理状态 |
| POST | `/metrics/web-vitals` | 前端性能指标上报，载荷 ≤ 10 KB |
| WS | `/ws/{thread_id}` | 会话实时消息；JWT 通过查询参数或 Header 传递 |
| WS | `/ws/admin/{admin_id}` | 管理员通知、审批与状态推送 |
| GET | `/health` | PostgreSQL / Redis / Qdrant 健康检查 |

管理端还包括：

- 审批：`/admin/tasks`、`/admin/tasks-all`、`/admin/confidence-tasks`、`/admin/resume/{id}`
- Agent：`/admin/agents/*` 配置、版本、回滚、审计、Prompt 实验、路由规则
- 知识库：`/admin/knowledge` 增删查、`/admin/knowledge/{id}/sync`、同步状态
- 质量：`/admin/evaluation/*`、`/admin/feedback`、`/admin/review-queue`、`/admin/complaints`
- 运营：`/admin/experiments`、`/admin/analytics`、`/admin/metrics/*`、`/admin/token-usage`、`/admin/alerts/*`

流式事件包含 `metadata`、`confidence`、`trace_id` 与 `[DONE]` 结束标记；前端只渲染 `user_visible` 令牌。

---

## 十、数据与存储

### PostgreSQL

按域分组：

- 业务：`users`、`loss_reports`、`audit_logs`、`complaint_tickets`、`message_cards`、`review_tickets`
- 知识与配置：`knowledge_documents`、`agent_configs`、`agent_config_versions`、`agent_config_audit_logs`、`routing_rules`
- 记忆：`user_profiles`、`user_preferences`、`user_facts`、`interaction_summaries`
- 评测与实验：`message_feedbacks`、`quality_scores`、`shadow_test_results`、`adversarial_test_runs`、`experiments`、`experiment_variants`、`experiment_assignments`、`experiment_metrics`
- 可观测：`graph_execution_logs`、`graph_node_logs`、`supervisor_decisions`、`token_usage_logs`、`pii_audit_logs`、`multi_intent_decision_logs`
- 告警：`alert_rules`、`alert_events`、`alert_notifications`

### Redis

| 键 / 用途 | 说明 |
|---|---|
| LangGraph Checkpoint | 会话状态，Redis DB 0；压缩差分 + 30 天 TTL |
| `repair_ticket:{sn}` | 报修工单详情，TTL 30 天 |
| `repair_tickets:{store_code}` | 门店工单列表 |
| `material_request:{user_id}` | 物料申领草稿，TTL 24 小时 |
| `intent:session:{id}` | 意图会话缓存，TTL 30 分钟 |
| `agent_config:{name}` | Agent 配置缓存，TTL 60 秒；后台修改即失效 |
| 检索 / 记忆缓存 | 检索 600 秒，向量记忆 300 秒 |
| Celery Broker / Backend | 与业务缓存分库配置 |

### Qdrant

| 集合 | 用途 | 向量 |
|---|---|---|
| `store_knowledge` | 门店制度库 | dense 1024 + sparse BM25 |
| `store_catalog` | 商品与配方 | dense 1024，cosine |
| `conversation_memory` | 历史消息与摘要 | dense 1024，cosine，`user_id` 负载索引 |

> `QDRANT_COLLECTION_NAME` 默认 `store_knowledge`，以 `.env` 实际配置为准。

---

## 十一、异步任务

Celery 应用 `store_service_desk`，默认队列同名。任务硬超时 300 秒、软超时 240 秒，prefetch 4。

| 任务 | 说明 |
|---|---|
| `memory.extract_and_save_facts` | 会话后抽取结构化事实 |
| `memory.prune_vector_memory` | 清理过期向量记忆（每日） |
| `store.notify_admin_audit` | 审批通知：写消息卡片（高风险单据） |
| `knowledge.sync_document` | 知识文档解析、切分、向量入库 |
| `observability.log_chat_observability` | 会话后异步落观测数据，避免阻塞 SSE |
| `notifications.*` | 客诉告警、状态通知、质量检查（每 5 分钟） |
| `alerting.evaluate_rules` / `check_service_health` | 告警规则评估（60 秒）/ 服务健康检查（30 秒） |
| `autoheal.*` | Worker 存活、数据库连接池、Redis 过期键巡检与自愈 |
| `checkpoint.cleanup_old_checkpoints` | 清理旧检查点（每日） |
| `prompt_effect.generate_monthly_report` | Prompt 效果月报 |
| `evaluation.run_adversarial_suite` | 对抗样本回归（每日） |
| `shadow.run_shadow_test` | 影子流量抽样执行 |

Windows 下 Worker 必须使用 `--pool=solo`。

---

## 十二、可观测性

应用通过 `app/observability/metrics.py` 暴露 Prometheus 指标，`/metrics` 直接挂载：

- 流量与质量：`chat_requests_total`、`chat_errors_total`、`chat_latency_seconds`、`node_latency_seconds`、`agent_latency_seconds`
- 模型与 RAG：`confidence_score`、`intent_accuracy`、`rag_precision`、`hallucination_rate`、`answer_correctness`
- 成本与上下文：`token_usage_total`、`token_efficiency`、`context_utilization_ratio`、`agent_context_tokens`、`agent_context_reduction_ratio`
- 安全：`safety_blocks_total`、`pii_detections_total`、`injection_attempts_total`、`pii_breaches_total`、`rate_limit_hits_total`
- 基础设施：Redis 连接/命中率、检查点大小/压缩比、Web Vitals
- 人工：`human_transfers_total`

监控栈由 `docker-compose.monitoring.yml` 提供：Prometheus、Grafana、Alertmanager、Loki、Promtail、Tempo、OTel Collector、Mimir。预置告警包括错误率 > 1%、P95 延迟 > 2s、转人工率 > 0.3、意图准确率 < 0.7、幻觉率 > 0.05、RAG 精确率 < 0.6 与服务不可用。

---

## 十三、评测与实验

- 黄金集：`data/golden_store_v1.jsonl`（25 条，覆盖 10 类意图、22 个测试维度）。
- 指标：意图准确率、槽位召回、RAG 精确率、答案正确性、Token 效率、语气一致性、兜底率、延迟分布。
- 对抗集：`data/adversarial_test_cases.jsonl`（57 条），覆盖 Prompt 注入、敏感信息套取、边界条件、意图混淆、有害输入。
- 影子流量：按确定性哈希采样 10%，对比线上与影子版本意图、答案相似度与延迟回归（阈值 500 ms）。
- A/B 实验：按用户哈希分配变体，可覆盖 System Prompt、LLM 模型、检索 top-k、是否启用 Reranker 与记忆上下文配置。
- 回归门禁：基线对比允许最大降幅 5%，超出则评测命令非零退出。

```bash
uv run python scripts/run_evaluation.py
uv run python -m app.evaluation.run --dataset data/golden_store_v1.jsonl
uv run python scripts/adversarial_run.py
```

---

## 十四、前端

两个独立入口，同一套 Vite 构建，产物由 FastAPI 直接托管：

| 入口 | 构建文件 | 路由 |
|---|---|---|
| 门店端 | `index.html` → `frontend/dist/index.html` | `/app` |
| 运营后台 | `admin.html` → `frontend/dist/admin.html` | `/admin` |

- 门店端：登录、单页对话、流式消息、反馈组件、WebSocket 断线重连、Web Vitals 上报。
- 后台：仪表盘、审批任务、知识库、Agent 配置、反馈、指标、实验、会话日志、客诉队列。
- 状态与数据层：Zustand（登录态）+ TanStack Query（服务端状态）+ 自研 `useChat` / `useWebSocket` / `useAuth`。
- 开发代理：`/api` → `http://localhost:8000`，`/grafana` → `http://localhost:3000`。

---

## 十五、快速开始

### 前置依赖

- Python 3.12–3.13
- Node.js 22
- PostgreSQL 16、Redis Stack 7.4、Qdrant 1.16（可直接用 `docker compose`）
- Embedding 服务（1024 维，OpenAI 兼容）与 Rerank 服务（`POST /reranks`）
- 可选：Docker Desktop（Windows 上 Celery 使用 `--pool=solo`）

### 本地启动

```bash
cd /d/store-service-desk

# 1. Python 依赖（uv 读取 pyproject.toml + uv.lock）
uv sync

# 2. 环境变量
cp .env.example .env
#    至少配置：POSTGRES_*、REDIS_*、QDRANT_URL/API_KEY、
#    OPENAI_BASE_URL/OPENAI_API_KEY、LLM_MODEL、
#    EMBEDDING_BASE_URL、RERANK_BASE_URL、SECRET_KEY
#    注意：compose 将 Qdrant 的 6333 映射到宿主 6335，
#          若沿用 compose，请把 QDRANT_URL 改为 http://localhost:6335

# 3. 基础设施
docker compose up -d db redis qdrant

# 4. 建表与种子数据
uv run python scripts/init_db.py
uv run python scripts/seed_data.py

# 5. 制度库与商品库入库
uv run python scripts/etl_qdrant.py
uv run python scripts/seed_product_catalog.py

# 6. 后端（另开终端）
PYTHONUTF8=1 PYTHONPATH=. uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 7. Celery（另开终端，包含 Beat）
PYTHONUTF8=1 PYTHONPATH=. uv run celery -A app.celery_app worker \
  --loglevel=info --concurrency=4 --pool=solo --beat

# 8. 前端构建
cd frontend && npm ci && npm run build
```

也可以直接执行 `./start.sh`：脚本会安装依赖、构建前端、拉起基础设施、执行 Alembic 迁移并启动 API 与 Worker。

访问地址：

- 门店端：<http://localhost:8000/app>
- 运营后台：<http://localhost:8000/admin>
- 健康检查：<http://localhost:8000/health>
- OpenAPI：需设置 `ENABLE_OPENAPI_DOCS=true` 后访问 `/docs`

> 若 8000 被占用，可改为 `--port 8100`，同时把前端 `VITE_API_BASE_URL` 指向新端口后重新构建。

### 演示账号

| 账号 | 密码 | 角色 |
|---|---|---|
| `store1001` | `store123` | 上海张江店 |
| `store1002` | `store123` | 北京中关村店 |
| `store1003` | `store123` | 广州天河店 |
| `store1004` | `store123` | 深圳福田店 |
| `admin` | `admin123` | 运营后台 / 审批 |

演示口令仅用于本地环境，部署前必须更换。

### 演示路径

1. `store1001` 登录，问「张江店今天设备都正常吗」→ 门店概况 + 故障设备清单。
2. 问「打碎了一箱牛奶，报损 350 元」→ 建单、归到 ST1001、状态「待店长审批」。
3. 问「报损 2500 元怎么办」→ 提示区域督导审批，提交后进入审批队列。
4. 问「我的报损审批到哪了」→ 最近报损单与状态。
5. `admin` 登录后台 → 通过审批 → 回到会话查看状态变更。
6. 问「报损金额多少需要区域督导审批」→ 制度原文 + `[来源: ...]` + 置信度。

---

## 十六、目录结构

```
app/
  agents/        router / supervisor / policy / store / supply / settlement /
                 material / staff / product / complaint / evaluator / config_loader
  graph/         workflow、nodes、parallel、subgraphs、checkpointer
  intent/        classifier、multi_intent、clarification、slot_validator、
                 topic_switch、safety、service
  tools/         store / supply / settlement / material / staff / product /
                 complaint / registry
  services/      loss_service、admin_service、auth_service、status_service、
                 online_eval、continuous_improvement、experiment、review_queue、alert_service
  retrieval/     retriever、client、reranker、rewriter、embeddings、sparse_embedder
  memory/        vector_manager、structured_manager、summarizer、compactor、extractor
  confidence/    signals、evaluator
  safety/        rules、patterns、output_moderator、llm_judge、embeddings
  context/       masking、pii_filter、token_budget
  models/        SQLModel 表与 AgentState
  api/v1/        chat、auth、status、websocket、web_vitals、admin/*
  tasks/         Celery 任务（audit、knowledge、memory、alert、autoheal、shadow 等）
  observability/ Prometheus 指标、OTel、执行/延迟/Token 记录
  evaluation/    数据集、指标、对抗、影子、基线、CLI
  main.py        FastAPI 入口、生命周期、静态站点托管
data/
  knowledge/     门店制度库（6 篇 Markdown）
  store_mock.json  4 门店 / 8 设备 / 配送 / 日结 / 员工
  products.json    商品与配方
  prompt_examples/ few-shot 示例
  golden_store_v1.jsonl / adversarial_test_cases.jsonl
frontend/        React + Vite，customer 与 admin 双入口
scripts/         init_db / seed_data / etl_qdrant / seed_product_catalog /
                 run_evaluation / adversarial_run / clear_user_memory 等
migrations/      Alembic 迁移
tests/           pytest 测试（单元、集成、性能、监控配置）
grafana/ prometheus/ loki/ tempo/ mimir/ otel/ alertmanager/  监控栈配置
```

---

## 十七、关键配置

| 变量 | 默认 / 说明 |
|---|---|
| `LLM_MODEL` | `qwen-plus` |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | OpenAI 兼容模型网关 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `text-embedding-v3` / `1024` |
| `QDRANT_URL` / `QDRANT_COLLECTION_NAME` | 代码默认 `localhost:6333` / `store_knowledge`；本地 compose 需用宿主端口 6335，集合名以 `.env` 为准 |
| `RETRIEVER_DENSE_TOPK` / `SPARSE_TOPK` | 15 / 15 |
| `RETRIEVER_RRF_K` / `FINAL_TOPK` | 60 / 5 |
| `CONFIDENCE__THRESHOLD` | 0.7（转人工） |
| `CONFIDENCE__RAG_WEIGHT` / `LLM_WEIGHT` / `EMOTION_WEIGHT` | 0.3 / 0.5 / 0.2 |
| `LOSS_AUTO_APPROVE_AMOUNT` | 200 元 |
| `LOSS_REGIONAL_APPROVAL_AMOUNT` | 2000 元 |
| `MAX_ROUTER_ITERATIONS` / `MAX_EVALUATOR_RETRIES` | 5 / 3 |
| `CONFIDENCE_RETRY_THRESHOLD` / `FUNCTION_CALLING_THRESHOLD` | 0.3 / 0.7 |
| `MEMORY_CONTEXT_TOKEN_BUDGET` | 2048 |
| `COMPACTION_THRESHOLD` / `VECTOR_MEMORY_SCORE_THRESHOLD` | 0.75 / 0.5 |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | Redis 地址，建议与业务缓存分库 |
| `CORS_ORIGINS` | JSON 数组，禁止 `*` |
| `SECRET_KEY` | JWT 签名密钥，必须修改 |

完整变量清单见 `.env.example`。

---

## 十八、测试与代码质量

```bash
# 后端（用 python -m 确保项目根目录在 sys.path）
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .

# 前端
cd frontend
npm run test          # Vitest
npm run lint
npx playwright test   # E2E：门店对话、管理后台
```

测试覆盖图编排、上下文隔离、意图分类、多意图、槽位与澄清、检索与重排、记忆压缩、置信度信号、PII 掩码、输出审核、审批服务、WebSocket、限流、可观测性与监控配置。

---

## 十九、常见问题

| 现象 | 处理 |
|---|---|
| `Cannot create index on db != 0` | LangGraph Redis Checkpointer 只支持 DB 0，检查 `REDIS_DB` |
| 启动报 `No module named uvicorn` | 未使用项目环境，改用 `uv run` 或项目 `.venv` |
| 提问长时间无响应 | 检查 `/health` 中 PostgreSQL、Redis、Qdrant 是否正常，以及 Embedding/Rerank 服务 |
| 同一 `thread_id` 答案不变 | 命中了检查点；新建会话或清理对应 Redis 键 |
| 知识问答没有引用 | 确认 `scripts/etl_qdrant.py` 已执行、Qdrant 集合有数据 |
| Windows 中文乱码 | 进程与终端设置 `PYTHONUTF8=1` |
| Celery 在 Windows 起不来 | 使用 `--pool=solo`，不要用默认 prefork |
| 前端 404 / 白屏 | 先 `npm run build`，确认 `frontend/dist` 存在且由 FastAPI 托管 |
"""意图识别配置（连锁零售门店服务台）

定义意图兼容矩阵、三级意图约束、槽位优先级。
槽位缺失默认交给专家 Agent 追问，避免 Router 反复澄清打断多轮流程。
"""

from app.intent.models import IntentAction, IntentCategory

# ============== 意图兼容性矩阵 ==============
INTENT_COMPATIBILITY: dict[str, list[str]] = {
    "LOSS/APPLY": ["LOSS/QUERY", "STORE/APPLY", "POLICY/CONSULT"],
    "LOSS/QUERY": ["LOSS/APPLY", "SETTLEMENT/QUERY"],
    "STORE/QUERY": ["STORE/APPLY", "SUPPLY/QUERY"],
    "STORE/APPLY": ["STORE/QUERY", "LOSS/APPLY"],
    "SUPPLY/QUERY": ["STORE/QUERY", "COMPLAINT/QUERY"],
    "SETTLEMENT/QUERY": ["SETTLEMENT/CONSULT"],
    "PRODUCT/QUERY": ["PRODUCT/COMPARE"],
    "PRODUCT/COMPARE": ["PRODUCT/QUERY"],
    "MATERIAL/QUERY": ["MATERIAL/ADD", "MATERIAL/MODIFY"],
    "MATERIAL/ADD": ["MATERIAL/QUERY"],
    "STAFF/QUERY": ["STAFF/APPLY", "POLICY/CONSULT"],
    "STAFF/APPLY": ["STAFF/QUERY"],
    "COMPLAINT/QUERY": ["COMPLAINT/CONSULT"],
    "POLICY/CONSULT": ["LOSS/APPLY", "STAFF/QUERY"],
}


# ============== 三级意图配置 ==============
TERTIARY_INTENT_CONFIG: dict[tuple[str, str], dict] = {
    ("LOSS", "APPLY"): {
        "tertiary_intents": ["LOSS_DAMAGE", "LOSS_EXPIRED", "LOSS_EQUIPMENT"],
        "description": "提交报损",
    },
    ("LOSS", "QUERY"): {
        "tertiary_intents": ["LOSS_STATUS", "LOSS_HISTORY"],
        "description": "报损查询",
    },
    ("STORE", "QUERY"): {
        "tertiary_intents": ["STORE_INFO", "DEVICE_STATUS"],
        "description": "门店/设备查询",
    },
    ("STORE", "APPLY"): {
        "tertiary_intents": ["REPAIR_CREATE", "REPAIR_STATUS"],
        "description": "设备报修",
    },
    ("POLICY", "CONSULT"): {
        "tertiary_intents": [
            "POLICY_SCHEDULE",
            "POLICY_LEAVE",
            "POLICY_LOSS",
            "POLICY_MATERIAL",
        ],
        "description": "制度咨询",
    },
    ("SUPPLY", "QUERY"): {
        "tertiary_intents": ["DELIVERY_STATUS", "COLD_CHAIN_ALERT", "REPLENISH_ETA"],
        "description": "配送查询",
    },
    ("SETTLEMENT", "QUERY"): {
        "tertiary_intents": ["SETTLEMENT_DAILY", "SETTLEMENT_DIFF", "INVOICE_STATUS"],
        "description": "结算查询",
    },
    ("PRODUCT", "QUERY"): {
        "tertiary_intents": [
            "PRODUCT_RECIPE",
            "PRODUCT_SHELF_LIFE",
            "PRODUCT_STATUS",
            "PRODUCT_PRICE",
        ],
        "description": "商品查询",
    },
    ("PRODUCT", "COMPARE"): {
        "tertiary_intents": ["PRODUCT_SPEC_COMPARE", "PRODUCT_PRICE_COMPARE"],
        "description": "商品比较",
    },
    ("MATERIAL", "QUERY"): {
        "tertiary_intents": ["MATERIAL_VIEW", "MATERIAL_QUOTA"],
        "description": "申领单查询",
    },
    ("MATERIAL", "ADD"): {
        "tertiary_intents": ["MATERIAL_ADD_ITEM", "MATERIAL_ADD_BULK"],
        "description": "添加申领",
    },
    ("MATERIAL", "REMOVE"): {
        "tertiary_intents": ["MATERIAL_REMOVE_ITEM", "MATERIAL_CLEAR"],
        "description": "移除申领",
    },
    ("STAFF", "QUERY"): {
        "tertiary_intents": ["STAFF_PROFILE", "STAFF_PERMISSION", "STAFF_TRAINING"],
        "description": "员工查询",
    },
    ("STAFF", "APPLY"): {
        "tertiary_intents": ["PERMISSION_APPLY", "ACCOUNT_UNLOCK"],
        "description": "权限申请",
    },
    ("COMPLAINT", "QUERY"): {
        "tertiary_intents": [
            "COMPLAINT_FOOD_SAFETY",
            "COMPLAINT_SERVICE",
            "COMPLAINT_ENVIRONMENT",
        ],
        "description": "投诉受理",
    },
}


def validate_tertiary_intent(
    primary: IntentCategory,
    secondary: IntentAction,
    tertiary: str | None,
) -> bool:
    """验证三级意图是否合法"""
    if tertiary is None:
        return True

    key = (primary.value, secondary.value)
    if key not in TERTIARY_INTENT_CONFIG:
        return False

    allowed = TERTIARY_INTENT_CONFIG[key]["tertiary_intents"]
    return tertiary in allowed


# ============== 槽位优先级配置 ==============
# P0 留空：缺槽位由 Agent 多轮追问，避免 Router 提前澄清打断报损/报修流程。
SLOT_PRIORITY_CONFIG: dict[str, dict[str, dict[str, list[str]]]] = {
    "LOSS": {
        "APPLY": {
            "P0": [],
            "P1": ["store_code", "amount", "reason_category"],
            "P2": ["evidence", "device_code"],
        },
        "QUERY": {"P0": [], "P1": ["report_sn"], "P2": []},
    },
    "STORE": {
        "QUERY": {"P0": [], "P1": ["store_code", "device_type"], "P2": []},
        "APPLY": {
            "P0": [],
            "P1": ["store_code", "device_type", "issue"],
            "P2": ["urgency"],
        },
    },
    "SUPPLY": {
        "QUERY": {"P0": [], "P1": ["store_code", "delivery_sn"], "P2": []},
    },
    "SETTLEMENT": {
        "QUERY": {"P0": [], "P1": ["store_code", "settle_date"], "P2": []},
    },
    "PRODUCT": {
        "QUERY": {
            "P0": [],
            "P1": ["product_name", "specification"],
            "P2": ["price_range"],
        },
        "COMPARE": {
            "P0": [],
            "P1": ["product_names", "compare_aspect"],
            "P2": [],
        },
    },
    "MATERIAL": {
        "QUERY": {"P0": [], "P1": [], "P2": []},
        "ADD": {"P0": [], "P1": ["material_name", "quantity"], "P2": []},
        "REMOVE": {"P0": [], "P1": ["material_name"], "P2": []},
    },
    "STAFF": {
        "QUERY": {"P0": [], "P1": ["employee_no"], "P2": []},
        "APPLY": {"P0": [], "P1": ["employee_no", "permission"], "P2": []},
    },
    "POLICY": {
        "CONSULT": {"P0": [], "P1": ["policy_topic"], "P2": []},
    },
    "COMPLAINT": {
        "QUERY": {"P0": [], "P1": ["incident_type", "store_code"], "P2": []},
    },
}


def check_intent_compatibility(
    current_intent: str,
    new_intent: str,
) -> bool:
    """检查两个意图之间是否可以平滑切换"""
    if current_intent == new_intent:
        return True

    if current_intent in INTENT_COMPATIBILITY:
        return new_intent in INTENT_COMPATIBILITY[current_intent]

    if new_intent in INTENT_COMPATIBILITY:
        return current_intent in INTENT_COMPATIBILITY[new_intent]

    return False

"""测试意图配置（连锁零售门店服务台域）"""

from app.intent.config import (
    TERTIARY_INTENT_CONFIG,
    check_intent_compatibility,
    validate_tertiary_intent,
)
from app.intent.models import IntentAction, IntentCategory


def test_tertiary_intent_config_exists():
    assert ("LOSS", "APPLY") in TERTIARY_INTENT_CONFIG
    assert ("STORE", "QUERY") in TERTIARY_INTENT_CONFIG


def test_validate_tertiary_intent_valid():
    result = validate_tertiary_intent(
        IntentCategory.LOSS,
        IntentAction.APPLY,
        "LOSS_DAMAGE",
    )
    assert result is True


def test_validate_tertiary_intent_invalid():
    result = validate_tertiary_intent(
        IntentCategory.LOSS,
        IntentAction.APPLY,
        "INVALID_INTENT",
    )
    assert result is False


def test_validate_tertiary_intent_none():
    result = validate_tertiary_intent(
        IntentCategory.LOSS,
        IntentAction.APPLY,
        None,
    )
    assert result is True


def test_check_intent_compatibility_same():
    result = check_intent_compatibility(
        "STORE/QUERY",
        "STORE/QUERY",
    )
    assert result is True


def test_check_intent_compatibility_compatible():
    # 报损申请后可以平滑切换到设备报修（见 INTENT_COMPATIBILITY）
    result = check_intent_compatibility(
        "LOSS/APPLY",
        "STORE/APPLY",
    )
    assert result is True


def test_check_intent_compatibility_incompatible():
    result = check_intent_compatibility(
        "STORE/QUERY",
        "STAFF/APPLY",
    )
    assert result is False

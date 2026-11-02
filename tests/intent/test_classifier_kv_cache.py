"""Tests for KV-Cache optimization in IntentClassifier."""

import pytest

from app.intent.classifier import IntentClassifier
from tests._llm import DeterministicChatModel


@pytest.fixture
def classifier():
    return IntentClassifier(llm=DeterministicChatModel())


def test_create_messages_puts_few_shot_in_human_message(classifier):
    classifier._few_shot_examples = [
        {"query": "ST1001 设备状态", "primary_intent": "STORE", "secondary_intent": "QUERY"},
        {"query": "报损 申请 流程", "primary_intent": "LOSS", "secondary_intent": "APPLY"},
    ]
    messages = classifier._create_messages("ST1001 设备状态 查询")

    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"
    assert "ST1001 设备状态" in messages[1].content
    assert "ST1001 设备状态 查询" in messages[1].content

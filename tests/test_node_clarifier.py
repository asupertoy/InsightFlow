import pytest
from unittest.mock import patch, MagicMock
from insightflow_core.nodes.clarifier import clarifier_node
from insightflow_core.state import AgentState

@patch('insightflow_core.nodes.clarifier.get_model_router')
def test_clarifier_fuzzy_input(mock_get_router):
    """测试模糊输入时生成澄清问题"""
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="1. What is the scope?\n2. What is the deadline?")
    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm
    mock_get_router.return_value = mock_router

    state: AgentState = {
        "original_task": "帮我写个报告",
        "clarification_answers": None,
        "clarified_task": None,
        "clarification_questions": [],
        "metadata": {},
        "plan": [],
        "current_step_index": 0,
        "last_step_success": True,
        "research_findings": [],
        "running_summary": "",
        "raw_data_context": [],
        "code_snippets": [],
        "code_outputs": [],
        "figure_paths": [],
        "draft_report": "",
        "review_comments": "",
        "review_status": "",
        "revision_count": 0,
        "messages": []
    }

    result = clarifier_node(state)

    # 验证返回澄清问题
    assert "clarification_questions" in result
    assert isinstance(result["clarification_questions"], list)
    assert len(result["clarification_questions"]) > 0
    assert "clarified_task" not in result or result.get("clarified_task") is None

@patch('insightflow_core.nodes.clarifier.get_model_router')
def test_clarifier_with_answers(mock_get_router):
    """测试有回答时生成最终任务"""
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Write a report about AI Agent technology")
    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm
    mock_get_router.return_value = mock_router

    state: AgentState = {
        "original_task": "帮我写个报告",
        "clarification_answers": "关于AI Agent的报告",
        "clarified_task": None,
        "clarification_questions": [],
        "metadata": {},
        "plan": [],
        "current_step_index": 0,
        "last_step_success": True,
        "research_findings": [],
        "running_summary": "",
        "raw_data_context": [],
        "code_snippets": [],
        "code_outputs": [],
        "figure_paths": [],
        "draft_report": "",
        "review_comments": "",
        "review_status": "",
        "revision_count": 0,
        "messages": []
    }

    result = clarifier_node(state)

    # 验证返回澄清后的任务
    assert "clarified_task" in result
    assert isinstance(result["clarified_task"], str)
    assert len(result["clarified_task"].strip()) > 0
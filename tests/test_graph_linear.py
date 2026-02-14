import pytest
from unittest.mock import patch, MagicMock
from insightflow_core.graph import create_graph
from insightflow_core.state import AgentState

@pytest.fixture
def mock_graph():
    """创建模拟的图，mock 未实现的节点"""
    with patch('insightflow_core.graph.clarifier_node') as mock_clarifier, \
         patch('insightflow_core.graph.planner_node') as mock_planner, \
         patch('insightflow_core.graph.researcher_node') as mock_researcher, \
         patch('insightflow_core.graph.coder_node') as mock_coder, \
         patch('insightflow_core.graph.writer_node') as mock_writer, \
         patch('insightflow_core.graph.reviewer_node') as mock_reviewer:

        # 设置 mock 节点的返回值
        # Clarifier: Returns clarified task, no questions
        mock_clarifier.return_value = {
            "clarified_task": "搜索 LangGraph 的最新特性并总结",
            "clarification_questions": []
        }
        # Planner: Returns a plan
        mock_planner.return_value = {
            "plan": [{"id": 1, "description": "Step 1", "status": "pending"}],
            "current_step_index": 0
        }
        # Researcher: Returns findings AND updates plan step to completed to avoid infinite loop
        # We need to simulate that the current step is completed
        def list_update(state: AgentState):
            plan = state.get("plan", [])
            idx = state.get("current_step_index", 0)
            if idx < len(plan):
                # Update status of current step
                # Note: Mutable update on the list in state might not propagate if not returned
                # but in LangGraph state updates are merged. 
                # Be careful: we return a diff. We can't easily deep update a list item via diff return unless we return the WHOLE list.
                # So let's return a new plan list with updated status.
                import copy
                new_plan = copy.deepcopy(plan)
                new_plan[idx]["status"] = "completed"
                return {
                    "research_findings": ["Found feature 1", "Found feature 2"],
                    "last_step_success": True,
                    "plan": new_plan,
                    # Move to next step or indicate done
                    "current_step_index": idx + 1
                }
            return {}

        mock_researcher.side_effect = list_update
        
        # Coder (acts as Analyst): Returns nothing special, maybe code output
        mock_coder.return_value = {}
        
        mock_writer.return_value = {"draft_report": "Mock report content"}
        mock_reviewer.return_value = {"review_status": "approve"}

        graph = create_graph()
        return graph

def test_linear_workflow(mock_graph):
    """测试线性流程：澄清 -> 规划 -> 研究 -> 写作 -> 审核 -> 结束"""
    # 初始状态：清晰的任务，不需要澄清
    initial_state: AgentState = {
        "original_task": "搜索 LangGraph 的最新特性并总结",
        "clarified_task": "搜索 LangGraph 的最新特性并总结",  # 假设已澄清
        "clarification_answers": None,
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

    # 运行图
    config = {"configurable": {"thread_id": "test_linear"}}
    result = mock_graph.invoke(initial_state, config=config)

    # 验证最终输出
    assert "draft_report" in result
    assert isinstance(result["draft_report"], str)
    assert len(result["draft_report"]) > 0

    print("Linear workflow test passed")
import pytest
from unittest.mock import patch, MagicMock
from insightflow_core.graph import create_graph
from insightflow_core.state import AgentState

@pytest.fixture
def mock_graph_with_revision():
    """Create a graph where reviewer rejects once then approves."""
    call_count = 0

    def mock_reviewer(state):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: reject
            return {
                "review_status": "reject",
                "review_comments": "Need more details on performance metrics"
            }
        else:
            # Second call: approve
            return {
                "review_status": "approve",
                "review_comments": ""
            }

    with patch('insightflow_core.graph.researcher_node') as mock_researcher,          patch('insightflow_core.graph.coder_node') as mock_coder,          patch('insightflow_core.graph.writer_node') as mock_writer,          patch('insightflow_core.graph.reviewer_node', side_effect=mock_reviewer):

        # Set mock returns
        # Researcher returns findings
        mock_researcher.return_value = {"research_findings": []}
        mock_coder.return_value = {}
        mock_writer.return_value = {"draft_report": "Mock report content"}

        graph = create_graph()
        yield graph

@patch('insightflow_core.nodes.planner.get_model_router')
@patch('insightflow_core.nodes.planner.NoteTool')
def test_revision_workflow(mock_note_tool, mock_get_router, mock_graph_with_revision):
    """Test cycle: Planner -> Researcher -> Writer -> Reviewer(reject) -> Planner -> ..."""
    
    # Mock LLM for Planner
    # Planner is called twice:
    # 1. Initial plan (since plan is empty in initial_state)
    # 2. Re-plan (after rejection)
    
    mock_llm = MagicMock()
    
    # Mock responses for invokes
    # Response 1: Initial plan
    resp1 = MagicMock()
    resp1.content = '{"steps": [{"id": 1, "description": "Step 1", "status": "pending" }]}'
    
    # Response 2: Revised plan
    resp2 = MagicMock()
    resp2.content = '{"steps": [{"id": 1, "description": "Step 1", "status": "completed"}, {"id": 2, "description": "Step 2", "status": "pending" }]}'
    
    # side_effect for multiple calls
    mock_llm.invoke.side_effect = [resp1, resp2]
    
    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm
    mock_get_router.return_value = mock_router
    
    # Mock NoteTool
    mock_note_instance = MagicMock()
    mock_note_tool.return_value = mock_note_instance
    mock_note_instance.get_note.return_value = None

    initial_state = {
        "original_task": "Compare Python and Go",
        "clarified_task": "Compare Python and Go performance", 
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

    config = {"configurable": {"thread_id": "test_revision_thread"}}
    
    # Run graph
    result = mock_graph_with_revision.invoke(initial_state, config=config)
    
    # Verify result
    assert result is not None
    assert "draft_report" in result
    assert result["review_status"] == "approve"
    
    # Verify Planner was called at least twice (implied by LLM calls)
    assert mock_llm.invoke.call_count >= 2

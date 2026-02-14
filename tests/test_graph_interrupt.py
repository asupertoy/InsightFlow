import pytest
from unittest.mock import patch, MagicMock
from insightflow_core.graph import create_graph
from insightflow_core.state import AgentState

@pytest.fixture
def mock_graph():
    """Create graph with mocked main nodes."""
    with patch('insightflow_core.graph.researcher_node') as mock_researcher,          patch('insightflow_core.graph.coder_node') as mock_coder,          patch('insightflow_core.graph.writer_node') as mock_writer,          patch('insightflow_core.graph.reviewer_node') as mock_reviewer:

        mock_researcher.return_value = {"research_findings": []}
        mock_coder.return_value = {}
        mock_writer.return_value = {"draft_report": "Mock report content"}
        mock_reviewer.return_value = {"review_status": "approve"}

        graph = create_graph()
        yield graph

@patch('insightflow_core.nodes.clarifier.get_model_router')
def test_interrupt_before_human_response(mock_get_router, mock_graph):
    """Test interrupt before human_response node."""
    
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "1. Question?\n2. Question?" 
    mock_llm.invoke.return_value = mock_response
    
    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm
    mock_get_router.return_value = mock_router

    initial_state = {
        "original_task": "Research AI",
        "clarified_task": None,
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

    config = {"configurable": {"thread_id": "test_thread_interrupt"}}
    
    # Run the graph
    result = mock_graph.invoke(initial_state, config=config)
    
    # In LangGraph with interrupt, invoke returns the snapshot state *or* None if we use .stream()?
    # CompiledGraph.invoke returns the final state *Output*.
    # If interrupted, it returns the output up to that point.
    
    # If result is None, we need to inspect why.
    # However, create_graph uses SqliteSaver.
    
    current_state = mock_graph.get_state(config)
    # Check if we have questions
    assert current_state.values.get("clarification_questions")

@patch('insightflow_core.nodes.clarifier.get_model_router') 
@patch('insightflow_core.nodes.planner.get_model_router')
@patch('insightflow_core.nodes.planner.NoteTool')
def test_resume_after_user_input(mock_note_tool, mock_planner_router, mock_clarifier_router, mock_graph):
    """Test resuming execution after user input."""
    
    # Mock Clarifier LLM
    mock_c_llm = MagicMock()
    mock_c_resp = MagicMock()
    mock_c_resp.content = "Research AI Agents"
    mock_c_llm.invoke.return_value = mock_c_resp
    mock_router_c = MagicMock()
    mock_router_c.get_model.return_value = mock_c_llm
    mock_clarifier_router.return_value = mock_router_c
    
    # Mock Planner LLM
    mock_p_llm = MagicMock()
    mock_p_resp = MagicMock()
    mock_p_resp.content = '{"steps": [{"id": 1, "description": "Search", "search_query": "AI Agents", "status": "pending" }]}'
    mock_p_llm.invoke.return_value = mock_p_resp
    mock_router_p = MagicMock()
    mock_router_p.get_model.return_value = mock_p_llm
    mock_planner_router.return_value = mock_router_p
    
    # Mock NoteTool
    mock_note_tool.return_value.get_note.return_value = None

    # Important: In LangGraph, if we want to "resume" or simulate that user input has arrived,
    # we usually invoke with the updated state *and* same thread_id?
    # Here we are simulating a NEW invocation with a state that HAS answers.
    # But wait, create_graph() creates a NEW memory (SqliteSaver uses file path).
    # If db_path is default, it might reuse DB across tests if not cleaned?
    # create_graph default db_path="data/insightflow.db".
    # This might share state across tests!
    
    # We should use unique thread_id for each test invocation to avoid pollution.
    # And maybe we should use :memory: or temp file for DB path in tests.
    
    resumed_state = {
        "original_task": "Research AI",
        "clarified_task": None,
        "clarification_answers": "Focus on Agents", 
        "clarification_questions": ["Q1"],
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
    
    config = {"configurable": {"thread_id": "test_thread_resume_v2"}}
    
    # Since we provide 'clarification_answers' in input state, Clarifier Node *should* see it
    # provided that 'mock_graph' (which is CompiledGraph) accepts this input.
    # If the graph has checkpointing enabled, invoke(input, config) updates the state.
    
    result = mock_graph.invoke(resumed_state, config=config)
    
    # Debug print if result is None
    if result is None:
        print("Result is None. Current State:", mock_graph.get_state(config))
    
    assert result is not None
    assert "draft_report" in result

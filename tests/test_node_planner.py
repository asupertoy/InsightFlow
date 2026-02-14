import pytest
import json
from unittest.mock import patch, MagicMock
from insightflow_core.nodes.planner import planner_node
from insightflow_core.state import AgentState

@patch('insightflow_core.nodes.planner.get_model_router')
@patch('insightflow_core.nodes.planner.NoteTool')
def test_planner_initial_plan(mock_note_tool, mock_get_router):
    """Test initial plan generation."""
    # Mock LLM and its response
    plan_data = {
        "steps": [
            {
                "id": 1, 
                "description": "Research Python concurrency", 
                "search_query": "Python concurrency performance", 
                "reasoning": "Need to understand Python's GIL", 
                "status": "pending"
            },
            {
                "id": 2, 
                "description": "Research Go concurrency", 
                "search_query": "Go concurrency performance", 
                "reasoning": "Go has built-in concurrency", 
                "status": "pending"
            }
        ]
    }
    
    mock_response = MagicMock()
    mock_response.content = json.dumps(plan_data)
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    
    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm
    mock_get_router.return_value = mock_router

    state = {
        "original_task": "Compare Python and Go concurrency",
        "clarified_task": "Compare Python and Go concurrency",
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

    result = planner_node(state)

    assert "plan" in result
    assert len(result["plan"]) == 2
    assert result["plan"][0]["description"] == "Research Python concurrency"
    assert result["plan"][0]["status"] == "pending"

@patch('insightflow_core.nodes.planner.get_model_router')
@patch('insightflow_core.nodes.planner.NoteTool')
def test_planner_replanning(mock_note_tool_cls, mock_get_router):
    """Test replanning based on feedback."""
    
    # Mock NoteTool instance
    mock_note_tool = MagicMock()
    mock_note_tool_cls.return_value = mock_note_tool
    
    # Mock get_note returning existing note content with TAGS
    mock_note_tool.get_note.side_effect = lambda note_id: {
        "id": note_id,
        "content": "Existing content",
        "tags": ["completed"],  # Crucial: planner checks 'tags', not 'status'
        "status": "completed"
    } if note_id == "note_001" else None

    # Existing plan in state
    existing_plan = [
        {
            "id": 1,
            "description": "Research Python concurrency",
            "search_query": "Python concurrency",
            "reasoning": "Initial step",
            "status": "completed",
            "note_id": "note_001"
        }
    ]

    # New plan from LLM (revising step 1, adding step 2)
    new_plan_data = {
        "steps": [
            {
                "id": 1,
                "description": "Refined Python concurrency",
                "search_query": "Python GIL deep dive",
                "reasoning": "Need more details",
                "status": "pending"
            },
            {
                "id": 2,
                "description": "Research Go channels",
                "search_query": "Go channels patterns",
                "reasoning": "Compare with Python",
                "status": "pending"
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps(new_plan_data)
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    
    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm
    mock_get_router.return_value = mock_router

    state = {
        "original_task": "Compare Python and Go concurrency",
        "clarified_task": "Compare Python and Go concurrency",
        "plan": existing_plan,
        "current_step_index": 0,
        "last_step_success": True,
        "research_findings": [],
        "review_comments": "Please add details on Go channels.",
        "review_status": "revision_needed",
        "revision_count": 0,
        "messages": []
    }

    result = planner_node(state)
    new_plan = result["plan"]

    assert len(new_plan) == 2
    
    # Check Step 1 (reused note)
    assert new_plan[0]["id"] == 1
    assert new_plan[0]["note_id"] == "note_001"
    # Status handling depends on implementation, but let's check basic structure

    # Verify NoteTool._run was called to update/append revision
    # The planner calls _run directly
    mock_note_tool._run.assert_called()
    
    # Verify arguments for the update
    # We expect _run(action="update", note_id="note_001", ...)
    called = False
    for call in mock_note_tool._run.call_args_list:
        kwargs = call.kwargs
        if kwargs.get('action') == 'update' and kwargs.get('note_id') == 'note_001':
            if "REVISION" in kwargs.get('content', '') or "Revision" in kwargs.get('content', ''):
                called = True
                break
    
    assert called, "Expected _run to be called with update action for note_001 containing revision text"

    # Check Step 2 (new note)
    assert new_plan[1]["id"] == 2
    assert new_plan[1]["note_id"] != "note_001"

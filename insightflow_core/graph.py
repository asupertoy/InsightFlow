import sqlite3
import os
from typing import Literal

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState, AgentStateInput, AgentStateOutput
from .utils.logger import setup_logger

# 初始化日志
logger = setup_logger("insightflow.graph")

# 导入节点
# 注意: 这些模块目前可能是空的，需要实现相应的节点函数
from .nodes.clarifier import clarifier_node
from .nodes.planner import planner_node
from .nodes.researcher import researcher_node
from .nodes.coder import coder_node
from .nodes.writer import writer_node
from .nodes.reviewer import reviewer_node

def human_response_node(state: AgentState):
    """
    等待用户输入的空节点。
    实际运行时，Graph 会在此节点前中断 (interrupt_before=['human_response'])。
    用户在 State 中注入 clarification_answers 后，继续运行此节点。
    """
    pass

# --- 路由逻辑 ---

def route_clarification(state: AgentState) -> Literal["human_response", "planner"]:
    """
    决定澄清后的去向：
    - 如果生成的 state 中包含 clarified_task，说明澄清已完成（或用户已回答），进入 Planner。
    - 如果只有 clarification_questions，说明需等待用户回答，进入中断/等待状态。
    """
    if state.get("clarified_task"):
        return "planner"
    return "human_response"

def route_research_to_next(state: AgentState) -> Literal["analyst", "researcher", "writer"]:
    """
    决定搜索后的去向
    优化策略：
    1. 优先完成所有搜索任务 (Loop Researcher)。
    2. 搜索全部完成后，如果有数据，进行一次集中分析 (Analyst)。
    3. 如果没数据，直接写报告 (Writer)。
    """
    logger.info("ROUTER: Evaluating Research Next Step...")

    # 1. Check Loop (Priority: Finish all research first)
    plan = state.get("plan", [])
    all_completed = all(step.get("status") == "completed" for step in plan)
    
    if not all_completed:
        logger.info("ROUTER: -> Researcher (Next research step)")
        return "researcher"

    # 2. Check for Analysis (Batch Analysis at the end)
    # 只有当 raw_data_context 有数据，并且计划中有明确需要 Coding/Analysis 的步骤时才进入 Analyst
    # 简单的启发式：检查是否有 'code', 'plot', 'visualize', 'analyze data' 等关键词
    has_context = state.get("raw_data_context") and len(state["raw_data_context"]) > 0
    
    analysis_keywords = ["code", "python", "plot", "chart", "visualize", "graph", "analyze data", "calculation", "regression"]
    needs_analysis = any(
        any(k in step.get("description", "").lower() for k in analysis_keywords)
        for step in plan
    )
    
    if has_context and needs_analysis:
        logger.info("ROUTER: -> Analyst (Data & Analysis intent detected)")
        return "analyst"
        
    # 3. All done & No data -> Writer
    logger.info("ROUTER: -> Writer (All done)")
    return "writer"

def route_analyst_to_next(state: AgentState) -> Literal["writer"]:
    """
    Analyst 完成后的去向。
    由于我们将 Analyst 移到了 Research 循环之后，
    所以 Analyst 完成后必然进入 Writer。
    """
    logger.info("ROUTER: Analyst -> Writer")
    return "writer"

def route_writer_next(state: AgentState) -> Literal["reviewer"]:
    logger.info("ROUTER: Writer -> Reviewer")
    return "reviewer"

def route_review(state: AgentState) -> Literal["planner", "end"]:
    """
    审核后的流转逻辑：
    - 如果审核通过，结束。
    - 如果超过最大修改次数，强制结束。
    - 否则回滚给 Planner 重新规划。
    """
    # 熔断机制
    if state.get("revision_count", 0) >= 3:
        return "end"
    
    # 审核通过
    if state.get("review_status") == "approve":
        return "end"
        
    # 审核不通过，回退到 Planner
    return "planner"

# --- 图构建 ---

def create_graph(db_path: str = "data/insightflow.db"):
    # 1. 初始化图
    workflow = StateGraph(AgentState, input=AgentStateInput, output=AgentStateOutput)

    # 2. 添加节点
    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("human_response", human_response_node) # 新增: 用户交互节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    
    # 使用 "analyst" 作为节点名，对应 coder_node 实现
    workflow.add_node("analyst", coder_node)
    
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # 3. 设置入口和基础连接
    workflow.add_edge(START, "clarifier")
    
    # [修改] Clarifier 不再直接连 Planner，而是走条件路由
    # workflow.add_edge("clarifier", "planner")
    workflow.add_conditional_edges(
        "clarifier",
        route_clarification,
        {
            "human_response": "human_response",
            "planner": "planner"
        }
    )
    
    # [新增] 用户响应 -> 重新澄清
    workflow.add_edge("human_response", "clarifier")

    workflow.add_edge("planner", "researcher")

    # 4. 添加条件边
    
    # Researcher -> Analyst (如果有数据) OR Next Loop OR Writer
    workflow.add_conditional_edges(
        "researcher",
        route_research_to_next,
        {
            "analyst": "analyst", 
            "writer": "writer",
            "researcher": "researcher"
        }
    )

    # Analyst -> Writer
    workflow.add_conditional_edges(
        "analyst",
        route_analyst_to_next,
        {
            "writer": "writer"
        }
    )

    # Writer -> Reviewer (Drafting is done once)
    workflow.add_conditional_edges(
        "writer",
        route_writer_next,
        {
            "reviewer": "reviewer"
        }
    )

    # Reviewer -> Planner (如果不通过) OR End
    workflow.add_conditional_edges(
        "reviewer",
        route_review,
        {
            "planner": "planner",
            "end": END
        }
    )

    # 5. 设置持久化 (Checkpointer)
    # 确保存储目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    # 6. 编译
    # 关键修改：设置 interrupt_before=["human_response"] 以支持交互
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_response"]
    )
    return app

# 为了方便直接导入 app 使用 default path
try:
    app = create_graph()
except Exception as e:
    # 允许在没有正确设置时的 import (例如单元测试或是节点未实现时)
    print(f"Warning: Failed to compile graph: {e}")
    app = None

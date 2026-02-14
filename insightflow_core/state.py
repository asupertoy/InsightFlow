from typing import TypedDict, List, Annotated, Union, Optional, Any, Dict
from typing_extensions import NotRequired # Python 3.11+, fallback to manually importing if needed or using Total=False
import operator
from langgraph.graph.message import add_messages

# --- 结构化对象定义 ---

class PlanStep(TypedDict):
    """单个规划步骤的定义"""
    id: int
    description: str            
    search_query: str           
    reasoning: NotRequired[str]    
    status: str                 
    result: NotRequired[str]       
    
    # [新增] 关联产物：使用 NotRequired 标记为可选
    generated_code: NotRequired[str] 
    figure_path: NotRequired[str]    
    
    critique: NotRequired[str]     
    note_id: NotRequired[str] # [新增] 关联的笔记ID     

class Finding(TypedDict):
    """单个研究发现片段，包含溯源信息"""
    content: str                # 摘要或原文片段
    url: str                    # 来源 URL
    title: Optional[str]        # 网页标题
    score: Optional[float]      # 相关性评分 (0-1)

# --- 状态定义 ---

class AgentStateInput(TypedDict):
    """进入 Graph 时的初始输入"""
    original_task: str

class AgentStateOutput(TypedDict):
    """Graph 执行结束时的最终输出"""
    draft_report: str
    figure_paths: List[str]

class AgentState(TypedDict):
    """
    Agent 的核心工作流状态 (Shared Memory)。
    包含所有节点产生的中间数据。
    """
    # --- 输入与目标 ---
    original_task: str             # 用户最开始的输入
    clarified_task: str            # 经过澄清后的具体需求 (Clarifier 产出)
    clarification_questions: List[str] # [新增] 生成的澄清问题
    clarification_answers: str     # [新增] 用户的回答
    metadata: Dict[str, Any]       # 全局上下文/用户配置 (e.g. {'language': 'zh', 'max_loops': 5})
    
    # --- 规划与记忆 (核心大脑) ---
    plan: List[PlanStep]
    current_step_index: int        # [新增] 步骤指针 (0-based)，指示当前执行到 plan 的哪一步
    last_step_success: bool        # [新增] 上一步是否执行成功，用于条件路由
    
    # --- 搜索与知识 (增量追加) ---
    # [修改] 结构化的发现列表，支持引用溯源
    research_findings: Annotated[List[Finding], operator.add]
    
    # [新增] 滚动摘要 (Running Summary)
    # 针对海量网页场景：当 research_findings 太长时，Researcher 或 Summarizer 节点
    # 会对已知信息进行压缩汇总，避免 Context Window 爆炸。
    # 这也是 Researcher 下一轮搜索的 Context 基础。
    running_summary: str
    
    # --- 代码与数据分析 (核心差异化能力) ---
    
    # [关键修改] 改为 Dict[step_id, str] 或增量列表，避免后续步骤覆盖前序数据
    # 这里使用 Dict 更方便通过 Key (e.g. "financial_2023") 检索，或者简单已 current_step_index 为 Key
    # 为了通用性，可以是 List[str] append，Coder 节点读取全部历史 context
    raw_data_context: Annotated[List[str], operator.add] 
    
    # 代码历史：保留所有尝试过的代码片段，用于 Debug 或 Self-correction
    code_snippets: Annotated[List[str], operator.add]       
    
    # 代码执行输出：保留历史输出
    code_outputs: Annotated[List[str], operator.add]
    
    # 生成的图片路径：保留所有图片
    figure_paths: Annotated[List[str], operator.add]        
    
    # --- 写作与审核 ---
    draft_report: str              # 当前生成的报告草稿 (Writer 覆盖)
    review_comments: str           # Reviewer 的修改意见 (Reviewer 覆盖)
    review_status: str             # "approve" | "reject" | "continue"
    revision_count: int            # 循环计数器，防止无限循环
    
    # --- 消息历史 (Standard LangGraph) ---
    # 用于保留对话上下文，或者 Human-in-the-loop 时的交互记录
    # LangGraph 的 add_messages 已经完美兼容 ToolMessage
    messages: Annotated[List[Any], add_messages]

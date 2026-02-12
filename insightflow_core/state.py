from typing import TypedDict, List, Annotated, Union
import operator

class AgentState(TypedDict):
    # --- 输入与目标 ---
    original_task: str             # 用户最开始的输入
    clarified_task: str            # 经过澄清后的具体需求
    
    # --- 规划与记忆 ---
    plan: List[str]                # 当前的待办任务列表 (Steps)
    research_findings: List[str]   # 搜集到的文本信息 (经过 vLLM 摘要)
    
    # --- 代码与数据 (核心差异点) ---
    raw_data_context: str          # 从网页提取的原始表格/数值文本
    code_snippets: List[str]       # Agent 生成的 Python 代码历史
    code_outputs: str              # 代码执行的文本结果 (stdout)
    figure_paths: List[str]        # 生成的图片路径 (如 ./output/chart.png)
    
    # --- 产出与控制 ---
    draft_report: str              # 报告草稿
    review_comments: str           # 审核意见
    revision_count: int            # 循环计数器 (防止死循环)
    messages: Annotated[List, operator.add] # 对话历史 (用于 Debug)
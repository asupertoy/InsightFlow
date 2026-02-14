import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

from ..state import AgentState, PlanStep
from ..llm.router import get_model_router
from ..tools.note_tool import NoteTool 
from ..utils.parser import strip_thinking_tokens
from .prompts import (
    PLANNER_SYSTEM_PROMPT_INITIAL, 
    PLANNER_USER_TEMPLATE_INITIAL,
    PLANNER_SYSTEM_PROMPT_REFACTOR,
    PLANNER_USER_TEMPLATE_REFACTOR
)

# --- 输出结构定义 (Pydantic) ---

class PlanStepSchema(BaseModel):
    id: int = Field(description="Unique identifier for the step, starting from 1")
    description: str = Field(description="Clear, actionable task description")
    search_query: str = Field(description="The specific search query to use for this step on Tavily.")
    reasoning: Optional[str] = Field(description="Why this step is necessary")
    status: str = Field(description="Initial status, usually 'pending'", default="pending")

class PlanSchema(BaseModel):
    steps: List[PlanStepSchema] = Field(description="List of ordered research steps")

def planner_node(state: AgentState):
    """
    Planner Node (规划节点)
    
    功能：
    1. 首次运行：根据 clarified_task 生成初步的研究计划。
    2. 循环运行：根据 Reviewer 的反馈 (review_comments) 调整计划。
       - 并不是简单覆盖，而是“重构式追加”：
       - 保留已完成(completed)且未被批评的步骤。
       - 修改被批评的步骤。
       - 增加新的步骤以弥补信息的缺失。
    """
    
    print("--- 🧠 Planner Node: Planning/Refining Strategies ---")
    
    clarified_task = state.get("clarified_task")
    original_task = state.get("original_task")
    current_plan = state.get("plan", [])
    review_comments = state.get("review_comments", "")
    review_status = state.get("review_status", "")
    
    # 任务目标 (优先使用澄清后的)
    task_input = clarified_task if clarified_task else original_task
    
    # 获取智能模型
    llm = get_model_router().get_model("planning")
    note_tool = NoteTool() # 实例化 NoteTool
    
    # 构造解析器
    parser = JsonOutputParser(pydantic_object=PlanSchema)

    # --- 场景判断 ---
    
    # 场景 A: 首次规划 (没有现有计划，或者计划为空)
    if not current_plan:
        system_prompt = PLANNER_SYSTEM_PROMPT_INITIAL
        # 注入格式指令
        system_prompt += "\n" + parser.get_format_instructions()
        
        user_prompt = PLANNER_USER_TEMPLATE_INITIAL.format(task_input=task_input)
        
    # 场景 B: 基于反馈调整 (已有计划，且被 Reviewer 打回)
    else:
        # 将现有计划转为文本展示
        plan_str = json.dumps(current_plan, indent=2, ensure_ascii=False)
        
        system_prompt = PLANNER_SYSTEM_PROMPT_REFACTOR
        # 注入格式指令
        system_prompt += "\n" + parser.get_format_instructions()

        user_prompt = PLANNER_USER_TEMPLATE_REFACTOR.format(
            task_input=task_input,
            plan_str=plan_str,
            review_comments=review_comments
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # --- 调用模型 ---
    try:
        response = llm.invoke(messages)
        # 清洗可能存在的 Thinking token 或 markdown
        content = strip_thinking_tokens(response.content)
        
        # 尝试让解析器自动提取 JSON
        plan_output_dict = parser.parse(content)
        # 验证回 Pydantic 对象
        plan_output = PlanSchema(**plan_output_dict)
        
        # 将 Pydantic 对象转回 Dict 列表以存入 State
        new_plan = []
        from datetime import datetime
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 建立旧计划的 ID 映射 (如果存在)
        # 用 step.id (int) -> old_note_id (str)
        # 这样如果新计划保留了 step.id，我们就可以复用笔记
        old_note_map = {}
        if current_plan:
            for s in current_plan:
                # 兼容处理：检查 s 是否有 'id' 和 'note_id'
                if isinstance(s, dict) and "id" in s and "note_id" in s:
                    old_note_map[s["id"]] = s["note_id"]

        for step in plan_output.steps:
            # 这里的 step 是 Pydantic 对象
            
            # --- 确定 Note ID ---
            # 策略：如果 old_note_map 里有 step.id，优先复用。
            #       否则生成新的 note_YYYYMMDD_HHMMSS_{step.id}
            
            note_id = old_note_map.get(step.id)
            is_new_note = False
            
            if not note_id:
                note_id = f"note_{now_str}_{step.id}"
                is_new_note = True
            
            # --- 确定 Action ---
            
            # 清洗标题：去掉 "Review", "Rewrite" 等字眼 (以及可能的冗余前缀)
            clean_description = step.description.replace("Rewrite: ", "").replace("Review: ", "").replace("Task: ", "").replace("重写笔记：", "").replace("修改笔记：", "").strip()
            
            # 格式化标题，增加 Task ID
            note_title = f"Task {step.id}: {clean_description}"
            
            # 修改后的稳健逻辑：
            # 1. 尝试读取旧笔记（如果存在）
            # 2. 如果旧笔记存在且标为 "completed"，并且新任务描述与旧笔记标题差异不大，
            #    则认为该步骤无需完全重做，仅进行增量更新（或跳过）。
            # 3. 如果是全新步骤或者需要重写，则追加内容而不是覆盖。
            
            note_content = f"**{note_title}**\n**Reasoning**: {step.reasoning}\n**Status**: Pending"
            
            if is_new_note:
                # 全新笔记，直接创建
                note_tool._run(
                    action="create", 
                    title=note_title, 
                    content=note_content,
                    note_type="task_state",
                    tags=["plan", "pending"],
                    note_id=note_id
                )
                print(f"Created Note for step {step.id}: {note_id}")
            else:
                # 尝试读取旧笔记内容
                try:
                    # 使用 get_note 方法获取结构化数据，而不是 _run (返回字符串)
                    existing_note = note_tool.get_note(note_id)
                    
                    if existing_note:
                        existing_tags = existing_note.get("tags", [])
                        existing_content = existing_note.get("content", "")
                        
                        if "completed" in existing_tags:
                            print(f"Skipping overwrite for completed step {step.id}: {note_id}")
                            # 可选：如果描述变了，追加一个 Revision Note，但保留原内容
                            if clean_description not in existing_content:
                                update_content = f"\n\n---\n**Revision Task**: {clean_description}\n**Reasoning**: {step.reasoning}"
                                note_tool._run(
                                    action="update",
                                    note_id=note_id,
                                    content=update_content, # 追加
                                    tags=["completed", "revision"] # 保持 completed 状态
                                )
                        else:
                            # 如果未完成（Pending），则可以安全更新/覆盖
                            # 为了保留历史，我们选择追加而不是覆盖，或者用分隔符
                            print(f"Updating pending step {step.id}: {note_id}")
                            note_tool._run(
                                action="update",
                                note_id=note_id,
                                title=note_title, # 使用带 Task ID 的新标题
                                content=note_content, 
                                tags=["plan", "pending", "updated"]
                            )
                    else:
                         # 笔记不存在，fallback to create/overwrite
                        print(f"Note tool returned None for {note_id}. Creating new note.")
                        note_tool._run(
                            action="create", # Create new if not exist
                            title=note_title,
                            content=note_content,
                            note_type="task_state",
                            tags=["plan", "pending"],
                            note_id=note_id
                        )

                except Exception as read_err:
                    print(f"Error accessing existing note {note_id}: {read_err}. Fallback to create/overwrite.")
                    # Fallback
                    note_tool._run(
                        action="update",
                        note_id=note_id,
                        title=note_title,
                        content=note_content,
                        tags=["plan", "pending", "fallback"]
                    )

            new_plan.append({
                "id": step.id,
                "description": clean_description,
                "search_query": step.search_query, 
                "reasoning": step.reasoning,
                "status": step.status, 
                "result": None,
                "generated_code": None,
                "figure_path": None,
                "critique": None,
                "note_id": note_id # 存储关联的笔记ID
            })
            
        return {
            "plan": new_plan,
            "current_step_index": 0, # [新增] 初始化指针，指向第一个任务
            "last_step_success": True, # [新增] 初始化状态
            # 如果是打回重做，增加计数器
            "revision_count": state.get("revision_count", 0) + 1
        }
        
    except Exception as e:
        print(f"Planner Error: {e}")
        # Fallback: 如果结构化输出失败，至少不要崩溃
        return {
            "plan": [],
            "current_step_index": 0,
            "last_step_success": False,
            "revision_count": state.get("revision_count", 0)
        }

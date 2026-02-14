from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState
from ..llm.router import get_model_router
from .prompts import CLARIFIER_SYSTEM_PROMPT, CLARIFIER_QUESTIONS_TEMPLATE, CLARIFIER_FINAL_TEMPLATE

def clarifier_node(state: AgentState):
    """
    Clarifier Node (意图澄清节点 - 交互式)
    
    功能实现方案 B（进阶版）：
    1. 首次运行时：
       - 分析用户原始输入。
       - 如果任务模糊，生成 3 个澄清问题 (clarification_questions)。
       - 此时工作流会暂停 (需在 Graph 中配置 interrupt_after)，等待用户通过 update_state 注入回答。
    
    2. 二次运行时（用户已回答）：
       - 获取 clarification_answers。
       - 结合原始任务 + 问题 + 回答，生成最终的 clarified_task。
    """
    original_task = state["original_task"]
    clarification_answers = state.get("clarification_answers")
    clarified_task = state.get("clarified_task")
    
    # 如果已经有最终任务了，直接透传（或是 Re-entry 的情况）
    if clarified_task:
        return {} # No updates needed

    llm = get_model_router().get_model("clarifying")
    
    # --- 分支 1: 生成最终任务 (当已有用户回答时) ---
    if clarification_answers:
        print("--- 💡 Clarifier: Finalizing Task with User Feedback ---")
        prompt = CLARIFIER_FINAL_TEMPLATE.format(
            original_task=original_task,
            answers=clarification_answers
        )
        messages = [
            SystemMessage(content=CLARIFIER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        final_task = response.content.strip()
        
        return {
            "clarified_task": final_task,
            "messages": [
                SystemMessage(content=f"Clarified Goal: {final_task}")
            ]
        }
    
    # --- 分支 2: 生成澄清问题 (首次运行或未回答) ---
    # 如果还没有生成过问题，或者需要重新生成
    print("--- ❓ Clarifier: Generating Questions ---")
    prompt = CLARIFIER_QUESTIONS_TEMPLATE.format(original_task=original_task)
    messages = [
        SystemMessage(content=CLARIFIER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    questions_text = response.content.strip()
    
    # 简单的文本处理将 response 转为 list
    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    
    # 返回问题列表，Graph 需要在此处暂停
    return {
        "clarification_questions": questions,
        "messages": [
            SystemMessage(content=f"Please clarify: {questions_text}")
        ]
    }

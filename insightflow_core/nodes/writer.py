from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState
from ..llm.router import get_model_router
from ..utils.parser import strip_thinking_tokens
from .prompts import WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE

def writer_node(state: AgentState):
    """
    Writer Node (报告撰写节点)
    
    功能：
    1. 汇总所有 Plan Step 的执行结果 (result, generated_code, figure_path)。
    2. 基于原始查询 (query) 和当前收集到的信息，撰写或更新最终报告。
    3. 如果是中间步骤，生成 running_summary。
    4. 如果是最后一步，生成 draft_report。
    """
    print("--- ✍️ Writer Node: Drafting Report ---")
    
    plan = state.get("plan", [])
    query = state.get("query", "")
    current_index = state.get("current_step_index", 0)
    
    # 收集当前所有已完成步骤的信息
    completed_steps_content = []
    
    # Iterate through plan to collect context
    for i, step in enumerate(plan):
        if step.get("status") == "completed":
            content = f"**Step {i+1}: {step['description']}**\n"
            content += f"- Result: {step.get('result', 'No text result')}\n"
            if step.get("generated_code"):
                content += f"- Code Generated: Yes (Length: {len(step['generated_code'])})\n"
            if step.get("figure_path"):
                content += f"- Figure Created: {step['figure_path']}\n"
            completed_steps_content.append(content)
            
    context_text = "\n\n".join(completed_steps_content)
    
    # 获取写作模型 (DeepSeek)
    llm = get_model_router().get_model("writing") # Use smart model for high quality writing

    system_prompt = WRITER_SYSTEM_PROMPT

    user_prompt = WRITER_USER_TEMPLATE.format(
        query=query,
        context_text=context_text
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = strip_thinking_tokens(response.content)
    
    # 简单的逻辑：如果 plan 所有步骤都 completed，则认为 draft_report 是最终版
    # 否则只是更新 running_summary
    
    # Check if all steps are completed
    all_completed = all(s.get("status") == "completed" for s in plan)
    
    print(f"Report Drafted (Length: {len(content)})")

    return {
        "running_summary": content,
        "draft_report": content if all_completed else None
    }

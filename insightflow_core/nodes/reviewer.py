from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState
from ..llm.router import get_model_router
from ..utils.parser import strip_thinking_tokens
from .prompts import REVIEWER_SYSTEM_PROMPT, REVIEWER_USER_TEMPLATE

def reviewer_node(state: AgentState):
    """
    Reviewer Node (审查节点)
    
    功能：
    1. 接收 `draft_report` 和原始 `query`，以及 `revision_count`。
    2. 判断是否满足用户需求。
    3. 如果满足，标记 `review_status="approve"`。
    4. 如果不满足且可以改进，标记 `review_status="reject"` 并给出具体 `feedback`。
    5. 自动增加 `revision_count`。
    """
    print("--- 🧐 Reviewer Node: Checking Results ---")
    
    query = state.get("query", "")
    draft_report = state.get("draft_report", "")
    revision_count = state.get("revision_count", 0) + 1
    
    if int(revision_count) >= 3:
        print("Maximum revisions reached. Forcing approval.")
        return {
            "review_status": "approve",
            "feedback": "Max revisions reached. Finalizing report.",
            "revision_count": revision_count
        }
        
    if not draft_report:
        print("No report drafted yet. Approving empty (likely internal error or incomplete plan).")
        # 如果没有报告，可能还在 Planner 阶段，不应该走到 review，除非 Planner 完了但 Writer 没产出
        # 稳妥起见，我们认为如果不完整则不需要 review，直接通过（或者回滚）
        # 这里假设 Writer 只有最后才产 draft
        return {
            "review_status": "approve", # Force approval to end loop if empty
            "revision_count": revision_count
        }

    # 获取审查模型 (Reasoning / High Context)
    llm = get_model_router().get_model("reviewing")

    system_prompt = REVIEWER_SYSTEM_PROMPT

    user_prompt = REVIEWER_USER_TEMPLATE.format(
        query=query,
        draft_report=draft_report
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        content = strip_thinking_tokens(response.content)
        
        # 简单解析 JSON string used reasoning model might output natural language wrapper
        # 尝试正则提取 JSON block
        import json
        import re
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            decision = result.get("decision", "approve").lower()
            feedback = result.get("feedback", "")
        else:
            # Fallback
            if "approve" in content.lower():
                decision = "approve"
                feedback = "Initial approval (JSON parsing failed)."
            else:
                decision = "reject" 
                feedback = content
        
    except Exception as e:
        print(f"Review failed: {e}. Defaulting to approve.")
        decision = "approve"
        feedback = f"Error during review: {e}"

    print(f"Review Decision: {decision.upper()} (Round {revision_count})")
    
    return {
        "review_status": decision,
        "feedback": feedback,
        "revision_count": revision_count, # Increment here
        # Optionally pass feedback to history
        "revision_history": [f"Round {revision_count}: {decision} - {feedback}"] # Append logic via reducer? No need, list concat in operator
    }

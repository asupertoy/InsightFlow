import os
import re
from typing import Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentState
from ..llm.router import get_model_router
from ..tools.sandbox import Sandbox
from ..utils.parser import strip_thinking_tokens
from .prompts import CODER_SYSTEM_PROMPT, CODER_USER_TEMPLATE

# 预创建 output 目录
FIGURES_DIR = "data/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def coder_node(state: AgentState):
    """
    Analyst/Coder Node (数据分析师节点)
    
    功能：
    1. 接收 raw_data_context (包含多步搜索到的数据)。
    2. 针对当前 plan step 的要求 (description)，编写 Python 代码。
       - 代码需具备：数据提取(Regex/String)、数据清洗(Pandas)、绘图(Matplotlib)。
    3. 在 PythonREPL 沙箱中执行代码。
    4.Self-Healing: 如果报错，自动重试 (max_retries=2)。
    5. 保存图片路径和代码到 State。
    """
    print("--- 💻 Analyst Node: Coding & Analyzing ---")
    
    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0) - 1 # 注意：Researcher 跑完后 index 已经 +1 了，所以我们要回头看上一步（或者 Planner 设置时不指针对）
    
    # 修正指针逻辑：
    # 如果 last_step_success 为 True，且当前 index 指向的是下一个还没跑的 step，
    # 但 Analyst 是紧接着 Researcher 跑的，Researcher 跑完把 index+1 了。
    # 实际上 Analyst 处理的是 "刚刚完成搜索的那一步" (即 index-1)。
    # 但如果 Analyst 是独立的一步 Plan，那应该处理 Current Index。
    # **约定**：在我们的图里，Researcher -> Conditional(Analyst)。
    # 意味着 Analyst 是对 Researcher 结果的**后处理**。所以 Analyst 处理的是 plan[index-1]。
    
    # 更加稳健的逻辑：找到最近一个 status="completed" 但 generated_code 为空的步骤？
    # 或者简单点，我们回退一个 index
    step_idx = current_index - 1
    if step_idx < 0: 
        step_idx = 0 # Fallback
        
    current_step = plan[step_idx]
    
    description = current_step.get("description")
    raw_context = state.get("raw_data_context", [])
    
    # 将 List[str] 合并为一个大文本供 LLM 阅读
    context_str = "\n\n".join(raw_context)
    
    print(f"Analyzing data for step: {description}")
    
    # 获取智能模型 (DeepSeek)
    llm = get_model_router().get_model("coding")
    repl = Sandbox()

    # --- Prompt 设计 ---
    system_prompt = CODER_SYSTEM_PROMPT.format(
        context_str=context_str,
        FIGURES_DIR=FIGURES_DIR,
        step_idx=step_idx
    )

    user_prompt = CODER_USER_TEMPLATE.format(description=description)
    
    # --- 循环执行 (Self-Healing) ---
    max_retries = 2
    code_content = ""
    execution_result = ""
    figure_path = f"{FIGURES_DIR}/step_{step_idx}_fig.png"
    success = False
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    for attempt in range(max_retries + 1):
        try:
            # 1. 生成代码
            response = llm.invoke(messages)
            raw_output = strip_thinking_tokens(response.content)
            
            # 清洗 markdown 标记
            code_content = raw_output.replace("```python", "").replace("```", "").strip()
            
            print(f"Generated Code (Attempt {attempt}):\n{code_content[:100]}...")
            
            # 2. 执行代码
            # Capture stdout
            output = repl.run(code_content)
            
            # 检查是否有运行时错误 (PythonREPL 有时会把 stderr 混在 output 里，或者直接抛出异常)
            # LangChain PythonREPL 会捕获异常并返回为 string 形式的 "Checking..." 或者 Traceback
            # 我们假设只要 output 包含 "Traceback" 或 "Error"，就是失败
            
            if "Traceback" in output or "Error" in output or "Exception" in output:
                raise Exception(f"Runtime Error: {output}")
            
            print(f"Execution Success. Output: {output.strip()}")
            execution_result = output
            success = True
            break
            
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            # 生成错误反馈消息，让 LLM 重试
            messages.append(HumanMessage(content=raw_output)) # 把上次生成的代码放进去作为上下文
            messages.append(HumanMessage(content=f"The code execution failed with error:\n{str(e)}\n\nPlease fix the code and output the corrected version."))
    
    # --- 更新 State ---
    
    # 无论成功与否，都要更新
    # 成功的话，status 保持 completed (由 Researcher 设置)，但我们要填入 gathered artifacts
    
    updated_plan = []
    for i, step in enumerate(plan):
        if i == step_idx:
            updated_step = {
                **step,
                "generated_code": code_content if success else None,
                "figure_path": figure_path if success and os.path.exists(figure_path) else None,
                "result": (step.get("result", "") + f"\n\nAnalysis Result:\n{execution_result}").strip()
            }
            updated_plan.append(updated_step)
        else:
            updated_plan.append(step)
            
    return {
        "plan": updated_plan,
        "code_outputs": [execution_result],
        "code_snippets": [code_content],
        "figure_paths": [figure_path] if success and os.path.exists(figure_path) else [],
        # 如果分析失败了，要不要 fail? 暂时为了走通流程，保持 success 但记录 error
        "last_step_success": success 
    }

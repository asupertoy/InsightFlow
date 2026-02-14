import json
import os
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentState, Finding
from ..llm.router import get_model_router
from ..tools.search_tool import SearchTool
from ..tools.note_tool import NoteTool
from ..utils.parser import strip_thinking_tokens
from .prompts import RESEARCHER_SYSTEM_PROMPT_SUMMARIZER, RESEARCHER_USER_TEMPLATE_SUMMARIZER
from ..utils.logger import get_logger

logger = get_logger("researcher")

def researcher_node(state: AgentState):
    """
    Researcher Node (研究员节点)
    
    功能：
    1. 获取当前 plan 中 current_step_index 指向的任务。
    2. 使用 Tavily 执行搜索 (search_query)。
    3. (可选但推荐) 使用 vLLM (fast_llm) 对搜索结果进行阅读和摘要。
       - 这一步是为了防止把几十个网页的无关内容全部塞给后面的 Writer。
       - 这里我们实现一个简化的 map-reduce：Search -> Raw Content -> Summary。
    4. 更新 research_findings 和 running_summary。
    5. 标记步骤状态 (Status) 和 last_step_success。
    """
    logger.info("--- 🔎 Researcher Node: Searching and Reading ---")
    
    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0)
    
    # 边界检查
    if current_index >= len(plan):
        logger.warning("Researcher: All steps completed or index out of bounds.")
        return {"last_step_success": False}
        
    current_step = plan[current_index]
    query = current_step.get("search_query")
    description = current_step.get("description")
    
    logger.info(f"Executing Step {current_index + 1}: {description}")
    logger.info(f"Search Query: {query}")
    
    # --- 1. 执行搜索 ---
    try:
        # 使用封装好的 SearchTool
        search_tool = SearchTool(max_results=5)
        search_results = search_tool.invoke(query)
        
        # 将结果转换为 Finding 对象列表
        new_findings: List[Finding] = []
        raw_texts = []
        
        # SearchTool 已经保证返回 List[Dict] 并且包含 standard keys (url, content, title)
        if search_results:
            for res in search_results:
                url = res.get("url")
                content = res.get("content")
                title = res.get("title")
                
                # 简单清洗
                if content and len(content) > 50:
                    new_findings.append({
                        "url": url,
                        "content": content,
                        "title": title, 
                        "score": 0.8 # 默认置信度
                    })
                    raw_texts.append(f"Source ({title} - {url}): {content}")
        else:
            # 如果搜索失败或返回空
            logger.warning(f"Search warning: No results found for '{query}'")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        # 标记失败，但也许可以通过 retry 机制恢复，这就体现了 last_step_success 的作用
        return {
            "last_step_success": False,
            # 也可以更新 plan 里的 status 为 failed
            "plan": [
                {**step, "status": "failed"} if i == current_index else step 
                for i, step in enumerate(plan)
            ]
        }

    # --- 2. 阅读与摘要 (Machine Reading) ---
    # 如果找到了内容，我们用 fast_llm (vLLM) 做一个快速总结
    summary = ""
    if raw_texts:
        # 获取“快”模型
        llm = get_model_router().get_model("summarization")
        
        # 拼接上下文
        context_str = "\n\n".join(raw_texts)
        
        system_prompt = RESEARCHER_SYSTEM_PROMPT_SUMMARIZER
        user_prompt = RESEARCHER_USER_TEMPLATE_SUMMARIZER.format(
            description=description,
            context_str=context_str
        )
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            summary = strip_thinking_tokens(response.content)
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            summary = "Failed to generate summary, but raw findings are saved."

    # --- 3. 构造更新后的 Plan 和 更新笔记 ---
    # 标记当前步骤为 "completed"，并把结果填进去
    note_tool = NoteTool()
    
    updated_plan = []
    for i, step in enumerate(plan):
        if i == current_index:
            updated_step = {
                **step,
                "status": "completed",
                "result": summary
            }
            
            # [集成 NoteTool] 更新笔记内容
            note_id = step.get("note_id")
            if note_id:
                # 将新内容追加到笔记中
                # 先读取旧内容？或者直接覆盖。Summarizer 生成的内容通常是完整的 markdown 笔记。
                # 但为了保留之前的 metadata，我们用 update。
                logger.info(f"Updating Note {note_id} with research findings...")
                note_tool._run(
                    action="update",
                    note_id=note_id,
                    content=summary, # 用生成的摘要覆盖内容
                    tags=["completed", "research"]
                )
            else:
                logger.warning(f"Warning: No Note ID found for step {i+1}")

            updated_plan.append(updated_step)
        else:
            updated_plan.append(step)

    # --- 4. 返回 State ---
    
    # [优化] 为 raw_data_context 添加标签，方便 Coder 识别
    tagged_context = []
    if raw_texts:
        # 使用 Plan Description 作为 Tag，让 Coder 知道这段数据是干嘛的
        tag = f"Data source from step '{description}':\n"
        tagged_context = [tag + context_str]

    return {
        "plan": updated_plan,
        "research_findings": new_findings, # 这里会自动 append (operator.add)
        "current_step_index": current_index + 1, # 指针自动后移！
        "last_step_success": True,
        # 这里其实应该把 summary 追加到 running_summary，暂且简化处理
        "raw_data_context": tagged_context # append to list
    }

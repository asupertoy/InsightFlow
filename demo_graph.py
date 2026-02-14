import uuid
import sys
import os

# 确保能导入 insightflow_core
sys.path.append(os.getcwd())

from insightflow_core.graph import create_graph
from insightflow_core.llm.router import get_model_router

def run_demo():
    print("==================================================")
    print("   InsightFlow Agent Graph - End-to-End Test")
    print("==================================================")

    # 1. 初始化图
    # 使用唯一的 thread_id 分隔不同会话
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Initializing Graph (Thread ID: {thread_id})...")
    try:
        app = create_graph()
    except Exception as e:
        print(f"Error initializing graph: {e}")
        return

    # 2. 定义初始任务
    # 选择一个简单的任务以快速测试
    task = "给我讲一下DeepReserach的业界进展"
    print(f"\n[User Task]: {task}")

    initial_state = {
        "original_task": task,
        "revision_count": 0,
        # 初始化其他必要字段为空
        "clarified_task": None,
        "clarification_answers": None,
    }
    
    print("\n--- Phase 1: Clarification ---")
    
    # 3. 运行图 (第一阶段：澄清)
    # Graph 会一直运行直到遇到 interrupt_before=["human_response"]
    step_count = 0
    for event in app.stream(initial_state, config=config):
        step_count += 1
        for node_name, state_update in event.items():
            print(f"Step {step_count}: Node [{node_name}] executed.")
            # 可以在这里打印某些状态变化，比如 clarifier 提出的问题
            if node_name == "clarifier":
                qs = state_update.get("clarification_questions", [])
                if qs:
                    print(f"   => Clarifier asked {len(qs)} questions.")

    # 4. 检查中断状态
    snapshot = app.get_state(config)
    if snapshot.next and "human_response" in snapshot.next:
        print("\n[System]: Graph interrupted waiting for Human Input.")
        
        # 获取当前状态中的问题
        current_state = snapshot.values
        questions = current_state.get("clarification_questions", [])
        if questions:
            print(f"Questions:\n" + "\n".join([f"- {q}" for q in questions]))
        
        # 5. 模拟用户输入 (Human-in-the-loop)
        fake_user_answer = "我说的是泛指agent“深度研究”的领域，我想了解相关技术图片，截至到目前的最新进展（截至2026年1月）是什么。最好是包含具体时间线和引用的详细分析，最后再概括一下。"
        print(f"\n[User]: Providing Answer -> '{fake_user_answer}'")
        
        # 更新状态：注入回答
        # 注意：这里我们使用 update_state 来像 'human_response' 节点一样行为，或者直接修改 state
        # 实际上 human_response 是个空节点，更新状态后继续 stream 就会经过它流向 clarifier
        app.update_state(
            config, 
            {"clarification_answers": fake_user_answer},
            as_node="human_response" 
        )
        
        print("\n--- Phase 2: Execution (Planner -> Researcher -> ...) ---")
        
        # 6. 继续运行 (Resume)
        # 传入 None 表示从当前断点继续
        try:
            for event in app.stream(None, config=config):
                step_count += 1
                for node_name, state_update in event.items():
                    print(f"Step {step_count}: Node [{node_name}] executed.")
                    
                    if node_name == "planner":
                        plan = state_update.get("plan", [])
                        print(f"   => Planner created {len(plan)} steps.")
                    
                    if node_name == "researcher":
                        findings = state_update.get("research_findings", [])
                        current_idx = state_update.get("current_step_index")
                        print(f"   => Researcher found {len(findings)} items (Next Step: {current_idx}).")
                        
                    if node_name == "analyst":
                        print("   => Analyst ran code analysis.")
                        
                    if node_name == "writer":
                        report_content = state_update.get("draft_report")
                        report_len = len(report_content) if report_content else 0
                        print(f"   => Writer generated draft ({report_len} chars).")
                        
                    if node_name == "reviewer":
                        status = state_update.get("review_status")
                        print(f"   => Reviewer status: {status}")

        except Exception as e:
            print(f"Execution Error: {e}")
            import traceback
            traceback.print_exc()

    # 7. 查看最终结果
    final_snapshot = app.get_state(config)
    final_state = final_snapshot.values
    
    print("\n==================================================")
    
    # 打印 Token 消耗
    try:
        router = get_model_router()
        if router:
            stats = router.get_token_usage()
            print("\n📊 Token Usage Statistics:")
            print(f"Total Tokens: {stats['total']['total_tokens']}")
            print(f"  - Smart Model: {stats['smart_model']['total_tokens']} tokens ({stats['smart_model']['successful_requests']} requests)")
            print(f"  - Fast Model:  {stats['fast_model']['total_tokens']} tokens ({stats['fast_model']['successful_requests']} requests)")
            print("==================================================\n")
    except Exception as e:
        print(f"Could not fetch token stats: {e}")

    if "draft_report" in final_state and final_state["draft_report"]:
        print("✅ Workflow Completed Successfully!")
        print(f"Report Output (Preview):\n{final_state['draft_report'][:200]}...")
        print(f"\nFull report saved to 'data/output/' (simulated)")
    else:
        print("❌ Workflow Finished but no report found (or Review rejected multiple times).")
        print(f"Final Review Status: {final_state.get('review_status')}")

if __name__ == "__main__":
    run_demo()

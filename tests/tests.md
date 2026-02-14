# InsightFlow 测试计划

本文档列出了 InsightFlow 项目的关键测试点，旨在确保各个组件（LLM、Tools、Nodes）以及整体工作流逻辑的正确性。每个测试点都标注了需要提供给 AI 的**关联文件 (Context Files)**，以便生成测试代码。

## 1. 基础设施与单元测试 (Infrastructure & Unit Tests)

这些测试主要验证项目的基础组件是否配置正确，能否与外部服务（API、Docker容器）通信。

### 1.1 LLM 连接性测试 (`test_llm_connection.py`)
*   **测试目标**: 验证 `ModelRouter` 能否正确加载配置并连接到后端模型。
*   **关联文件**:
    - `insightflow_core/llm/router.py` (路由逻辑)
    - `config.py` (配置入口)
*   **测试功能**:
    1.  **Smart Model (DeepSeek)**: 调用 `model_router.get_model("planning")`，发送简单 Hello 消息，验证是否返回 DeepSeek 的响应 (DeepSeek-V3/R1)。
    2.  **Fast Model (vLLM)**: 调用 `model_router.get_model("summarization")`，发送简单摘要请求，验证是否收到本地容器 (localhost:8000) 的响应。
    3.  **Authentication**: 验证 API Key 是否被正确读取（不报错）。

### 1.2 工具集测试 (`test_tools.py`)
*   **测试目标**: 验证各个 Tool Class 的功能实现。
*   **关联文件**:
    - `insightflow_core/tools/search_tool.py`
    - `insightflow_core/tools/note_tool.py`
    - `config.py`
*   **测试功能**:
    1.  **SearchTool**: 模拟调用 `duckduckgo`/`tavily`，验证是否返回包含 `url` 和 `content` 的标准列表结构。需要处理网络超时或无结果的情况。
    2.  **NoteTool**:
        -   `add_note`: 写入内容到 `data/notes/test_note.txt`。
        -   `read_note`: 读取刚写入的文件，验证内容一致。
        -   `list_notes`: 验证列表包含刚创建的文件。
        -   `delete_note`: 清理测试文件。

## 2. 节点逻辑测试 (Node Logic Tests)

这些测试将独立运行每个 Node 函数，通过 Mock State 输入，验证输出 State 的变更。

### 2.1 澄清节点 (`test_node_clarifier.py`)
*   **测试目标**: 验证交互式澄清逻辑（Socratic Questioning）。
*   **关联文件**:
    - `insightflow_core/nodes/clarifier.py`
    - `insightflow_core/nodes/prompts.py`
    - `insightflow_core/state.py`
*   **测试功能**:
    1.  **Case A (模糊输入)**: 输入 `original_task="帮我写个报告"`, `clarification_answers=None`。
        -   *Expect*: 返回 `clarification_questions` (List)，且 `clarified_task` 为空。
    2.  **Case B (有回答)**: 输入 `original_task="..."`, `clarification_answers="关于AI Agent的报告"`。
        -   *Expect*: 返回 `clarified_task` (String)，且不为空。

### 2.2 规划节点 (`test_node_planner.py`)
*   **测试目标**: 验证分解任务的能力。
*   **关联文件**:
    - `insightflow_core/nodes/planner.py`
    - `insightflow_core/nodes/prompts.py`
    - `insightflow_core/state.py`
*   **测试功能**:
    -   输入 `clarified_task="比较 Python 和 Go 的并发性能"`.
    -   *Expect*: 返回 `plan` (List[PlanStep])，其中包含至少一个步骤（搜索、写代码、分析）。

### 2.3 路由逻辑测试 (`test_router_edges.py`)
*   **测试目标**: 验证 `graph.py` 中的条件边 (`conditional_edges`) 逻辑。
*   **关联文件**:
    - `insightflow_core/graph.py` (包含 route_* 函数)
    - `insightflow_core/state.py`
*   **测试功能**:
    1.  `route_clarification`:
        -   State 有 `clarified_task` -> 路由到 `planner`。
        -   State 无 `clarified_task` -> 路由到 `human_response`。
    2.  `route_review`:
        -   `review_status="approve"` -> 路由到 `end`。
        -   `review_status="reject"` & `revision_count < 3` -> 路由到 `planner`。
        -   `revision_count >= 3` -> 路由到 `end` (熔断机制)。

## 3. 集成工作流测试 (Integration/End-to-End Tests)

这些测试将运行完整的 `StateGraph`，验证节点间的协作和数据流转。

### 3.1 线性流程测试 (`test_graph_linear.py`)
*   **测试目标**: 验证一个不需要人工干预和反复修改的简单全自动流程。
*   **关联文件**:
    - `insightflow_core/graph.py`
    - `insightflow_core/nodes/*.py` (建议提供所有 Node 文件以供参考)
    - `insightflow_core/state.py`
*   **测试场景**:
    -   User Task: "搜索 LangGraph 的最新特性并总结" (假设任务足够清晰，不需要澄清，或者 Mock 掉澄清环节)。
    -   **Expect**: Graph 必须成功运行到 `END`，且 `AgentState` 中包含 `draft_report`。

### 3.2 人机交互中断测试 (`test_graph_interrupt.py`)
*   **测试目标**: 验证 `interrupt_before=["human_response"]` 机制。
*   **关联文件**:
    - `insightflow_core/graph.py`
    - `insightflow_core/nodes/clarifier.py`
    - `insightflow_core/state.py`
*   **测试场景**:
    1.  启动 Graph，输入模糊任务 "调研一下那个新出的AI"。
    2.  Verify: Graph 停止在 `human_response` 节点前。
    3.  Action: 获取当前的 `clarification_questions`。
    4.  Action: 手动更新 State (`update_state`) 注入 `clarification_answers`。
    5.  Action: `resume` Graph 执行。
    6.  Verify: Graph 继续运行进入 `planner`。

### 3.3 循环与自我修正测试 (`test_graph_revision.py`)
*   **测试目标**: 验证 Reviewer 拒绝后，Planner 能够接收反馈并更新计划。
*   **关联文件**:
    - `insightflow_core/graph.py`
    - `insightflow_core/nodes/reviewer.py`
    - `insightflow_core/nodes/planner.py`
*   **测试场景**:
    -   Mock `reviewer_node` 在第一次调用时总是返回 `reject` 和 critique，第二次返回 `approve`。
    -   **Expect**: 最终路径应包含: `... -> writer -> reviewer -> planner -> ... -> writer -> reviewer -> end`。

## 4. Docker 环境验证 (Docker Environment Verification)

在运行上述测试之前，必须确认本地环境状态。

*   **关联文件**:
    - `docker-compose.yml`
*   **容器状态检测**:
    -   命令: `docker ps` 确认 `insightflow-vllm` 状态为 `Up`。
    -   端口检测: `curl http://localhost:8000/v1/models` 确认 HTTP 服务响应正常。
*   **显存监控 (Optional)**:
    -   在大量并发请求时，观察 `nvidia-smi` 确保显存未溢出。

---

**建议测试执行顺序**:
1.  Docker 环境验证 (手动或脚本)。
2.  LLM 连接性测试 (确保模型可用)。
3.  单元测试 (Nodes & Tools)。
4.  集成测试 (完整 Graph)。

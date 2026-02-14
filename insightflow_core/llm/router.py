import os
import sys
from typing import Literal, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 添加项目根目录到路径，以便导入config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config

class TokenMonitorCallback(BaseCallbackHandler):
    """
    Token 消耗监控回调
    用于统计所有通过 Router 调用的模型 Token 消耗
    """
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.successful_requests = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """当 LLM 运行结束时调用"""
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                self.total_tokens += token_usage.get("total_tokens", 0)
                self.prompt_tokens += token_usage.get("prompt_tokens", 0)
                self.completion_tokens += token_usage.get("completion_tokens", 0)
                self.successful_requests += 1

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "successful_requests": self.successful_requests
        }
    
    def reset(self):
        """重置统计数据"""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.successful_requests = 0

class ModelRouter:
    """
    混合推理路由器：根据任务类型选择合适的模型。
    - 复杂逻辑/规划/编程 -> Cloud API (DeepSeek/GPT-4o)
    - 文本摘要/信息提取 -> Local/Low-cost Model (vLLM Qwen/Llama)
    """
    def __init__(self, mode: str = None):
        # 加载 .env 文件到环境变量
        load_dotenv()
        
        # 1. 确定运行模式
        # 如果传入了 mode，则使用传入值；否则从 config 读取
        self.mode = mode if mode else config.model_routing_mode
        print(f"[ModelRouter] Initialized in mode: {self.mode}")
        
        # 初始化 Token 监控 (分离监控)
        self.smart_token_monitor = TokenMonitorCallback()
        self.fast_token_monitor = TokenMonitorCallback()

        # 配置模型参数
        self.smart_model_name = config.smart_llm_model
        self.smart_base_url = config.smart_llm_base_url
        self.smart_api_key = config.smart_llm_api_key
        
        # 确定 Fast Model 参数
        if self.mode == "cloud_only":
            self.fast_model_name = self.smart_model_name
            self.fast_base_url = self.smart_base_url
            self.fast_api_key = self.smart_api_key
        else:
            self.fast_model_name = config.fast_llm_model
            self.fast_base_url = config.fast_llm_base_url
            self.fast_api_key = config.fast_llm_api_key

        # 模型实例缓存表 (lazy registry)
        self._models: Dict[str, ChatOpenAI] = {}
        
        # 为了兼容性，保留 default 属性 (指向 planning/summarization)
        # 这些属性将在首次访问时或 __init__ 结束时初始化，这里先设为 property 或者直接初始化
        # 为了简单，我们手动初始化所有任务类型的模型
        self._init_models()
        self.smart_llm = self._models["planning"] # Backwards compatibility
        self.fast_llm = self._models["summarization"] # Backwards compatibility

    def _create_smart_llm(self, temperature: float = 0.0) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.smart_model_name,
            base_url=self.smart_base_url,
            api_key=self.smart_api_key,
            temperature=temperature,
            callbacks=[self.smart_token_monitor]
        )

    def _create_fast_llm(self, temperature: float = 0.1) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.fast_model_name,
            base_url=self.fast_base_url,
            api_key=self.fast_api_key,
            temperature=temperature,
            callbacks=[self.fast_token_monitor]
        )

    def _init_models(self):
        """初始化各个任务专用的模型实例"""
        # --- Smart Tasks ---
        # 规划 (严谨)
        self._models["planning"] = self._create_smart_llm(temperature=0.0)
        # 编程 (严谨)
        self._models["coding"] = self._create_smart_llm(temperature=0.0)
        # 审查 (严谨)
        self._models["reviewing"] = self._create_smart_llm(temperature=0.0)
        # 澄清 (严谨)
        self._models["clarifying"] = self._create_smart_llm(temperature=0.0)
        # 写作 (严谨/深度思考)
        self._models["writing"] = self._create_smart_llm(temperature=0.0)
        
        # --- Fast Tasks ---
        # 摘要 (稍灵活，用于润色)
        self._models["summarization"] = self._create_fast_llm(temperature=0.1)
        # 提取 (严谨)
        self._models["extraction"] = self._create_fast_llm(temperature=0.0)

    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        """
        获取当前会话的 Token 消耗统计，区分 Smart 和 Fast 模型
        """
        smart_stats = self.smart_token_monitor.get_stats()
        fast_stats = self.fast_token_monitor.get_stats()
        
        # 计算总和
        total_stats = {
            "total_tokens": smart_stats["total_tokens"] + fast_stats["total_tokens"],
            "prompt_tokens": smart_stats["prompt_tokens"] + fast_stats["prompt_tokens"],
            "completion_tokens": smart_stats["completion_tokens"] + fast_stats["completion_tokens"],
            "successful_requests": smart_stats["successful_requests"] + fast_stats["successful_requests"]
        }
        
        return {
            "total": total_stats,
            "smart_model": smart_stats,
            "fast_model": fast_stats
        }

    def reset_token_usage(self):
        """重置所有 Token 统计"""
        self.smart_token_monitor.reset()
        self.fast_token_monitor.reset()

    def get_model(self, task_type: str):
        """根据任务类型路由到不同的模型实例"""
        return self._models.get(task_type, self.smart_llm)

# 全局单例 - 延迟初始化
model_router = None

def get_model_router(mode: str = None) -> ModelRouter:
    """
    获取模型路由器单例。
    
    自动适配逻辑：
    1. 检查当前 config 中的模式。
    2. 如果单例已存在，但其 mode 与 config 不一致（或与传入的 mode 不一致），
       则自动触发重置和重新初始化，同时保留 Token 统计信息。
    """
    global model_router
    
    # 确定目标模式：优先使用传入参数，否则使用 config
    target_mode = mode if mode else config.model_routing_mode
    
    # 如果实例已存在，检查模式是否需要切换
    if model_router is not None:
        if model_router.mode != target_mode:
            print(f"[ModelRouter] Mode change detected ({model_router.mode} -> {target_mode}). Reloading router...")
            
            # 保存旧实例的 token 监控数据
            old_smart = model_router.smart_token_monitor
            old_fast = model_router.fast_token_monitor
            
            # 重新初始化
            model_router = ModelRouter(mode=target_mode)
            
            # 恢复 token 监控数据（累加）
            # 1. Smart Monitor
            model_router.smart_token_monitor.total_tokens += old_smart.total_tokens
            model_router.smart_token_monitor.prompt_tokens += old_smart.prompt_tokens
            model_router.smart_token_monitor.completion_tokens += old_smart.completion_tokens
            model_router.smart_token_monitor.successful_requests += old_smart.successful_requests

            # 2. Fast Monitor
            model_router.fast_token_monitor.total_tokens += old_fast.total_tokens
            model_router.fast_token_monitor.prompt_tokens += old_fast.prompt_tokens
            model_router.fast_token_monitor.completion_tokens += old_fast.completion_tokens
            model_router.fast_token_monitor.successful_requests += old_fast.successful_requests
            
    # 如果实例不存在，直接初始化
    if model_router is None:
        model_router = ModelRouter(mode=target_mode)
        
    return model_router

def reset_model_router():
    """手动重置 ModelRouter 单例（通常不需要，get_model_router 会自动处理）"""
    global model_router
    if model_router:
        print(f"[ModelRouter] Resetting router from mode: {model_router.mode}...")
    model_router = None

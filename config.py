import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# =============================================================================
# 搜索服务配置
# =============================================================================

# 使用的搜索服务，可选项: 'duckduckgo', 'tavily', 'perplexity', 'searxng'
search_api = 'duckduckgo'
# Tavily 搜索 API 密钥，从环境变量读取
tavily_api_key = os.getenv("TAVILY_API_KEY")

# =============================================================================
# LLM 提供者配置
# =============================================================================

# LLM 请求超时时间（秒），默认 60 秒
llm_timeout = 60

# =============================================================================
# Cloud LLM 配置 (Smart Brain) - 用于规划、写代码、Review
# =============================================================================

# 路由模式配置
# 'hybrid': 混合模式 (Cloud for Brain, Local for Reader)
# 'cloud_only': 全云端模式 (All Cloud, 适用于没有本地显卡的情况)
model_routing_mode = "hybrid"  # 默认使用混合模式

# Cloud LLM 模型名称
# 推荐使用 deepseek-chat (DeepSeek-V3) 平衡速度与智力，或者 deepseek-reasoner (DeepSeek-R1) 用于深度思考
smart_llm_model = "deepseek-chat"

# Cloud LLM API 密钥，从环境变量读取
smart_llm_api_key = os.getenv("SMART_LLM_API_KEY")

# Cloud LLM 基础 URL，从环境变量读取，默认为 DeepSeek 官方 API
smart_llm_base_url = os.getenv("SMART_LLM_BASE_URL", "https://api.deepseek.com")

# =============================================================================
# Local LLM 配置 (Fast Reader) - 用于阅读长文档、摘要
# =============================================================================

# Local LLM 模型名称
fast_llm_model = "qwen2.5-14b"

# Local LLM API 密钥，从环境变量读取，默认 "EMPTY"
# vLLM 本地部署通常不需要 Key，但在客户端设为 "EMPTY" 以兼容 OpenAI SDK
fast_llm_api_key = os.getenv("FAST_LLM_API_KEY", "EMPTY")

# Local LLM 基础 URL，从环境变量读取，默认 "http://localhost:8000/v1"
fast_llm_base_url = os.getenv("FAST_LLM_BASE_URL", "http://localhost:8000/v1")



# =============================================================================
# 服务器配置
# =============================================================================
host = "0.0.0.0"  # 服务器主机地址
port = 8000  # 服务器端口

# =============================================================================
# CORS 配置
# =============================================================================
cors_origins = "http://localhost:5173,http://localhost:3000"  # 允许的 CORS 源，用逗号分隔

# =============================================================================
# 日志配置
# =============================================================================
log_level = "INFO" # 日志级别，可选项: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# =============================================================================
# Web 搜索配置
# =============================================================================
max_web_research_loops = 3  # 最大 Web 研究循环次数
fetch_full_page = True  # 是否获取完整页面内容
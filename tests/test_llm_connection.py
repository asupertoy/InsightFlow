import pytest
import os
from insightflow_core.llm.router import get_model_router
import config

def test_smart_model_connection():
    """测试 Smart Model (DeepSeek) 连接性 - 使用真实API"""
    # 获取 model router
    router = get_model_router()

    # 获取 smart model
    smart_llm = router.get_model("planning")

    # 发送简单 Hello 消息
    try:
        response = smart_llm.invoke("Hello, can you respond with 'Hello from DeepSeek'?")
        assert response.content is not None
        assert len(response.content.strip()) > 0
        print("Smart Model (DeepSeek) connection successful")
        print(f"Response: {response.content[:100]}...")
    except Exception as e:
        pytest.fail(f"Smart Model connection failed: {e}")

def test_fast_model_connection():
    """测试 Fast Model (vLLM) 连接性 - 使用真实API"""
    # 获取 model router
    router = get_model_router()

    # 获取 fast model
    fast_llm = router.get_model("summarization")

    # 发送简单摘要请求
    try:
        response = fast_llm.invoke("Summarize this text: 'This is a test message.'")
        assert response.content is not None
        assert len(response.content.strip()) > 0
        print("Fast Model (vLLM) connection successful")
        print(f"Response: {response.content[:100]}...")
    except Exception as e:
        pytest.fail(f"Fast Model connection failed: {e}")

def test_api_key_loading():
    """测试 API Key 是否正确读取"""
    assert config.fast_llm_api_key is not None, "FAST_LLM_API_KEY not set"
    print("API Keys loaded successfully")

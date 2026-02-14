import pytest
from insightflow_core.llm.router import get_model_router, reset_model_router
import config

def test_auto_reload_config():
    """测试 Config 变更后 Router 的自动重载"""
    # 确保之前的状态被清除
    reset_model_router()
    
    # 1. 初始为 Hybrid 模式
    config.model_routing_mode = "hybrid"
    print("\n--- Phase 1: Hybrid Mode ---")
    router1 = get_model_router()
    assert router1.mode == "hybrid"
    
    # 模拟消耗一些 tokens (在 smart monitor 上)
    router1.smart_token_monitor.total_tokens = 100
    router1.smart_token_monitor.successful_requests = 1
    
    # 验证初始状态
    usage1 = router1.get_token_usage()
    assert usage1["total"]["total_tokens"] == 100
    
    # 2. 更改 Config 为 Cloud Only
    print("\n--- Phase 2: Switch to Cloud Only ---")
    config.model_routing_mode = "cloud_only"
    
    # 获取 Router（应该是一个新实例，且保留了 Token 统计）
    router2 = get_model_router()
    
    # 验证模式切换
    assert router2.mode == "cloud_only"
    assert router2 is not router1  # 应该是新对象
    
    # 验证 Token 统计继承
    usage2 = router2.get_token_usage()
    print(f"Stats after reload: {usage2}")
    
    # 继承了之前的 100 stats
    assert usage2["total"]["total_tokens"] == 100
    assert usage2["total"]["successful_requests"] == 1
    
    # 验证新对象能够继续累加
    router2.smart_token_monitor.total_tokens += 50
    usage3 = router2.get_token_usage()
    assert usage3["total"]["total_tokens"] == 150

if __name__ == "__main__":
    test_auto_reload_config()

def test_singleton_pattern():
    """Test that get_model_router returns the same instance"""
    reset_model_router()
    router1 = get_model_router()
    router2 = get_model_router()
    assert router1 is router2

def test_mode_selection():
    """Test that mode is correctly set from config or arg"""
    reset_model_router()
    config.model_routing_mode = "hybrid"
    router = get_model_router()
    assert router.mode == "hybrid"
    
    # Test override
    reset_model_router()
    router = get_model_router(mode="cloud_only")
    assert router.mode == "cloud_only"

def test_model_initialization():
    """Test that models are initialized eagerly or lazily"""
    reset_model_router()
    router = get_model_router()
    assert "planning" in router._models
    assert "summarization" in router._models

def test_get_model_method():
    """Test get_model returns correct model for task"""
    reset_model_router()
    router = get_model_router()
    model = router.get_model("planning")
    assert model == router._models["planning"]
    
    # Test fallback
    model = router.get_model("unknown_task")
    assert model == router.smart_llm

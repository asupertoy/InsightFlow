import re

def strip_thinking_tokens(text: str) -> str:
    """
    清理 DeepSeek 等模型产生的 Thinking Tokens (<think>...</think>)
    """
    pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    return pattern.sub("", text).strip()

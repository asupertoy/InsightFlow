from langchain_experimental.utilities import PythonREPL

class Sandbox:
    """
    Python REPL 沙箱工具封装。
    用于执行生成的 Python 代码 (Data Analysis, Visualization)。
    """
    def __init__(self):
        self._repl = PythonREPL()

    def run(self, code: str) -> str:
        """
        执行 Python 代码并在沙箱中捕获输出。
        """
        try:
            return self._repl.run(code)
        except Exception as e:
            return f"Sandbox Execution Error: {str(e)}"

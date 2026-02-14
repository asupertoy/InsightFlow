import os
from typing import List, Dict, Any, Union
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
import config

class SearchTool:
    """
    统一的搜索工具封装类。
    根据 config.py 中的 search_api配置，选择具体的搜索实现。
    """
    def __init__(self, max_results: int = 5):
        self.api_type = config.search_api
        self.max_results = max_results
        self._tool = self._initialize_tool()

    def _initialize_tool(self):
        """根据配置初始化底层工具"""
        if self.api_type == "tavily":
            api_key = config.tavily_api_key
            if not api_key:
                print("Warning: Tavily API Key not found, falling back to DuckDuckGo")
                self.api_type = "duckduckgo" # Fallback
                return DuckDuckGoSearchResults(max_results=self.max_results)
            return TavilySearchResults(max_results=self.max_results, tavily_api_key=api_key)
        
        elif self.api_type == "duckduckgo":
             return DuckDuckGoSearchResults(max_results=self.max_results)
        else:
             # Default fallback
             print(f"Warning: Unknown search_api '{self.api_type}', processing with DuckDuckGo")
             return DuckDuckGoSearchResults(max_results=self.max_results)

    def invoke(self, query: str) -> List[Dict[str, str]]:
        """
        执行搜索并返回标准化的结果列表。
        Returns:
            List[Dict]: [{'url': '...', 'content': '...', 'title': '...'}]
        """
        try:
            # invoke() returns different structures based on the tool
            raw_results = self._tool.invoke(query)
            return self._standardize_results(raw_results)
        except Exception as e:
            print(f"SearchTool execution failed: {e}")
            return []

    def _standardize_results(self, results: Any) -> List[Dict[str, str]]:
        """将不同工具的返回结果统一格式"""
        standardized = []

        if isinstance(results, str):
            # 某些工具出错或仅返回字符串时
            # DuckDuckGo 此时可能返回的是 JSON string 或者 plain text，简单处理
            standardized.append({
                "url": "",
                "content": results,
                "title": "Search Result"
            })
            return standardized

        if isinstance(results, list):
            for item in results:
                # Tavily keys: url, content
                # DuckDuckGo keys: link, snippet, title
                
                url = item.get("url") or item.get("link") or ""
                content = item.get("content") or item.get("snippet") or item.get("body") or ""
                title = item.get("title") or url
                
                if content:
                    standardized.append({
                        "url": url,
                        "content": content,
                        "title": title
                    })
        
        return standardized

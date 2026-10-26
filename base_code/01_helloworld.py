"""
测试1：deepagents的第一个测试例子
"""
import os
from typing import Literal

from tavily import TavilyClient
from langchain_core.tools import tool

from dotenv import load_dotenv

# 加载.env环境配置，如果在环境变量中配置了，就不会加载.env文件
load_dotenv()
# 创建TavilyClient对象
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# 网络搜索的工具函数
# 函数体开头必须指定函数功能的文档说明 =》 智能体会自动发送给大模型来理解工具的功能
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """使用 Tavily 进行联网搜索"""
    print(f"正在搜索: {query}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

print(internet_search.invoke({"query": "deepagents是什么？"}))
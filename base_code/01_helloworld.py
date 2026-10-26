"""
测试1：deepagents的第一个测试例子
"""
import os
from typing import Literal

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
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
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """使用 Tavily 进行联网搜索"""
    # print(f"正在搜索: {query}")
    print("执行internet_search工具")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# print(internet_search.invoke({"query": "deepagents是什么？"}))

# 初始化模型
llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 创建深度智能体
agent = create_deep_agent(
    model = llm, # 模型
    tools=[internet_search], # 工具
    system_prompt="""你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 internet_search 工具来收集信息。""",  # 系统提示
)

# 执行agent
result = agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                "content": "请搜索deepagents是什么？"
            }
        ]
    }
)

# 读取结果数据
print(result)
print(result["messages"][-1].content)
"""
测试4_4: 通过response_format配置给子智能体指定输出数据的格式
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from deepagents import create_deep_agent
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def internet_search(
    query: str, # 搜索内容
    max_results: int = 3, # 最大返回结果数量
    topic: Literal["general", "news", "finance"] = "general", # 搜索主题
    include_raw_content: bool = False, # 是否返回原始内容 False为精简内容
):
    """
    使用tavily进行网络搜索
    :param query: 搜索内容
    :param max_results: 最大返回结果数量
    :param topic: 搜索主题
    :param include_raw_content: 是否返回原始内容
    :return: 搜索结果
    """
    print(f"---开始搜索：{query}")
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

chat_model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 子智能体的 response_format 会强制指定 tool_choice，而 DeepSeek V4 的思考模式
# 不支持强制 tool_choice，因此子智能体单独使用关闭思考模式的模型实例
# 400: Thinking mode does not support this tool_choice
subagent_model = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    extra_body={"thinking": {"type": "disabled"}},
)

# 自定义输出数据格式类型，在类型中指定数据名称和类型来约束响应数据的格式
class WeatherResponse(BaseModel):
    city: str = Field(description="城市名称")
    temperature: str = Field(description="温度")
    condition: str = Field(description="天气状况")

subagent = {
    "name": "researcher",
    "description": "你是一个强大的助手",
    "system_prompt": "通过internet_search工具搜索结果",
    "tools": [internet_search],
    "model": subagent_model, # 使用关闭思考模式的模型，避免 response_format 与思考模式冲突
    "response_format": WeatherResponse, # 在子智能体配置上传递`response_format`， 并指定为自定义的BaseModel类型
}

agent = create_deep_agent(
    model=chat_model, # 主智能体保持思考模式不变
    subagents=[subagent],
    system_prompt="你是一个强大的管家，你不要亲自处理任务，将任务交给researcher子智能体处理",

)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "武汉的天气怎么样？以json的格式返回数据"}]}
)

print(result['messages'])
print(result['messages'][-1].content)
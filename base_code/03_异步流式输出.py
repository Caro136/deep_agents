"""
测试3：异步执行
    agent.ainvoke(): 异步非流式输出
    agent.astream()：异步流式输出
    注意：astream() 返回异步迭代器，必须用 `async for`进行遍历
"""
import asyncio
import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from tavily import TavilyClient

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

# 初始化模型
llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 创建深度智能体
agent = create_deep_agent(
    model=llm,  # 模型
    tools=[internet_search],  # 工具
    system_prompt="""你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 internet_search 工具来收集信息。""",  # 系统提示
)


# 流式执行agent
async def query_astream(query:str):
    async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": query}]}
    ):
        # print(chunk)
        for key,value in chunk.items():
            if key not in ["model","tools"]:
                continue
            # 得到当前message对象
            message = value["messages"][0]
            # 模型处理的返回
            if key == "model":
                tool_calls = message.tool_calls
                # 准备调用工具
                if tool_calls:
                    for tool_call in tool_calls:
                        name = tool_call["name"]
                        # 准备调用子agent (task tool)
                        if name == "task":
                            # "args": {
                            #            "subagent_type": "网络搜索助手",  # 目标子 Agent
                            #            "description": "查询2024政策"     # 下发的具体任务
                            #         },
                            args = tool_call["args"]
                            print(f"-----模型返回，准备调用子智能体[{args['subagent_type']}]，具体的说明：{args['description']}")
                        # 准备调用tool
                        else:
                            # "name": "read_file_content",
                            # "args": {"filename": "需求.docx"},
                            print(f"-----模型返回，准备调用工具[{name}]，参数为：{tool_call['args']}")

                # 返回最终的结果
                else:
                   print(f"-----模型返回最终结果：{message.content}")
            # 工具处理的返回
            elif key == "tools":
                name = message.name
                # 子智能体的返回 task tool
                if name == "task":
                    print(f"-----子智能体返回结果：{message.content}")
                # 工具的返回
                else:
                    print(f"-----工具返回结果：{message.content}")


# 智能体3次执行是串行执行
async def batch_astream():
    await query_astream("今天武汉天气怎么样？")
    await query_astream("今天重庆天气怎么样？")
    await query_astream("今天成都天气怎么样？")

# 智能体3次执行是并行执行
async def batch_astream_parallel():
    cor1 = query_astream("今天武汉天气怎么样？")
    cor2 = query_astream("今天重庆天气怎么样？")
    cor3 = query_astream("今天成都天气怎么样？")
    await asyncio.gather(cor1, cor2, cor3)


if __name__ == '__main__':
    # asyncio.run(batch_astream())
    asyncio.run(batch_astream_parallel())
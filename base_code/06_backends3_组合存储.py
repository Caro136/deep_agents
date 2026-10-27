"""
  测试06_3: 演示使用CompositeBackend实现长期记忆
 	不同路由采用不同的存储方案
"""
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend, FilesystemBackend, CompositeBackend
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
from pathlib import Path

load_dotenv()

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="""你是一个智能助手。
    - 普通文件：直接写入文件名（如 `report.txt`），保存到本地工作区。
    - 重要记忆：写入 `/store/` 目录（如 `/store/profile.txt`），保存到store指定的存储方式中。
    """,
    # store存储
    store=InMemoryStore(),
    # 组合后端
    backend=CompositeBackend(
        # 默认使用文件存储
        default=FilesystemBackend("../backend_dir"),
        # 配置路由对应的存储
        routes={
            # 当前路由路径使用Store存储
            "/store/": StoreBackend(namespace=lambda _rt: ("ns_test",)),
        },
    )
)

# 运行 Agent 测试
config = {"configurable": {"thread_id": "thread_composite"}}
config2 = {"configurable": {"thread_id": "thread_composite2"}}

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """1. 创建文件 local.txt，内容'本地文件内容数据...'。
                      2. 创建文件 /store/memory，内容'重要记忆数据...'。"""
    }]
}, config=config)

print("Agent 回复:", result["messages"][-1].content)

result2 = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """1. 读取 local.txt，内容。
                      2. 读取 /store/memory，内容。"""
    }]
}, config=config2)

print("Agent 回复2:", result2["messages"][-1].content)
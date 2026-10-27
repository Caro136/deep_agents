"""
  测试06_2: 演示使用StoreBackend实现长期记忆
  实现跨会话的数据共享
"""
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend, StateBackend,FilesystemBackend,CompositeBackend
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
import os
load_dotenv()

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 创建一个内存存储对象，可以跨会话，但应用信息，数据丢失
# 以key-value形式存储，key是虚拟文件路径，value是要保存的数据
# 创建用于store存储的对象，当前是内存store存储 =》如果要持久化保存，需要使用特定数据库对应store
store = InMemoryStore()

main_agent = create_deep_agent(
    model=llm,
    system_prompt="""
    你要把用户的重要信息保存到user_profile.txt文件中！
    获取用户信息可以读取user_profile.txt文件！
    """,
    store=store, # 配置数据存储的store
    # 配置store后端
    backend=StoreBackend(
        namespace=lambda runtime:("ns_user",) # 指定存储数据的命名空间
    )
)


config_a  = {"configurable":{"thread_id":"a"}}
config_b  = {"configurable":{"thread_id":"b"}}

#第一遍执行 =>保存一些重要数据
result_a = main_agent.invoke(
    {
        "messages":[{"role":"user","content":"我是大风哥，我今年19岁！"}]
    },
    config=config_a
)
print(f"第1次回复结果：{result_a['messages'][-1].content}")


#第二遍执行 -》 读取一些重要信息
result_b = main_agent.invoke(
    {
        "messages":[{"role":"user","content":"我叫啥和我的年龄！"}]
    },
    config=config_b
)
print(f"第2次回复结果：{result_b['messages'][-1].content}")


# 手动读取store存储的数据
result = store.search(("ns_user",))
print("*" * 10, result)
for item in result:
    print("*" * 10, item.namespace, item.key, item.value)

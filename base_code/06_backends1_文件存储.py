"""
  测试06_1: 演示使用fileSystemBackEnd实现长期记忆
   实现跨会话的数据共享
"""
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from deepagents.backends import FilesystemBackend

load_dotenv()

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


main_agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="""
        你是一个智能助手，可以使用文件工具进行文件读写操作！但是只有在用户明确要求的情况下，你才可以创建文件
    """,
    # root_dir: 所有文件操作的基础路径
    # virtual_mode 默认就是true, 可以不指定
    # virtual_mode=True:
    # - 路径安全隔离, agent不能访问root_dir之外的任何路径
    # - 虚拟路径展示，agent只能看到相对于root_dir的虚拟路径，看不到真实物理路径
    backend=FilesystemBackend(root_dir="../backend_dir", virtual_mode=True)
)

# 执行agent,将数据保存到工作区目录下指定的文件中
result = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "帮我查询下python语言的介绍,并且帮我写到 1.txt文件"}
    ]
})
print(f"最终结果: {result['messages'][-1].content}")

# 再次执行agent, 读取工作区目录下指定文件内容
result2 = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "读取 1.txt文件的内容给我，不要额外处理"}
    ]
})
print(f"最终结果2: {result2['messages'][-1].content}")

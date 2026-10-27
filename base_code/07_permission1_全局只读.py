"""
测试07_1：智能体的整个工作区目录下：只能读文件，不能写（禁止写入）
"""
import os
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(
        root_dir="../backend_dir"
    ),
    # 配置工作区文件的操作权限
    permissions=[
        # 限制工作区所有文件只读
        FilesystemPermission(
            operations=["write"], # 针对写操作
            paths=["/**"], # 针对的路径是工作区目录及其子目录
            mode="deny" # 拒绝/禁止上面的写操作
        )
    ]
)

print("=== 测试：智能体只读权限 ===")

# 测试1：让智能体执行【读取文件】→ 应该允许
result1 = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "读取local.txt文件内容"
    }]
})
print(f"读取结果：{result1['messages'][-1].content}")

# 测试2：让智能体执行【写入文件】→ 应该被拒绝
result2 = agent.invoke({
    "messages": [{
        "role": "user",
        # 这是一个当前只读（read-only）的沙箱环境，没有可用的写权限，所以 `local.txt` 的内容保持原样。
        # 当前该文件系统对写入操作整体被拒绝，不只是 `local.txt` 单个文件的问题（新建文件同样被拒）
        "content": "向local.txt写入: hello world"
    }]
})
print(f"写入结果：{result2['messages'][-1].content}")

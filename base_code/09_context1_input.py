"""
测试09_1：理解各个输入上下文
"""
import os
from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from dotenv import load_dotenv

load_dotenv()

# ======================== 1. 设置 Backend ========================
# 使用 FilesystemBackend 连接到本地文件系统
# 假设 skills 目录在当前脚本同级目录下的 skills/ 中
current_dir = Path(__file__).parent.parent.resolve()
# 我们将 root_dir 设置为当前目录 (base)
# 注意：FilesystemBackend 的 root_dir 是物理路径的根
fs_backend = FilesystemBackend(root_dir=current_dir)

# ======================== 2. 初始化 Agent ========================
llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

agent = create_deep_agent(
    model=llm,
    system_prompt=(
        "你是一名高情商中文助手。"
        "你的回答要自然、细腻、有陪伴感。"
        "当任务适合拆分时，可以交给子代理处理。"
        "如果用户要求写文案、祝福语、安慰话术，优先结合技能完成。"
    ),
    # 文件系统后端
    backend=fs_backend,
    # skill的目录，开始只是加载所有skill的元数据，并添加为系统提示词
    # 后续匹配上某个技能后，会加载这个技能的整个skill的内容添加为系统提示词
    skills=["/skills"],
    # 长期规则和用户偏好相关配置保存到记忆相关文件，初始加载，一直有效
    memory=["/memory/AGENTS.md", "/memory/preferences.md"],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我写一份中秋节节日祝福!"}]}
)
print(f"[Agent]: {result['messages'][-1].content}")

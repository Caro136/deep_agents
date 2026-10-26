"""
  测试4_2: 兼容langchain子智能体格式
"""
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


load_dotenv()

chat_model = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# langchain的agent
# 工具函数。需要指定文档说明，否则会报错ValueError: Function must have a docstring if description not provided.
def get_weather(city:str)->str:
    """查询指定城市天气"""
    print(f"-----调用工具函数，获取{city}的天气")
    return f"{city}今天的天气是晴天"

# langchain的单智能体
chain_agent = create_agent(
    model=chat_model,
    system_prompt="你是一个查询天气的助手，调用工具获取天气数据",
    tools=[get_weather]
)

# 将langchain的智能体包装为deepagents支持的子智能体
sub_agent = CompiledSubAgent(
    name="sub_agent",
    description="你是一个查询天气的子助手",
    runnable=chain_agent
)

# 创建主智能体，设置子智能体
main_agent = create_deep_agent(
    model=chat_model,
    system_prompt="你是一个智能助手，所有的业务动作，你不要亲自做，都找子智能体进行处理！",
    subagents=[sub_agent]
)


print("====开始执行任务====")
result = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "武汉今天的天气怎么样？"}
    ]
})
print(f"最终结果：{result['messages'][-1].content}")
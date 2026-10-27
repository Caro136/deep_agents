import os
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

load_dotenv()

# 自定义Context类型
class MyContext(BaseModel):
    user_id: str
    api_key: str

# 在工具中读取运行时上下文件对象中的数据
@tool
def fetch_user_data(query: str, runtime:ToolRuntime[MyContext]) -> str:
    """
    查询当前用户的数据，可获取用户活动、记录、资料等全部用户相关信息
    Args:
        query: 需要查询的用户信息描述，例如：最近7天活动记录、个人资料等
    """
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key
    print(f"查询用户数据的工具执行. query={query},user_id={user_id}, api_key={api_key}")

    # 根据user_id查询用户的相关数据信息
    user_info = "用户最近7天的活动记录：吃饭睡觉"
    return user_info

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

agent = create_deep_agent(
    model=llm,
    tools=[fetch_user_data],
    context_schema=MyContext, # 指定context的类型
    system_prompt="""你拥有工具 fetch_user_data，该工具专门用于查询用户相关数据、活动记录、个人信息。
        规则严格遵守：
        1. 只要用户提问涉及「我、我的、用户、活动记录、个人数据、我的记录」等用户相关查询，**必须调用 fetch_user_data 工具，禁止直接自行回答**；
        2. 调用工具时，将用户完整需求作为query参数传入；
        3. 不允许跳过工具直接编造用户信息，所有用户相关内容必须通过工具获取；
        4. 工具返回结果后，再整理内容回复给用户。"""
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "查询用户最近7天的活动记录"}]},
    # 指定包含数据的context对象
    context=MyContext(user_id="user-123", api_key="sk-示例"),
)
print(f"[Agent]: {result['messages'][-1].content}")

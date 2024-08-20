from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
from controller.agent_builder import build_supervisor

# Load the .env file
load_dotenv()
# llm = ChatOpenAI(model="gpt-4-turbo")
llm = ChatVertexAI(model_name="gemini-1.5-flash-001")
# llm = GenerativeModel("gemini-1.5-flash-001")


def sql_generator_agent(human_input: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a sql generator,"
                    "Return the answer directly without explanation, like 'SELECT * FROM student;'"
                )
            ),
            HumanMessage(content=human_input)
        ]
    )
    messages = prompt.format_messages()
    return llm.invoke(messages)



# members = {
#     "Researcher": {
#         "llm": llm,
#         "tools": [TavilySearchResults(max_results=5)],
#         "prompt": "You are a web researcher."
#     },
#     "SqlGenerator": {
#         "llm": llm,
#         "tools": [],
#         "prompt": "You are a SQL generator."
#     }
# }
#
# supervisor = build_supervisor(llm, members)
#
# user_inputs = {"messages": [HumanMessage(content="请帮我生成一个创建student表的语句，这个表只有两个字段，id，name")]}
#
#
# def execute_agent():
#     for s in supervisor.stream(user_inputs):
#         if "__end__" not in s:
#             print(s)
#             print("----")
#
#
# if __name__ == "__main__":
#     execute_agent()

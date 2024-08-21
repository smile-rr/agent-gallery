from typing import Literal, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, InjectedState

load_dotenv(".env.local")
llm = ChatVertexAI(model_name="gemini-1.5-flash-001", temperature=0)

tool_name_mapping = {
    "sql_generator": "Sql Generator",
    "get_coolest_cities": "Coolest Cities",
    "agent_gallery": "Agent Gallery"
}


def sql_generator_agent(human_input: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a sql generator,"
                    "Return the answer directly, do not explain, and do not give any hints other than the answer."
                )
            ),
            HumanMessage(content=human_input)
        ]
    )
    messages = prompt.format_messages()
    return llm.invoke(messages)


@tool
def sql_generator(state: Annotated[dict, InjectedState]):
    """Call to get sql"""
    messages = state["messages"]
    for message in messages:
        if isinstance(message, HumanMessage):
            return sql_generator_agent(message.content)


@tool
def get_coolest_cities(state: Annotated[dict, InjectedState]):
    """Get a list of coolest cities"""
    return "nyc, sf"


def agent_supervisor_run(user_input: str):
    tools = [sql_generator, get_coolest_cities]
    tool_node = ToolNode(tools)

    model_with_tools = llm.bind_tools(tools)

    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge("__start__", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    messages_maps = {
        "messages": [("system",
                      "You are a agent supervisor, identify the user's intent, "
                      "and call the tool related to the user's intent."
                      # "If we don't recognize what tool the user wants to use, "
                      # "we will list the tooltips we already have for the user to see. For example, "
                      # "we provide the following agent for you to choose from:"
                      # "1. sql generator"
                      # "2. coolest city"
                      # "Must be displayed to the user in a list"
                      ),
                     ("human", user_input)]
    }

    all_response_messages = []
    for chunk in app.stream(messages_maps, stream_mode="values"):
        all_response_messages = chunk["messages"]

    for message in all_response_messages:
        if isinstance(message, ToolMessage):
            return message.name, all_response_messages[-1].content
    return None, all_response_messages[-1].content
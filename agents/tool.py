from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI

from langgraph.prebuilt import ToolNode, InjectedState
from typing import Literal, Annotated
from langgraph.graph import StateGraph, MessagesState

from controller.agent_controller import sql_generator_agent

load_dotenv()


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

    model_with_tools = ChatVertexAI(
        model="gemini-1.5-flash", temperature=0
    ).bind_tools(tools)

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
                      "You are a task scheduling administrator, identify the user's intent, "
                      "and call my tools related to the user's intent."
                      "When the user's input does not match the corresponding tool, please reply to the user "
                      ),
                     ("human", user_input)]
    }

    for chunk in app.stream(messages_maps, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

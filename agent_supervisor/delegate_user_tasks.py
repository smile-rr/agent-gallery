from typing import Literal, Annotated
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, InjectedState
from doctor_happy.doctor_happy_chat import HappyChat
from snap_reader.image_to_text import textExtractionAssistant
import os

load_dotenv()

def vertex_client():
        # Initialize Vertex AI
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION")
        model_name = os.getenv("MODEL_NAME")
        return ChatVertexAI(
            model_name=model_name,
            project=project_id,
            location=location,
            temperature=0,
            max_output_tokens=256,
            top_p=0.8,
            top_k=40,
            max_retries=3,
        )

llm = vertex_client()
console = Console()

tool_name_mapping = {
    "load_content_from_confluence": "Sage",
    "image_text_extraction_assistant": "Snap Reader",
    "get_userinfo_from_workday": "Smart Bookmarts Assistant",
    "generate_ai_prompts": "Prompts Expert",
    "load_content_from_jira": "Jira Agent",
    "wukong": "Wukong",
}

def load_content_from_jira_agent(human_input: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Return jira-related information to the user"
                    "Return the answer directly, do not explain, and do not give any hints other than the answer."
                )
            ),
            HumanMessage(content=human_input)
        ]
    )
    messages = prompt.format_messages()
    return llm.invoke(messages)

def generate_ai_prompts_agent(human_input: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Generate professional AI prompts for users according to their needs. If you cannot provide them, return some professional knowledge of the prompts to users."
                    "Return the answer directly, do not explain, and do not give any hints other than the answer."
                )
            ),
            HumanMessage(content=human_input)
        ]
    )
    messages = prompt.format_messages()
    return llm.invoke(messages)

def get_userinfo_from_workday_agent(human_input: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Get some information from workday"
                    "Return the answer directly, do not explain, and do not give any hints other than the answer."
                )
            ),
            HumanMessage(content=human_input)
        ]
    )
    messages = prompt.format_messages()
    return llm.invoke(messages)

@tool
def load_content_from_jira(state: Annotated[dict, InjectedState]):
    """Return jira-related information to the user"""
    messages = state["messages"]
    for message in messages:
        if isinstance(message, HumanMessage):
            return load_content_from_jira_agent(message.content)

@tool
def generate_ai_prompts(state: Annotated[dict, InjectedState]):
    """Generate professional AI prompts for users according to their needs. If you cannot provide them, return some professional knowledge of the prompts to users."""
    messages = state["messages"]
    for message in messages:
        if isinstance(message, HumanMessage):
            return generate_ai_prompts_agent(message.content)


@tool
def load_content_from_confluence(state: Annotated[dict, InjectedState]):
    """Searches content from Confluence, loads content from the found pages, and generates a response."""
    messages = state["messages"]
    for message in messages:
        if isinstance(message, HumanMessage):
            happy_chat = HappyChat()
            return happy_chat.process_user_input(message.content)
        
@tool
def image_text_extraction_assistant(state: Annotated[dict, InjectedState]):
    """You are an image text recognition assistant, you can get the raw text from given immage. The recognized text needs to be exactly the same as the text in the image."""
    messages = state["messages"]
    for message in messages:
        if isinstance(message, HumanMessage):
            return textExtractionAssistant(message.content)

@tool
def get_userinfo_from_workday(state: Annotated[dict, InjectedState]):
    """Get some information from workday"""
    messages = state["messages"]
    for message in messages:
        if isinstance(message, HumanMessage):
            return get_userinfo_from_workday_agent(message.content)


def agent_supervisor_run(user_input: str):
    tools = [load_content_from_confluence,image_text_extraction_assistant,get_userinfo_from_workday,generate_ai_prompts,load_content_from_jira]
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
    with Live(Spinner("pong"), transient=True, refresh_per_second=10):
        try:
           
            for chunk in app.stream(messages_maps, stream_mode="values"):
                 all_response_messages = chunk["messages"]

            for message in all_response_messages:
                if isinstance(message, ToolMessage):
                    return message.name, all_response_messages[-1].content
            return None, all_response_messages[-1].content

        except Exception as e:
            console.print(f"[red]Error: {e}")
            return "Oops! Something went wrong. Please try again."

    

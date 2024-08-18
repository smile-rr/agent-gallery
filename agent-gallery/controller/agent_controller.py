from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from controller.agent_builder import build_supervisor

load_dotenv()

llm = ChatOpenAI(model="gpt-4-turbo")
# llm = ChatVertexAI(model="gemini-1.5-flash-001")

members = {
    "Researcher": {
        "llm": llm,
        "tools": [TavilySearchResults(max_results=5)],
        "prompt": "You are a web researcher.",
        "description": "Web Page researcher"
    },
    "SqlGenerator": {
        "llm": llm,
        "tools": [],
        "prompt": "You are a SQL generator.",
        "description": "SQL generator"
    }
}

supervisor = build_supervisor(llm, members)




def execute_agent(text):
    user_inputs = {"messages": [HumanMessage(content=text)]}
    for s in supervisor.stream(user_inputs):
        if "__end__" not in s:
            print(s)
            print("----")


def start():
    #print welcome message and list all members , all message use English
    print("Welcome to the agent gallery, we have below agents provide service for you :")
    print("--------------------------------------------------------------------------------------------------------")
    for name, member in members.items():
        print(f"{name}: {member['description']}")
    print("--------------------------------------------------------------------------------------------------------")
    print("\n")

    while True:
        text = input("Pls input your question：")
        if "exit" in text:
            break
            # exit
        execute_agent(text)
        print("\n")


if __name__ == "__main__":
    # Load the .env file
    load_dotenv()
    start()

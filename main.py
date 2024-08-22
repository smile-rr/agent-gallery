import time
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
import os
from agent_supervisor.delegate_user_tasks import tool_name_mapping, agent_supervisor_run

CHATBOT_COLOR = "green"
USER_COLOR = "blue"
console = Console()
load_dotenv()
llm = ChatVertexAI(model_name="gemini-1.5-flash-001")


class ConsoleManager:
    # Define colors for the chatbot and user

    def get_user_input(self, prompt: str) -> str:
        """Prompts the user for input and returns the input in blue color."""
        user_input = console.input(f"[{USER_COLOR}]{prompt}")
        return user_input

    def format_message(self, name: str, message: str, color: str) -> str:
        """Formats the message with the name and color."""
        return f"\n[{color}]{name}:[/] {message}"

    def clear_console(self):
        """Clears the console."""
        os.system('cls' if os.name == 'nt' else 'clear')


console_manager = ConsoleManager()

def get_user_input(prompt: str) -> str:
    """Prompts the user for input and returns the input in blue color."""
    print(f"[{USER_COLOR}]{prompt}", end="")
    return input()


def format_message(name: str, message: str, color: str) -> str:
    """Formats the message with the name and color."""
    # Remove the newline before the message to reduce spacing
    return f"\n[{color}]{name}:[/][{color}]{message}[/]"


def call_agent_supervisor(user_input: str):
    return agent_supervisor_run(user_input)


def get_chat_response(message: str) -> str:
    """Gets the chat response from the chatbot."""
    with Live(Spinner("pong"), transient=True, refresh_per_second=10):
        try:
            # response = conversation.invoke(input=message, config={"configurable": {"session_id": "1"}}, )
            response = llm.invoke(message)
            return response.content
        except Exception as e:
            console.print(f"[red]Error: {e}")
            return "Oops! Something went wrong. Please try again."


def display_initial_greeting():
    """Displays the initial greeting message."""
    initial_system_prompt = """
    You are a chat assistant named as 'Happy'. from next input you will get the chat message from user. 
    Firstly give a cool welcome message to user to start the interesting conversation. Please be brief.
    """

    # Send the system message
    initial_response = get_chat_response(initial_system_prompt)

    # Print the initial response if needed
    console.print(console_manager.format_message("Happy", initial_response, CHATBOT_COLOR))


if __name__ == '__main__':

    display_initial_greeting()
    while True:
        user_input = get_user_input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        tool_name, ai_message = call_agent_supervisor(user_input)
        if tool_name not in tool_name_mapping:
            tool_name = "agent_gallery"
            ai_message = """I am an agent gallery provided the following agent for you:
    1. Sql generator
    2. Coolest city
Please type in your needs directly.
I will automatically recognize your intention.
And help you complete the task.
                                 """
        print(format_message(tool_name_mapping[tool_name], "\n" + ai_message, CHATBOT_COLOR))
        # Sleep for 100 milliseconds
        time.sleep(0.1)

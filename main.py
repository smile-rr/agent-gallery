import time
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
import os
from agent_supervisor.delegate_user_tasks import tool_name_mapping,vertex_client, agent_supervisor_run

CHATBOT_COLOR = "green"
USER_COLOR = "blue"
console = Console()
load_dotenv()

llm = vertex_client()

# llm = ChatVertexAI(model_name="gemini-1.5-flash-001")


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
            response = llm.invoke(message)
            return response.content
        except Exception as e:
            console.print(f"[red]Error: {e}")
            return "Oops! Something went wrong. Please try again."


def display_initial_greeting():
    """Displays the initial greeting message."""
    agents_name = tool_name_mapping.values()
    initial_system_prompt = """
    You are a chat assistant named as 'Wukong'. from next input you will get the chat message from user. 
    Firstly give a cool welcome message to user to start the interesting conversation related these agents{agents}. Please be brief.
    """

    # Send the system message
    initial_response = get_chat_response(initial_system_prompt.format(agents=", ".join(get_agents_name_list())))

    # Print the initial response if needed
    console.print(console_manager.format_message("Wukong", initial_response, CHATBOT_COLOR))


def get_agents_name_list():
    """Returns the list of agent names."""
    return [value for value in tool_name_mapping.values() if value != "Wukong"]


if __name__ == '__main__':

    display_initial_greeting()
    while True:
        user_input = console_manager.get_user_input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        tool_name, ai_message = call_agent_supervisor(user_input)
        if tool_name not in tool_name_mapping:
            tool_name = "wukong"
            agents_name = tool_name_mapping.values()
            ai_message = llm.invoke("Sorry, I don't understand. Please try again or choose from the following agents: " + ", ".join(get_agents_name_list())).content
        console.print(console_manager.format_message(tool_name_mapping[tool_name], ai_message+"\n", CHATBOT_COLOR))
        # Sleep for 100 milliseconds
        time.sleep(0.1)

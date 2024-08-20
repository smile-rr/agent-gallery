import time
from rich import print

from agents.agent_supervisor import agent_supervisor_run, tool_name_mapping

CHATBOT_COLOR = "green"
USER_COLOR = "blue"


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


if __name__ == '__main__':
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
Please enter your needs directly.
I will automatically recognize your intention.
And help you complete the task.
                         """
        print(format_message(tool_name_mapping[tool_name], "\n" + ai_message, CHATBOT_COLOR))
        # Sleep for 100 milliseconds
        time.sleep(0.1)

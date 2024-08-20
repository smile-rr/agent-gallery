import time

from rich import print

from agents.tool import agent_supervisor_run

CHATBOT_COLOR = "green"
USER_COLOR = "blue"


def get_user_input(prompt: str) -> str:
    """Prompts the user for input and returns the input in blue color."""
    print(f"[{USER_COLOR}]{prompt}", end="")
    return input()


def format_message(name: str, message: str, color: str) -> str:
    """Formats the message with the name and color."""
    # Remove the newline before the message to reduce spacing
    return f"\n[{color}]{name}:[/] {message}"


def call_agent_supervisor(user_input: str):
    return agent_supervisor_run(user_input)


if __name__ == '__main__':
    while True:
        user_input = get_user_input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        ai_message = call_agent_supervisor(user_input)
        print(format_message("", ai_message, CHATBOT_COLOR))
        # Sleep for 100 milliseconds
        time.sleep(0.1)

import os
from dotenv import load_dotenv
from rich import print
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live
import vertexai

from vertexai.generative_models import GenerativeModel, ChatSession

load_dotenv(".env.local")

# Initialize Vertex AI
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
model_name = os.getenv("MODEL_NAME")

vertexai.init(project=project_id, location=location)

model = GenerativeModel("gemini-1.5-flash-001")

chat = model.start_chat()

# Define colors for the chatbot and user
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

def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

def print_markdown_response_with_spinner(prompt: str) -> str:
    with Live(Spinner("pong"), transient=True, refresh_per_second=10):
        response = get_chat_response(chat, prompt)
    # Convert the response to Markdown and return it as a string
    return response

# Initial greeting from Vertex AI
initial_prompt = """
You are a chat assistant named as 'Happy'. from next input you will get the chat message from user. 
firstly give a cool welcome message to user to start the interesting conversation. pls be in short.
"""
initial_response = print_markdown_response_with_spinner(initial_prompt)

# Print the initial response if needed
print(format_message("Happy", initial_response, CHATBOT_COLOR))

# Continuously prompt the user for input
while True:
    user_input = get_user_input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = print_markdown_response_with_spinner(user_input)
    
    # print(format_message("You", user_input, USER_COLOR))
    print(format_message("Happy", response, CHATBOT_COLOR))
    
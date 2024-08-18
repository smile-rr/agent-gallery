import os
from dotenv import load_dotenv
from rich import print
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI

# Define colors for the chatbot and user
CHATBOT_COLOR = "green"
USER_COLOR = "blue"

# Create a rich console instance
console = Console()

class HappyChat:
    def __init__(self):
        # Initialize Vertex AI
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION")
        model_name = os.getenv("MODEL_NAME")
        self.chat = ChatVertexAI(
            model_name=model_name,
            project=project_id,
            location=location,
            temperature=0,
            max_output_tokens=256,
            top_p=0.8,
            top_k=40
        )

    def get_user_input(self, prompt: str) -> str:
        """Prompts the user for input and returns the input in blue color."""
        # Use Console().input to keep the prompt and input on the same line
        return console.input(f"[{USER_COLOR}]{prompt}")

    def format_message(self, name: str, message: str, color: str) -> str:
        """Formats the message with the name and color."""
        # Remove the newline before the message to reduce spacing
        return f"\n[{color}]{name}:[/] {message}"

    def get_chat_response(self, messages: list) -> str:
        """Gets the chat response from the chatbot."""
        try:
            response = self.chat.invoke(messages)
            return response.content
        except Exception as e:
            console.print(f"[red]Error: {e}")
            return "Oops! Something went wrong. Please try again."

    def get_response_with_spinner(self, messages: list) -> str:
        """Prints the chatbot's response with a spinner."""
        with Live(Spinner("pong"), transient=True, refresh_per_second=10):
            response = self.get_chat_response(messages)
        return response

    def run(self):
        """Starts the chat loop."""
        # Initial greeting from Vertex AI
        initial_system_prompt = """
        You are a chat assistant named as 'Happy'. from next input you will get the chat message from user. 
        Firstly give a cool welcome message to user to start the interesting conversation. Please be brief.
        """
        
        # Send the system message
        initial_response = self.get_response_with_spinner([HumanMessage(content=initial_system_prompt)])

        # Print the initial response if needed
        console.print(self.format_message("Happy", initial_response, CHATBOT_COLOR))

        # Continuously prompt the user for input
        while True:
            user_input = self.get_user_input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            # Send the user input as a human message
            response = self.get_response_with_spinner([HumanMessage(content=user_input)])
            console.print(self.format_message("Happy", response, CHATBOT_COLOR))

if __name__ == "__main__":
    load_dotenv(".env.local")
    happy_chat = HappyChat()
    happy_chat.run()
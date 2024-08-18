import os
import re
from dotenv import load_dotenv
from rich import print
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts.prompt import PromptTemplate

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
        user_input = console.input(f"[{USER_COLOR}]{prompt}")

        # # Check if the input contains a URL
        # url_match = re.search(r"(https?://[^\s]+)", user_input)
        # if url_match:
        #     url = url_match.group(1)
        #     # Call the method to get a response based on the URL content
        #     return self.get_response_based_on_url(url, user_input)

        return user_input

    def format_message(self, name: str, message: str, color: str) -> str:
        """Formats the message with the name and color."""
        # Remove the newline before the message to reduce spacing
        return f"\n[{color}]{name}:[/] {message}"

    def get_chat_response(self, messages: list) -> str:
        """Gets the chat response from the chatbot."""
        with Live(Spinner("pong"), transient=True, refresh_per_second=10):
            try:
                response = self.chat.invoke(messages)
                return response.content
            except Exception as e:
                console.print(f"[red]Error: {e}")
                return "Oops! Something went wrong. Please try again."

    def get_response_based_on_url(self, url: str, user_input: str) -> str:
        """Fetches content from a given URL using WebBaseLoader, extracts key information, and generates a response based on that content."""
        try:
            # Load the content from the URL
            loader = WebBaseLoader(url)
            data = loader.load()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(data)

            # Create embeddings for the documents
            embeddings = VertexAIEmbeddings()

            # Create a vector store from the documents
            db = FAISS.from_documents(docs, embeddings)

            # Create a RetrievalQA chain
            qa = RetrievalQA.from_chain_type(
                llm=self.chat,
                chain_type="stuff",
                retriever=db.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                        input_variables=["context", "question"],
                    )
                }
            )

            # Get the answer from the chain
            result = qa({"query": user_input})

            # Extract the answer and source documents
            answer = result["result"]
            source_docs = result["source_documents"]

            # Format the answer and sources
            formatted_answer = f"{answer}\n\nSources:"
            for doc in source_docs:
                formatted_answer += f"\n- {doc.metadata['source']}"

            return formatted_answer
        except Exception as e:
            console.print(f"[red]An error occurred: {e}")
            return "Oops! Something went wrong. Please try again."

    def run(self):
        """Starts the chat loop."""
        # Initial greeting from Vertex AI
        initial_system_prompt = """
        You are a chat assistant named as 'Happy'. from next input you will get the chat message from user. 
        Firstly give a cool welcome message to user to start the interesting conversation. Please be brief.
        """
        
        # Send the system message
        initial_response = self.get_chat_response([("human",initial_system_prompt)])

        # Print the initial response if needed
        console.print(self.format_message("Happy", initial_response, CHATBOT_COLOR))
        
        try:    
            # Continuously prompt the user for input
            while True:
                user_input = self.get_user_input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    console.print("\n[red]Bye bye...")
                    break
                # Send the user input as a human message
                response = self.get_chat_response([("human",user_input)])
                console.print(self.format_message("Happy", response, CHATBOT_COLOR))
        except KeyboardInterrupt:
            console.print("\n\n[red]Bye bye ...")

if __name__ == "__main__":
    load_dotenv(".env.local")
    happy_chat = HappyChat()
    happy_chat.run()
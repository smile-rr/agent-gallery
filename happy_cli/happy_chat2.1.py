import os
import re
import json
from dotenv import load_dotenv
from rich import print
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain.docstore.document import Document

from langchain_google_vertexai import VertexAI
from langchain_google_community import VertexAISearchRetriever
from langchain_google_community import VertexAIMultiTurnSearchRetriever
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Dict, Any
import requests

from confluence_client import FUNCTION_DESCRIPTIONS, ConfluenceClient, ConfluenceSearchInput, ConfluenceSearchOutput

CHATBOT_COLOR = "green"
USER_COLOR = "blue"
console = Console()

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


class KeywordSuggester:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def suggest_keywords(self, user_input: str) -> list:
        """Asks the AI model to suggest keywords based on the user's input in JSON format."""
        # Prepare the prompt to ask for keywords in JSON format
        prompt_template = PromptTemplate(
            template="Suggest up to 5 keywords for the following query: '{query}'. Respond in JSON format: {{'keywords': ['keyword1', 'keyword2']}}.",
            input_variables=["query"]
        )

        # Format the prompt with the user's input
        prompt = prompt_template.format(query=user_input)

        # Get the AI model's response
        response = self.chatbot.get_chat_response(prompt)

        # Parse the JSON response
        keywords = self.parse_keywords_from_json_response(response)

        return keywords

    def parse_keywords_from_json_response(self, json_response: str) -> list:
        """Parses the JSON response from the AI model and returns a list of keywords."""
        try:
            # Parse the JSON response
            data = json.loads(json_response)
            keywords = data.get('keywords', [])
            return keywords
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON response from the AI model.")
            return []


class DocumentProcessor:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def load_and_split_documents(self, contents: List[str]) -> List[Document]:
        """Splits Confluence content into documents.
        
        Args:
            contents (List[str]): A list of strings representing Confluence content.
        
        Returns:
            List[Document]: A list of processed documents.
        """
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        for content in contents:
            # Split the document
            docs = text_splitter.split_text(content)
            
            # Add metadata (if needed)
            for i, doc in enumerate(docs):
                metadata = {
                    "source": f"confluence_{i}",
                    "chunk": i,
                }
                docs[i] = Document(page_content=doc, metadata=metadata)
            
            # Append to all_docs
            all_docs.extend(docs)

        return all_docs

    def create_vector_store(self, docs: list) -> FAISS:
        """Creates a vector store from the provided documents."""
        embeddings = VertexAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        return db

    def generate_answer(self, db: FAISS, user_input: str) -> str:
        """Generates an answer using the provided vector store and user input."""
        qa = RetrievalQA.from_chain_type(
            llm=self.chatbot.vertex_ai,
            chain_type="map_reduce",
            retriever=db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                    input_variables=["context", "question"],
                )
            }
        )

        result = qa({"query": user_input})

        answer = result["result"]
        source_docs = result["source_documents"]

        # Format the answer and sources
        formatted_answer = f"{answer}\n\nSources:"
        for doc in source_docs:
            formatted_answer += f"\n- {doc.metadata['source']}"

        return formatted_answer


class HappyChat:
    def __init__(self):
        # Initialize Vertex AI
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION")
        model_name = os.getenv("MODEL_NAME")
        self.confluence_search_function = RunnableLambda(
            lambda keywords: self.confluence_client.search(keywords)
        )
        self.vertex_ai = ChatVertexAI(
            model_name=model_name,
            project=project_id,
            location=location,
            function_descriptions=FUNCTION_DESCRIPTIONS,
        )
        # Initialize conversation chain
        self.conversation = RunnableWithMessageHistory(
            self.vertex_ai,
            get_session_history=self.get_session_history,
            verbose=False
        )
        self.confluence_client= ConfluenceClient()
        self.document_processor = DocumentProcessor(self)
        self.console_manager = ConsoleManager()
        self.store = {}  # memory is maintained outside the chain
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Returns a new instance of InMemoryChatMessageHistory for the given session ID."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    def get_chat_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        response = self.vertex_ai.invoke(messages)
        if "function_call" in response:
            # Handle function call
            function_name = response["function_call"]["name"]
            function_args = json.loads(response["function_call"]["arguments"])
            # Call the corresponding function
            function_result = self.call_function(function_name, function_args)
            # Invoke the model again with the function result
            final_response = self.vertex_ai.get_chat_response([
                {"role": "function", "content": f"{{'function_name': '{function_name}', 'result': {json.dumps(function_result)}}}"},
                *messages
            ])
            return final_response
        else:
            return response
    def display_initial_greeting(self):
        """Displays the initial greeting message."""
        initial_system_prompt = """
        You are a chat assistant named as 'Happy'. from next input you will get the chat message from user. 
        Firstly give a cool welcome message to user to start the interesting conversation. Please be brief.
        """
        
        # Send the system message
        initial_response = self.get_chat_response(initial_system_prompt)

        # Print the initial response if needed
        console.print(self.console_manager.format_message("Happy", initial_response, CHATBOT_COLOR))

    def run(self):
        """Starts the chat loop."""
        # Initial greeting from Vertex AI
        self.display_initial_greeting()

        try:
            # Continuously prompt the user for input
            while True:
                user_input = self.console_manager.get_user_input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    console.print("\n[red]Bye bye...")
                    break
                elif user_input.lower() in ["cls", "clear"]:
                    # Clear the console
                    self.console_manager.clear_console()
                else:
                    # Process the user input
                    self.process_user_input(user_input)
        except KeyboardInterrupt:
            console.print("\n\n[red]Bye bye ...")

    def process_user_input(self, user_input: str) -> None:
        response = self.function_call_confluence(user_input)
        print("Response:", response)
        # Step 1: Load and split documents
        loaded_documents = self.document_processor.load_and_split_documents(response)

        # Step 2: Create a vector store from the loaded documents
        vector_store = self.document_processor.create_vector_store(loaded_documents)

        # Step 3: Generate an answer using the vector store and user input
        answer = self.document_processor.generate_answer(vector_store, user_input)
        
    def function_call_confluence(self, user_input: str):
        tools = [
        StructuredTool.from_function(
            self.confluence_client.search_by_title_and_content,
            name="search_confluence",
            description="search confluence content based on keywords list from confluence page title and content"
        ),
    ]

    # Modify the prompt to ask for input parameters
    prompt = PromptTemplate(
        template="Suggest the input parameters for the function '{tool_name}' based on the following input: {input}",
        input_variables=["input", "tool_name"],
    )

    agent = create_structured_chat_agent(self.vertex_ai, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    # Invoke the agent executor with the modified prompt
    result = agent_executor.invoke({"input": user_input})

    # Extract the suggested input parameters
    suggested_input = None

    # Check the final output for the suggested input parameters
    output = result.get("output")
    if output:
        # Assuming the output format is "{input_parameters}"
        suggested_input = output.strip().strip('"')

    # Print or use the suggested input parameters
    print(f"Suggested Input Parameters: {suggested_input}")


if __name__ == "__main__":
    load_dotenv(".env.local")
    happy_chat = HappyChat()
    happy_chat.run()
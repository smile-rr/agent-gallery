import os
from dotenv import load_dotenv
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_google_vertexai import VertexAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import FunctionCallingToolkit
from vertexai.generative_models import FunctionDeclaration, Tool

load_dotenv(".env.local")

# Initialize Vertex AI
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
model_name = os.getenv("MODEL_NAME")

# Define Functionality
get_product_sku = FunctionDeclaration(
    name="get_product_sku",
    description="Get the SKU for a product",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "description": "Product name"}
        },
    },
)

get_store_location = FunctionDeclaration(
    name="get_store_location",
    description="Get the location of the closest store",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "Location"}},
    },
)

# Define Tools
retail_tool = Tool(
    function_declarations=[
        get_product_sku,
        get_store_location,
    ],
)

# Initialize the LLM
llm = VertexAI(
    model_name=model_name,
    temperature=0,
    tools=[retail_tool],
)

# Create a toolkit with the defined functions
toolkit = FunctionCallingToolkit.from_tools([get_product_sku(), get_store_location()])

# Initialize the agent with the toolkit
agent = initialize_agent(
    [toolkit.tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Send a prompt for the first conversation turn that should invoke the get_product_sku function
response = agent.run("Do you have the Pixel 8 Pro in stock?")
print(f"Response: {response}")

# Send a prompt for the second conversation turn that should invoke the get_store_location function
response = agent.run("Is there a store in Mountain View, CA that I can visit to try it out?")
print(f"Response: {response}")
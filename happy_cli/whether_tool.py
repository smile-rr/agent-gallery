import os
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.console import Console

load_dotenv(".env.local")

# Initialize Vertex AI
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
model_name = os.getenv("MODEL_NAME")

vertexai.init(project=project_id, location="us-central1")

# Initialize Gemini model
model = GenerativeModel("gemini-1.5-flash-001")

# Define the user's prompt in a Content object that we can reuse in model calls
user_prompt_content = Content(
    role="user",
    parts=[
        Part.from_text("What is the weather like in Boston?"),
    ],
)

# Specify a function declaration and parameters for an API request
function_name = "get_current_weather"
get_current_weather_func = FunctionDeclaration(
    name=function_name,
    description="Get the current weather in a given location",
    # Function parameters are specified in OpenAPI JSON schema format
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "Location"}},
    },
)

# Define a tool that includes the above get_current_weather_func
weather_tool = Tool(
    function_declarations=[get_current_weather_func],
)

# Send the prompt and instruct the model to generate content using the Tool that you just created
response = model.generate_content(
    user_prompt_content,
    generation_config=GenerationConfig(temperature=0),
    tools=[weather_tool],
)
function_call = response.candidates[0].function_calls[0]

# Display the function call with colors
console = Console()
function_call_text = Text(str(function_call), style="bold yellow")
console.print(Panel(function_call_text, title="Function Call"))

# Check the function name that the model responded with, and make an API call to an external system
if function_call.name == function_name:
    # Extract the arguments to use in your API call
    location = function_call.args["location"]  # noqa: F841

    # Here you can use your preferred method to make an API request to fetch the current weather, for example:
    # api_response = requests.post(weather_api_url, data={"location": location})

    # In this example, we'll use synthetic data to simulate a response payload from an external API
    api_response = """{ "location": "Boston, MA", "temperature": 38, "description": "Partly Cloudy",
                    "icon": "partly-cloudy", "humidity": 65, "wind": { "speed": 10, "direction": "NW" } }"""

    # Display the API response with colors
    api_response_text = Text(api_response, style="bold green")
    console.print(Panel(api_response_text, title="API Response"))

# Return the API response to Gemini so it can generate a model response or request another function call
response = model.generate_content(
    [
        user_prompt_content,  # User prompt
        response.candidates[0].content,  # Function call response
        Content(
            parts=[
                Part.from_function_response(
                    name=function_name,
                    response={
                        "content": api_response,  # Return the API response to Gemini
                    },
                ),
            ],
        ),
    ],
    tools=[weather_tool],
)

# Get the model response
print(response.text)
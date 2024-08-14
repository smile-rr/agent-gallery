import vertexai

from vertexai.generative_models import GenerativeModel

project_id = "lively-hull-431714-k3"
location = "us-central1"
# model_name = "gemini-1.5-flash-001"
model_name = "gemini-experimental"
# model_name="gemini-1.5-pro-001"
# model_name="gemini-1.0-pro-vision-001"

vertexai.init(project=project_id, location=location)

model = GenerativeModel(
    model_name=model_name,
    system_instruction=[
        "You are a helpful language translator.",
        "Your mission is to translate text in English to French.",
    ],
)

prompt = """
User input: I like bagels.
Answer:
"""

contents = [prompt]

response = model.generate_content(contents)
print(response.text)
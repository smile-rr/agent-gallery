import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text

load_dotenv()

# 初始化 Vertex AI 客户端
client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

# 模型 ID
model_id = "text-bison@001"

# 设置请求参数
parameters = {}
instances = [{"content": "Write a friendly welcome message."}]

# 发送请求
response = client.predict(
    endpoint=f"projects/your-project-id/locations/us-central1/endpoints/{model_id}",
    instances=instances,
    parameters=parameters,
)

# 解析响应
generated_text = response.predictions[0]["content"]

# 使用 rich 打印欢迎词
welcome_message = Text(generated_text, style="bold green")
rprint(Panel(welcome_message, title="[bold blue]Welcome to Happy!", padding=(1, 2)))

# 主循环，与用户交互
while True:
    user_input = input("What can I do for you today? ")
    if user_input.lower() == "exit":
        break
    # 在这里可以添加更多的逻辑，比如调用 Vertex AI 生成更多文本
    rprint(f"You said: [bold]{user_input}[/bold]")
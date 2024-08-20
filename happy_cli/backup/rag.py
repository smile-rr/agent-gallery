import os
from dotenv import load_dotenv
from langchain_community.embeddings import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rich import print
import vertexai
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import VertexAI
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(".env.local")

# Initialize Vertex AI
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
model_name = os.getenv("MODEL_NAME")

vertexai.init(project=project_id, location=location)

# Initialize the generative model
llm = VertexAI(model_name=model_name, max_output_tokens=256, temperature=0.2, top_p=0.8, top_k=40)

# Initialize Chroma vector store
embedding = VertexAIEmbeddings()
vectorstore = Chroma(persist_directory="db", embedding_function=embedding)

# Prepare documents
documents = []
for i in range(1, 1):
    loader = TextLoader(f"doc{i}.md")  # Assume doc1.txt, doc2.txt, doc3.txt exist
    documents.extend(loader.load())

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Add documents to Chroma
vectorstore.add_documents(docs)

# Create RAG chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)

# User interaction
while True:
    query = input("Enter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    
    result = qa({"question": query})
    print(result["answer"])
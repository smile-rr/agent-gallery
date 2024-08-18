from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# 1.把在线文档灌入向量数据库之后，就可以用retriever来检索了
web_loader = WebBaseLoader("http://hngy.hunancourt.gov.cn/article/detail/2024/01/id/7745909.shtml")
retriever = FAISS.from_documents(RecursiveCharacterTextSplitter().split_documents(web_loader.load()),
                                 OpenAIEmbeddings()).as_retriever()

# 2.创建一个document chain，用于处理检索到的文档
prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:
<context>{context}</context>

Question: {input}
""")
document_chain = create_stuff_documents_chain(ChatOpenAI(), prompt_template)

# 3.创建一个retrieval chain，用于检索文档
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "案子中的赔偿金是多少，怎么计算出来的"})

print(response["answer"])


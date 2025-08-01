# Loaders and text splitters
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and vector store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# LLM and RAG chain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()
#envfilevalue = dotenv_values().get("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -%(message)s')
#logging.debug()

loader = PyPDFLoader("./data/The_Complete_Guide_to_Building_AI_Agents_From_Zero_to_Production-4.pdf")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=100 )
chunks = text_splitter.split_documents(docs)


embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(chunks, embedding_model)


retriever = vector_store.as_retriever()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
qachain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents= True)

query = "What is an agent?"

response = qachain.invoke({"query": query})

logging.debug (f"The formatted response is {json.dumps(response, indent=2, default=str)}")
logging.info(f" The response is {response["result"]}")
for index,source in enumerate(response["source_documents"]):
    title = source.metadata.get("title", "Unknown Title")
    page = source.metadata.get("page_label", "N/A")
    details = source.page_content.replace("\n", " ")
    logging.debug(f"Link {index+1} details is ")
    logging.debug (f"\tPageDetails is {page}") 
    logging.debug (f"\tBrief source details is {details[:100]} ")




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

load_dotenv()
#envfilevalue = dotenv_values().get("OPENAI_API_KEY")

loader = PyPDFLoader("./data/The_Complete_Guide_to_Building_AI_Agents_From_Zero_to_Production-4.pdf")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=100 )
chunks = text_splitter.split_documents(docs)

#print (f" Split into chunk {len(chunks)} size")
#print (chunks[1].page_content[:500])

embedding_model = OpenAIEmbeddings()

#embed_1 = embedding_model.embed_query("Langchain is great")
#print (f" Embed 1 value is {str(embed_1)[:100]}")
vector_store = FAISS.from_documents(chunks, embedding_model)

retreived = vector_store.similarity_search("What is an agent?", k=2)
print (f" Retreived answer is {retreived[0].page_content}")

print (f" Length of retreived is {len(retreived)}")



import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


st.title("Q and A with AI")
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
question = st.text_input("Your question")

if question:
    response = llm.invoke(question)
    st.write(response.content)
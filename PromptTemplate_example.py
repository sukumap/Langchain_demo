import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


st.title("Q and A with AI")
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
prompt = PromptTemplate(input=["country"], template= "What is the currency of {country}. Provide a short paragraph answer")
country = st.text_input("Input Country")

if country:
    question = "Provide information about currency of " + country + "?"
    response = llm.invoke(prompt.format(country=country))
    st.write(response.content)
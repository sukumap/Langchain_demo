import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


st.title("Q and A with AI")
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
prompt = PromptTemplate(input=["country", "paragraph", "language" ], template= """What is the currency of {country}. 
Provide an answer in {paragraph} paragraphs.  Provide for the answer in {language}.""")
country = st.text_input("Input Country")
paragraph = st.number_input("Number of paragarphs", min_value=1, max_value=5)
language = st.text_input("Answer language")

if country and paragraph and language:
    #question = "Provide information about currency of " + country + "?"
    response = llm.invoke(prompt.format(country=country, paragraph=paragraph, language=language))
    st.write(response.content)
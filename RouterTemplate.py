from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os
load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

destinations = [
    "joke: Tells a Joke",
    "weather: Gives a weather",
    "math: Solves math problems",
    "default: Handle other queries"
]
destination_str = "\n".join(destinations)

router_template_full = """You are a routing assistant. Choose the most appropriate destination for the user's input.

Available destinations:
{dest_str}

User input:
{input}

Return the destination name only.
"""

router_prompt_full = PromptTemplate.from_template(router_template_full)
chain = router_prompt_full | llm

result = chain.invoke({"dest_str": destination_str, "input": "5+5=?"})
print(result)

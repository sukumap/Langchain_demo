import langchain
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values


load_dotenv()
env_file_value = dotenv_values().get("OPENAI_API_KEY")
#key = os.environ["OPENAI_API_KEY"]
#print (f" Setup complete and ke is {key}")
#print (f" Open ai key in file is {env_file_value}")
#print (f" Open ai key is {os.environ.get("OPENAI_API_KEY")}")

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
user_input = "Can I go for a run now"
router_result = chain.invoke({"dest_str": destination_str, "input": user_input})

selected_chain = router_result.content
print(f" The response of router prompt is {selected_chain}")

destination_chains = {
    "joke": PromptTemplate.from_template("Here's a joke for you:") | llm,
    "weather": PromptTemplate.from_template("Let me describe the weather today:") | llm,
    "math": PromptTemplate.from_template("Solving the following problem:\n{input}") | llm,
    "default": PromptTemplate.from_template("Sorry, I don’t understand your request.") | llm
}

input_param = {"input": user_input} if selected_chain == "math" else {}
dest_chain_result = destination_chains[router_result.content].invoke(input_param)
print (f" Destination chain result is {dest_chain_result.content}")
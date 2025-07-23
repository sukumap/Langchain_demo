import langchain
from langchain.chains import llmchain 
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
print (" Setup complete")
routerTemplate = "Given the user input below, decide which destination it should go to {input} "
routerprompt = PromptTemplate.from_template(routerTemplate)
print(routerprompt.format(input="Tell me a joke."))

destinations = ["joke: Tells a Joke", "weather: Gives a weather", "math: Solves math problems", "default: Handle other queries"]
destination_str = "\n".join(destinations)
print (destination_str)

router_template_full = """ You're a routing assistant. Choose the most appropriate destination for users input.
Available destinations: 
{dest_str}
User input :  
{input}
Return the destination name only"""

router_prompt_full = PromptTemplate.from_template(router_template_full)
print (router_prompt_full.format(input = "Tell me a Joke", dest_str = destination_str))

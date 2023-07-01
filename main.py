from langchain.llms import OpenAI
import openai
from dotenv import dotenv_values

config = dotenv_values(".env")

DEEP_LAKE_API = config["DEEP_LAKE_API"]
OPEN_AI_API = config["OPEN_AI_API"]

llm = OpenAI(model="text-ada-001", openai_api_key=OPEN_AI_API)

prompt = input("Enter a prompt: ")
print(llm(prompt))
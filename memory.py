from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from dotenv import dotenv_values
import streamlit as st
from streamlit_chat import message


config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]


llm = OpenAI(temperature=0, openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")

memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)




query = ""
while query != "exit":
    query = input("Enter your query: ")
    print(conversation.predict(input=query))
    history = memory.load_memory_variables({})['history']
    print(history)
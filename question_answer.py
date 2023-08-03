from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import dotenv_values
import streamlit as st
from streamlit_chat import message


config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]

model = ChatOpenAI(openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")
chat_history = []

def process_text(path: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    loader = PyPDFLoader(path)

    # it is the same as - docs = loader.load(), text = text_splitter.split_documents(docs)
    splitted_text = loader.load_and_split(text_splitter=text_splitter)
    print("Number of sentences: ", len(splitted_text))
    return splitted_text

def database(splitted_text):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
    db = Chroma.from_documents(splitted_text, embeddings)
    return db

def conversation(db: Chroma):
    retriever = db.as_retriever()
    # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever)

    return conversation_chain

def handle_query(query: str):
    result = st.session_state.conversation({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    # messages = st.session_state.get('chat_history', [])
    for i, msg in enumerate(chat_history):
        message(message=msg[0], is_user=True)
        message(message=msg[1], is_user=False)
    

if __name__ == "__main__":

    if "conversation" not in st.session_state:
        st.session_state.conversation = None


    st.title("PDF Q&A")
    splitted_text = process_text("example.pdf")
    db = database(splitted_text)

    st.session_state.conversation = conversation(db)

    query = st.sidebar.text_input("Ask question")
    
    while(query!="exit"):
        if query:
            handle_query(query)
        


# what is the total number of AI publications?
# What is this number divided by 2?
    
    

    


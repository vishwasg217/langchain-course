from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import dotenv_values


config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]

model = ChatOpenAI(openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")


def process_text(path: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    loader = PyPDFLoader(path)

    # it is the same as - docs = loader.load(), text = text_splitter.split_documents(docs)
    splitted_text = loader.load_and_split(text_splitter=text_splitter)
    return splitted_text

def database(spitted_text):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
    db = Chroma.from_documents(splitted_text, embeddings)
    return db

if __name__ == "__main__":
    splitted_text = process_text("example.pdf")
    db = database(splitted_text)
    


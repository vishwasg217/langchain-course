from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain import PromptTemplate
from dotenv import dotenv_values
import os

config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]

# disallowed_special=() is required to avoid Exception: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte from tiktoken for some repositories
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API, disallowed_special=())    

# index codebase
CODEBASE_PATH  = "./the-algorithm"

def load_text_from_dir(path: str, encoding: str = "utf-8") -> list:
    """
    loading all the text in each of the file in the codebase and then appending it to a list. 
    """
    docs = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, filename), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

    return docs


def split_text(docs: list) -> list:
    """
    Splitting the text into sentences and then appending it to a list.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000)
    texts = text_splitter.split_documents(docs)
    return texts


docs = load_text_from_dir(CODEBASE_PATH)
texts = split_text(docs)
    


        






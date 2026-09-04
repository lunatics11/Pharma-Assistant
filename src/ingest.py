import os
import pandas as pd 
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pharma_knowledge"

def load_text(file_path: str):
    loader = TextLoader(file_path, encoding = "utf-8")
    return loader.load()

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_excel_or_csv(file_path: str):
    docs = []
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else :
        df = pd.read_excel(file_path, engine="openpyxl")
    preview = df.head(50).to_markdown(index = False)

    schema = "\n".join([f"{col} : {str(dtype)}" for col, dtype in df.dtypes.items()])

    content = f"""
File : {os.path.basename(file_path)}
Schema : {schema}
Preview : {preview}
"""
    
    docs.append(Document(
        page_content= content, 
        meta_data = {"source": os.path.basename(file_path), "type" : "table"},
    ))
    return docs

def load_documents():
    all_docs = []

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if file_name.lower().endswith(".txt"):
            all_docs.extend(load_text(file_path))
        elif file_name.lower().endswith(".pdf"):
            all_docs.extend(load_pdf(file_path))
        elif file_name.lower().endswith((".csv", ".xlsx")):
            all_docs.extend(load_excel_or_csv(file_path))
    return all_docs

def main():
    docs = load_documents()

    if not docs:
        print("No docs found")
        return

    # one chunk is of 800, split text into semantic boundaries like para and sentences 
    # keep chunk_overlap so that context doesn't get missed
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 150)
    chunks = splitter.split_documents(docs)
    # convert text chunks into vector embeddings 
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    # Intialise chroma database, stores vectorised documents
    vector_store = Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function  = embeddings, 
        persist_directory = CHROMA_DIR
    )
    vector_store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into Chroma")

# One chunk is stored as this in vector_store 
# json{
#   "id": "chunk_942a7c_001",
#   "vector_embedding": [0.0124, -0.0451, 0.3109, "...", -0.0087], 
#   "page_content": "Chroma is an open-source vector database designed to store and query document embeddings efficiently.",
#   "metadata": {
#     "source": "document.pdf",
#     "page": 4,
#     "author": "Jane Doe"
#   }
# }

if __name__ == "__main__":
    main()
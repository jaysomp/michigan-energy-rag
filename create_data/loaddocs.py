from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import os
import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3


# Initialize ChromaDB with a local embedding model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="../chroma_db", embedding_function=embedding_function)

# Path to documents
DOCS_PATH = "documents"

def load_documents():
    """Loads text and PDF documents from the documents directory."""
    docs = []
    for file in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, file)
        
        if file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue
        
        documents = loader.load()
        docs.extend(documents)
    
    return docs

def process_documents():
    """Processes and indexes documents into ChromaDB."""
    docs = load_documents()
    
    # Split into chunks for better embedding performance
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    
    # Store embeddings in ChromaDB
    db.add_documents(split_docs)  # 🔥 FIXED: Now storing properly

    print("✅ Documents indexed successfully!")

if __name__ == "__main__":
    process_documents()
 
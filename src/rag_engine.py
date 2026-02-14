import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} 
)

VECTOR_DB_PATH = "vector_store"

def process_cleaned_csv(file_path):
    """Indexes Agent A's output into a persistent brain."""
    if not os.path.exists(file_path): return "⚠️ File not found."
    
    loader = CSVLoader(file_path=file_path)
    return _update_brain(loader, "CSV Data")

def process_uploaded_file(uploaded_file):
    """Indexes PDFs/Text into a persistent brain."""
    folder = "data"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path) if uploaded_file.name.endswith(".pdf") else TextLoader(file_path)
    return _update_brain(loader, uploaded_file.name)

def _update_brain(loader, source_name):
    """Rebuilds/Updates FAISS index with smarter chunking for speed."""
    try:
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        db = FAISS.from_documents(docs, embeddings)
        db.save_local(VECTOR_DB_PATH) 
        return f"✅ Brain Indexed: {source_name}"
    except Exception as e:
        return f"❌ Indexing Error: {str(e)}"

def get_retriever():
    """Loads the brain from disk instantly."""
    if os.path.exists(VECTOR_DB_PATH):
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return db.as_retriever(search_kwargs={"k": 5}) 
    return None
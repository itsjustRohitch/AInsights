import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
# --- FIX: New Import Path ---
from langchain_text_splitters import CharacterTextSplitter

# Setup: Use a free, local model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store():
    print("🔄 Reading market report...")
    try:
        # Load your text data
        loader = TextLoader('data/market_report.txt')
        documents = loader.load()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        print("🧠 Memorizing data (Creating Vector Store)...")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("vector_store")
        print("✅ Success: AI Memory saved to 'vector_store' folder!")
    except Exception as e:
        print(f"❌ Error: {e}")

def get_retriever():
    db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    build_vector_store()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def embed(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("local_vectordatabase")
        return "Vectorstore created and saved successfully."
    except Exception as e:
        print(f"Error embedding chunks: {e}")
        return None
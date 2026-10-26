from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def retrieve():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    database = FAISS.load_local(
        "local_vectordatabase", 
        embeddings=emb, 
        allow_dangerous_deserialization=True
    )
    
    retriever = database.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever

if __name__ == "__main__":
    retrieve()
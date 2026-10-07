from loader import document_loader
from text_splitter import text_split
from embedding import embed

if __name__ == "__main__":
    documents = document_loader("D:/nephrology-rag-assistant/uploads/comprehensive-clinical-nephrology.pdf")
    print("Documents loaded successfully.")

    chunks = text_split(documents)
    print("Text split into chunks successfully.")

    r = embed(chunks)
    if r:
        print(r)
    else:
        print("Failed to create vectorstore.")
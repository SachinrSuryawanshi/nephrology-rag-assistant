from langchain_community.document_loaders import PyPDFLoader

def document_loader(file_path):
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading document: {e}")
        return None

if __name__ == "__main__":
    documents = document_loader("D:/nephrology-rag-assistant/uploads/comprehensive-clinical-nephrology.pdf")
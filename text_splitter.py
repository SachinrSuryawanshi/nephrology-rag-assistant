from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_split(documents):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error splitting text: {e}")
        return None
    
import os
import traceback
import streamlit as st
from dotenv import load_dotenv

from create_retriver import retrieve
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
api = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Nephrology Q&A Assistant", page_icon="🧑‍⚕️")

st.markdown("""
    <h1 style="text-align: center;">🧑‍⚕️ Nephrology Q&A Assistant</h1>
    <p style="text-align: center;">
        Ask questions related to kidney health, nephrology topics, test reports, symptoms, etc.
    </p>
""", unsafe_allow_html=True)

if not api:
    st.error("Error: GROQ_API_KEY is missing. Add it to your .env file.")
    st.stop()

@st.cache_resource
def load_rag():
    retriever = retrieve()
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api
    )
    return retriever, llm

retriever, llm = load_rag()

def format_docs(docs):
    if not docs:
        return "No relevant context found."
    
    extracted_text = []
    for item in docs:
        # Extract page_content whether it is a Document or a (Document, score) tuple
        doc = item[0] if isinstance(item, tuple) else item
        if hasattr(doc, "page_content"):
            extracted_text.append(doc.page_content)
        elif isinstance(doc, str):
            extracted_text.append(doc)
            
    combined = "\n\n".join(extracted_text)
    return combined[:5000]

def extract_question(x):
    return x["question"]

prompt = ChatPromptTemplate.from_template("""
You are a medical assistant specializing in kidney health.
Answer the user's question clearly and accurately, drawing specifically from the clinical details provided below.

INSTRUCTIONS:
- Explain relevant concepts, physiological mechanisms, or conditions in detail using the provided context.
- Translate complex terms into simple, patient-friendly language.
- DO NOT use phrases like "according to the text", "the context mentions", or "retrieved documents".
- If the provided context lacks sufficient medical details to answer thoroughly, respond with: "I don't have enough specific information on that topic in my reference materials."

Context:
{context}

Question:
{question}

Helpful Detailed Answer:
""")

rag_chain = (
    {
        "context": RunnableLambda(extract_question) | retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(extract_question)
    }
    | prompt
    | llm
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask your nephrology question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical content..."):
            try:
                result = rag_chain.invoke({"question": question})
                answer = result.content
            except Exception as e:
                print("Detailed traceback:")
                traceback.print_exc()
                answer = f"Error details: {type(e).__name__} - {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text if text else "No extractable text found in this PDF."

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to store PDF embeddings persistently
def store_pdf_embeddings(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    
    # Store embeddings persistently
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vector_store = Chroma.from_texts(texts, embedding_model, persist_directory=persist_directory)
    return vector_store

# Function to answer user queries
def answer_query(query):
    persist_directory = "chroma_db"
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    docs = vector_store.similarity_search(query, k=3)
    if docs:
        return "\n".join([doc.page_content for doc in docs])
    return "No relevant answer found."

# Streamlit UI
st.title("📄 PDF Chatbot")
uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.info("Processing the PDF... Please wait.")
    text = extract_text_from_pdf(uploaded_file)
    
    # Store embeddings persistently
    vector_store = store_pdf_embeddings(text)
    
    query = st.text_input("💬 Ask a question about the PDF:")
    if query:
        answer = answer_query(query)
        st.write("📝 Answer:", answer)

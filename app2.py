import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings  # ✅ Changed: moved to langchain-ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ Changed: separate package
from langchain_core.prompts import ChatPromptTemplate  # ✅ Correct: in langchain_core
from langchain_community.vectorstores import FAISS  # ✅ Correct: stays in community
from langchain_community.document_loaders import PyPDFDirectoryLoader  # ✅ Correct: stays in community
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

st.title("Chatgroq with Llama 3 demo")

llm = ChatGroq(groq_api_key=groq_key,
               model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)


def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create a temporary directory to store uploaded files
        if not os.path.exists("./temp_papers"):
            os.makedirs("./temp_papers")
            
        # Clear existing files in temp directory to avoid mixing with previous uploads
        for file in os.listdir("./temp_papers"):
            os.remove(os.path.join("./temp_papers", file))

        # Save uploaded files
        for uploaded_file in uploaded_files:
            with open(os.path.join("./temp_papers", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())


        st.session_state.loader = PyPDFDirectoryLoader("./temp_papers") # Data ingestion
        st.session_state.docs = st.session_state.loader.load() # DOcument loading
        
        # Check if documents were loaded and have content
        if not st.session_state.docs:
            st.error("No documents were found in the uploaded file.")
            return
            
        # Check for empty content (common with scanned PDFs)
        valid_docs = [doc for doc in st.session_state.docs if doc.page_content and len(doc.page_content.strip()) > 10]
        if not valid_docs:
            st.error("Could not extract text from the uploaded PDF. It might be a scanned image. Please upload a text-based PDF.")
            return
        
        if len(valid_docs) < len(st.session_state.docs):
            st.warning("Some pages could not be read (likely scanned images). Proceeding with readable content.")
            st.session_state.docs = valid_docs

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # data to chunks
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs) # splitting
        
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.split_docs, st.session_state.embeddings) # vector store
        except Exception as e:
            if "ConnectionError" in str(e) or "connection" in str(e).lower():
                st.error("Failed to connect to Ollama. Please ensure Ollama is running (`ollama serve`).")
            else:
                st.error(f"An error occurred during embedding: {e}")
            return
        



# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button("Documents Embedding"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.write("vector store db is ready")
    else:
        st.error("Please upload at least one PDF file.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

import time

if user_prompt := st.chat_input("Enter your question from the documents"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if "vectors" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please click 'Documents Embedding' first to prepare the knowledge base.")
    else:
        retriever = st.session_state.vectors.as_retriever()
        
        # Create RAG chain with source retrieval using explicit LCEL
        retrieval_chain = RunnableParallel(
            {"context": retriever, "input": RunnablePassthrough()}
        )
        
        chain = retrieval_chain.assign(
            answer= prompt | llm | StrOutputParser()
        )
        
        start = time.process_time()
        response = chain.invoke(user_prompt)
        print("Response time: ", time.process_time() - start)
        
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("---------------------------------")
        
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})


    
    

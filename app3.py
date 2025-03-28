import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize user_question in session state
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Accelerated Recovery focuses on helping individuals recover from alcohol dependency through a private, intensive, and science-based treatment program. It emphasizes a non-12-step approach and is tailored to professionals, executives, and high-achievers who want a discreet and effective recovery process. answer any related question wether it is in context or not to the helping individuals recover from alcohol dependency  \n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        # Add the question and answer to the chat history
        st.session_state.chat_history.append({"question": user_question, "answer": response["output_text"]})
        
    except Exception as e:
        error_message = str(e)
        if "Resource has been exhausted" in error_message:
            error_response = "I apologize, but the Google API quota has been exceeded. Please try again later or consider upgrading your API plan."
        else:
            error_response = f"An error occurred: {error_message}"
        
        # Add the question and error message to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": error_response})

def clear_chat_history():
    # Simply clear the chat history without rerunning
    st.session_state.chat_history = []

def handle_input():
    if st.session_state.user_question:
        question = st.session_state.user_question
        user_input(question)
        # Clear the input by updating the session state
        st.session_state.user_question = ""

def main():
    st.set_page_config("Chat PDF")
    st.header(" AI Chatbot 🤖")
    
    # Create a container for chat history that will appear above the input
    chat_container = st.container()
    
    # User input with callback
    st.text_input(
        "Any Questions ? ", 
        key="user_question",
        on_change=handle_input
    )
    
    # Display chat history in the container (this will appear above the input box)
    with chat_container:
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for chat in st.session_state.chat_history:
                st.markdown(f"👤 **User**: {chat['question']}")
                st.markdown(f"🤖 **Bot**: {chat['answer']}")
                st.markdown("---")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        # Modified clear chat history button that doesn't call experimental_rerun
        if st.button("Clear Chat History"):
            clear_chat_history()
            st.success("Chat history cleared!")

if __name__ == "__main__":
    main()
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# Configure Google API key directly
GOOGLE_API_KEY = "AIzaSyC2CU7kmSHct5vKBPv58FHhxHbN5Ow8gxM"  # Replace with your actual API key

# Set up the API key for the chat model
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize user_question in session state
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant specialized in helping individuals recover from alcohol dependency. 
    You provide information about private, intensive, and science-based treatment programs, 
    focusing on non-12-step approaches tailored for professionals, executives, and high-achievers.
    
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def user_input(user_question):
    try:
        chain = get_conversational_chain()
        response = chain.run(question=user_question)
        
        # Add the question and answer to the chat history
        st.session_state.chat_history.append({"question": user_question, "answer": response})
        
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
    st.set_page_config("Alcohol Recovery Assistant")
    st.header("Alcohol Recovery Assistant 🤖")
    
    # Create a container for chat history that will appear above the input
    chat_container = st.container()
    
    # User input with callback
    st.text_input(
        "Ask any question about alcohol recovery: ", 
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
        st.title("Menu")
        if st.button("Clear Chat History"):
            clear_chat_history()
            st.success("Chat history cleared!")

if __name__ == "__main__":
    main()

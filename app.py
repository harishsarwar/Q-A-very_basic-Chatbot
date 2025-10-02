import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

# Load API Key 
load_dotenv()
api_key = os.getenv("API_KEY")

# Streamlit UI
st.set_page_config(page_title="Groq Q&A Chatbot", page_icon="", layout="wide")
st.title(" Groq Q&A Chatbot")

# Sidebar for model selection
st.sidebar.header(" Settings")
available_models = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma-7b-it", "llama-2-70b-chat"]
selected_model = st.sidebar.selectbox("Select LLM model:", available_models)

# Clear chat button
if st.sidebar.button(" Clear Chat History"):
    st.session_state.chat_history = []

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prompt Template 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer clearly and concisely."),
    ("human", "{question}")
])

# Chat Logic 
def get_groq_response(query, model):
    llm = ChatGroq(groq_api_key=api_key, model=model)
    formatted_prompt = prompt.format_messages(question=query)
    response = llm.invoke(formatted_prompt)
    return response.content

# Chat Container
chat_container = st.container()

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(" Ask a question:")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    if not api_key:
        st.error(" No GROQ_API_KEY found in .env file.")
    else:
        # Get response from LLM
        response = get_groq_response(user_input, selected_model)

        # Save to session history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

# Display Chat History 
with chat_container:
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f" **You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f" **AI:** {msg.content}")

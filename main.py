import os
import json

import streamlit as st
from typing import List, Tuple
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import inputChain
from PIL import Image  # Import the Image module from Pillow


# Access the API key
google_api_key = os.getenv("GOOGLE_API_KEY")


# Load the PNG icon image
jnaneshvara_icon = Image.open("janesvaraicon.ico")

GOOGLE_API_KEY=st.secrets['GOOGLE_API_KEY']
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local(
    "siva_puraan_faiss_index", embeddings, allow_dangerous_deserialization=True
)

# streamlit page configuration
st.set_page_config(
    page_title="Shiva Puraan",
    page_icon=jnaneshvara_icon,
    layout="centered"
)

st.sidebar.image("janesvara_logo.png")
st.sidebar.markdown(
    """
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px;">
       <h3>About <b>Jnaneshvara</b> </h3>
<p>
Meet <b>Jnaneshvara</b>, a chatbot created to answer your questions about Lord Shiva. Drawing upon authentic and sacred sources within Hindu mythology—specifically the <i>Shiva Purana</i>, <i>Skanda Purana</i>, <i>Linga Purana</i>, and <i>Shiv Gita</i> (part of the <i>Padma Purana</i>)—<b>Jnaneshvara</b> aims to help you connect with and learn more about Lord Shiva. 
</p>
<br>
<b>Disclaimer:</b> All information provided is sourced from these texts. Please forgive any inaccuracies, as <b>Jnaneshvara</b> is an AI-based bot.
  <br><br>  <b>Language Supported:</b> Hindi, English 
    """,
    unsafe_allow_html=True
)
    


# working_dir = os.path.dirname(os.path.abspath(__file__))
# config_data = json.load(open(f"{working_dir}/config.json"))

# GROQ_API_KEY = config_data["GROQ_API_KEY"]

# # save the api key to environment variable
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# client = Groq()

# initialize the chat history as streamlit session state of not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# streamlit page title
st.title("🔱📿🕉🪘𓆗 Har Har Mahadev 🙏")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message('assistant', avatar='janesvaraicon.png',):
        st.markdown(message["content"])


# input field for user's message:
user_prompt = st.chat_input("Namaste 🙏! I'm Jnaneshvara. Ask me anything about Lord Shiva...")

if user_prompt:

    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    docs = new_db.similarity_search(user_prompt)
    # sens user's message to the LLM and get a response
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]

    response = inputChain.user_input(user_prompt, docs)

    assistant_response = response
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)


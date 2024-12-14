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
    page_title="Jnanesvara ! A Gen AI Chatbot",
    page_icon=jnaneshvara_icon,
    layout="centered"
)

st.sidebar.image("shiva.png")
st.sidebar.markdown(
    """
    <style>
        .sidebar-content {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
        }
        .sidebar-content h3 {
            margin-top: 0;
        }
        .sidebar-content p {
            font-size: 12px;
            margin-bottom: 0; /*remove default margin from p tag*/
        }

    </style>
    <div class="sidebar-content">
        <h3><b>शिव बॉट के बारे में</b></h3>
       <p>
        <b>शिव बॉट </b> से मिलें, यह एक चैटबॉट है जो भगवान शिव के बारे में आपके सवालों के जवाब देने के लिए बनाया गया है। हिंदू पौराणिक कथाओं के प्रामाणिक और पवित्र स्रोतों पर आधारित - विशेष रूप से <i>शिव पुराण</i>, <i>स्कंद पुराण</i>, <i>लिंग पुराण</i> और <i>शिव गीता </i> (पद्म पुराण का हिस्सा)।
        </p>    </div>
    """,
    unsafe_allow_html=True, #Still needed for style tags
)

# Initialize the chat history if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    


# streamlit page title
st.title("🕉🪘𓆗 हर हर महादेव 🙏")

def process_response(prompt, role="user"):
    docs = new_db.similarity_search(prompt)
    response = inputChain.user_input(prompt, docs) #Check this function for accidental question appends
    return response, prompt #Return both response and prompt

# Suggested questions (unchanged)
suggested_questions = [
    "शिव जी का प्रिय मंत्र क्या है?",
    "हम शिवलिंग को बेल पत्र क्यों चढ़ाते हैं?",
    "शिवलिंग पर तुलसी क्यों नहीं चढ़ाते हैं?",
    "भगवान शिव को कौन प्रिय है?",
    "शिव जी को भस्म क्यों चढ़ाई जाती है?",
    "शिवलिंग पर चंदन का लेप क्यों लगते हैं?",
]

# Sidebar for suggested questions (unchanged)
with st.sidebar:
    st.header("सुझाए गए प्रश्न")
    clicked_question = None
    for question in suggested_questions:
        if st.button(question):
            clicked_question = question

if clicked_question:
    with st.chat_message("user"):
        st.markdown(clicked_question)
    response, prompt = process_response(clicked_question, "assistant")
    check = inputChain.check_for_answer(clicked_question, response)
    if (check =="Direct and Accurate\n"):
        assistant_response = response
    else:
        assistant_response = inputChain.generate_reponse(clicked_question)
    st.session_state.chat_history.append({"role": "user", "content": prompt}) #Use returned prompt
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

# input field for user's message:
user_prompt = st.chat_input("नमस्ते 🙏! मुझसे भगवान शिव के बारे में कुछ भी पूछें...")

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
    check = inputChain.check_for_answer(user_prompt, response)
    if (check =="Direct and Accurate\n"):
        assistant_response = response
    else:
        assistant_response = inputChain.generate_reponse(user_prompt)

    
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display chat history
for message in st.session_state.chat_history:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

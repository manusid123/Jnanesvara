from typing import List, Tuple
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import os

def get_pdf_text(directory) -> List[Document]:
    """Extracts text from PDF files and creates Document objects.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        A list of Document objects, each representing a PDF file.
    """
    all_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for i, doc in enumerate(docs):
                doc.metadata["source"] = f"{filename}_page_{i+1}" 
            all_docs.extend(docs)
    return all_docs



def get_text_chunks(docs: List[Document]):
    """Splits documents into smaller chunks while preserving metadata.

    Args:
        docs: A list of Document objects.

    Returns:
        A list of Document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    all_chunks = []
    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["source"] = doc.metadata["source"]
        all_chunks.extend(chunks)
    return all_chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local("siva_puraan_faiss_index")


def get_conversational_chain():
    prompt_template = """
    Assume you are a true devotee of Lord Shiva and you love to answer queries regarding Lord Shiva.

    Context:
    {context}

    Question:
    {question}
    Note: Dont answer if any question is not related to Lord Shiva at all. Politely decline to answer by mentioning I am true devotee of lord shiva, happy to answer query regarding lord shiva.
    If someone greets you simply greets back with Om Namah Shivaay.

    Also detect the language of user and respond in the user's language as far as possible.
   
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# def user_input(user_question: str, docs: List[Document]) -> dict:
#     """Gets the answer to the user's question, along with reference document sources.

#     Args:
#         user_question: The user's question.
#         docs: A list of Documents returned from the similarity search.

#     Returns:
#         A dictionary containing the model's answer and the reference document sources.
#     """

#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents": docs, "question": user_question}, return_only_outputs=True
#     )

    

#     return response['output_text']
def user_input(user_question: str, docs: List[Document]) -> str:
    """Gets the answer to the user's question, including the source if available.

    Args:
        user_question: The user's question.
        docs: A list of Documents returned from the similarity search.

    Returns:
        The model's answer as a string, including the source on a new line.
    """

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    text_answer = response['output_text']

    # Extract and format source information
    reference_docs = [doc.metadata["source"] for doc in docs]
    if reference_docs:
        unique_source_prefixes = set()  # Use a set for efficient deduplication
        for doc in docs:
            source = doc.metadata["source"].split("_text.pdf")[0]
            first_two_words = " ".join(source.split()[:2])
            unique_source_prefixes.add(first_two_words)

        source_text = "\n\n**Source: " + ", ".join(unique_source_prefixes) +"**"
        text_answer += source_text


    return text_answer

import streamlit as st
from langchain_community.llms import Cohere  # Correct import
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain.chains import RetrievalQA
import os
from docx import Document

COHERE_API_KEY = "TjktIf31DNGNNff3WzWvr1n3UBybuOF1R1jpu1Xy"
USER_AGENT = "mujtaba/1.0"

# Load Word Document
def load_word_document(doc_path):
    try:
        doc = Document(doc_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return ""

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load or Create FAISS Index
DOC_PATH = "C:/Users/Admin/Desktop/mujtaba/data/sample.docx"
if os.path.exists(DOC_PATH):
    doc_text = load_word_document(DOC_PATH)
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY, user_agent=USER_AGENT)

    store_filename = "word_doc_faiss.index"
    if os.path.exists(store_filename):
        vectorstore = FAISS.load_local(store_filename, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_texts([doc_text], embedding=embeddings)
        vectorstore.save_local(store_filename)

    retriever = vectorstore.as_retriever()
    llm = Cohere(model="command", temperature=0.7, cohere_api_key=COHERE_API_KEY, user_agent=USER_AGENT)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
else:
    st.error("Document not found!")
    qa_chain = None

# **Define `send_query` function**
def send_query():
    user_input = st.session_state.get("user_input", "").strip()
    if user_input and qa_chain:
        answer = qa_chain.run(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.user_input = ""  # Clear input field

# Streamlit UI
import streamlit as st

# Set page title and layout
st.set_page_config(page_title="CollegeGPT", layout="wide")

# Apply custom CSS for black background and chat styling
st.markdown(
    """
    <style>
        body { background-color: black; color: white; }
        .stApp { background-color: black; }
        .chat-box { background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .user { text-align: right; color: cyan; }
        .assistant { text-align: left; color: white; }
        .search-container { position: fixed; bottom: 20px; width: 80%; left: 10%; background-color: #333; padding: 10px; border-radius: 10px; }
        .search-input { flex: 1; background: transparent; color: white; border: none; outline: none; font-size: 16px; padding: 5px; }
        .search-button { background-color: #ff9800; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; margin-left: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with "New Chat" button
st.sidebar.button("New Chat", on_click=lambda: st.session_state.update(chat_history=[]))

# Display logo and title
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("C:/Users/Admin/Desktop/mujtaba/logo.jpg", width=80)  # Adjust path to logo
with col2:
    st.title("CollegeGPT")

# Display chat history
for chat in st.session_state.chat_history:
    role_class = "user" if chat["role"] == "user" else "assistant"
    st.markdown(f"<div class='chat-box {role_class}'><b>{chat['role'].capitalize()}:</b> {chat['content']}</div>", unsafe_allow_html=True)

# Input field for questions
st.text_input("Ask another question:", key="user_input", on_change=send_query)

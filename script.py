# Streamlit UI
import streamlit as st
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain.chains import RetrievalQA
import os
from docx import Document
import json
from io import StringIO
import requests

# Page Configuration
st.set_page_config(page_title="CollegeGPT", layout="wide")
COHERE_API_KEY = "TjktIf31DNGNNff3WzWvr1n3UBybuOF1R1jpu1Xy"
USER_AGENT = "mujtaba/1.0"

# Load custom CSS dropdown style
st.markdown("""
    <style>
    .dropdown {
        position: relative;
        display: inline-block;
    }
    .dropdown-content {
        display: none;
        position: absolute;
        background-color: #fff;
        min-width: 160px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        z-index: 1;
        border-radius: 8px;
        overflow: hidden;
    }
    .dropdown-content button {
        color: black;
        padding: 12px 16px;
        text-decoration: none;
        display: block;
        border: none;
        background: none;
        width: 100%;
        text-align: left;
        cursor: pointer;
    }
    .dropdown-content button:hover {
        background-color: #f1f1f1;
    }
    .dropdown:hover .dropdown-content {
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Load Word Document
def load_word_document(doc_path):
    try:
        response = requests.get(doc_path)
        response.raise_for_status()
        with open("temp.docx", "wb") as f:
            f.write(response.content)
        doc = Document("temp.docx")
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Failed to load document from URL: {e}")
        return ""


# Persistent Save/Load
SAVE_PATH = "chat_sessions.json"

def save_sessions():
    with open(SAVE_PATH, 'w') as f:
        json.dump(st.session_state.chat_sessions, f)

def load_sessions():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, 'r') as f:
            return json.load(f)
    return {"Default": []}

# Initialize State
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_sessions()
    st.session_state.active_session = list(st.session_state.chat_sessions.keys())[0]
if "session_to_rename" not in st.session_state:
    st.session_state.session_to_rename = None

# Rename Session
def rename_session(old_name, new_name):
    if new_name and new_name != old_name and new_name not in st.session_state.chat_sessions:
        # Copy chat to new name
        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.get(old_name, [])
        # Delete old one
        if old_name in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[old_name]
        # Update active session reference
        if st.session_state.active_session == old_name:
            st.session_state.active_session = new_name
        save_sessions()

    # Reset rename state to hide input
    st.session_state.session_to_rename = None

# Delete Session
def delete_session(session_name):
    if session_name in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_name]
        remaining = list(st.session_state.chat_sessions.keys())
        st.session_state.active_session = remaining[0] if remaining else "Default"
        if not remaining:
            st.session_state.chat_sessions["Default"] = []
        save_sessions()

# Sidebar Chat History
st.sidebar.markdown("### Chat History")

for session in list(st.session_state.chat_sessions.keys()):
    active = " (Active)" if session == st.session_state.active_session else ""
    with st.sidebar.container():
        col1, col2 = st.columns([0.75, 0.25])
        with col1:
            if st.session_state.session_to_rename == session:
                new_name = st.text_input("Rename", value=session, key=f"rename_input_{session}")
                if st.session_state.get(f"rename_input_{session}") != session:
                    rename_session(session, st.session_state[f"rename_input_{session}"])
            else:
                if st.button(session + active, key=f"select_{session}"):
                    st.session_state.active_session = session

        with col2:
            with st.expander("⋮", expanded=False):
                if st.button("🗑 Delete", key=f"delete_{session}"):
                    delete_session(session)
                    st.rerun()
                chat = st.session_state.chat_sessions[session]
                output = StringIO()
                for entry in chat:
                    output.write(f"{entry['role'].capitalize()}: {entry['content']}\n")
                st.download_button("📤 Export", output.getvalue(), file_name=f"{session}.txt", key=f"export_{session}")

# New Chat
def create_new_session():
    new_title = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[new_title] = []
    st.session_state.active_session = new_title
    save_sessions()

st.sidebar.markdown("---")
st.sidebar.button("➕ New Chat", on_click=create_new_session)

# Import Chats
uploaded_file = st.sidebar.file_uploader("Import Chat (.txt)", type="txt")
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().split('\n')
    imported_chat = []
    for line in lines:
        if line.startswith("User:"):
            imported_chat.append({"role": "user", "content": line[5:].strip()})
        elif line.startswith("Assistant:"):
            imported_chat.append({"role": "assistant", "content": line[9:].strip()})
    new_title = f"Imported {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[new_title] = imported_chat
    st.session_state.active_session = new_title
    save_sessions()
    st.success(f"Imported chat as '{new_title}'")

# Load FAISS Index
DOC_PATH = "https://raw.githubusercontent.com/5298479/college-GPT/main/data/sample.docx"
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

# Send Query
def send_query():
    user_input = st.session_state.get("user_input", "").strip()
    current_session = st.session_state.active_session

    if user_input and qa_chain:
        # Auto-rename on first message
        if (current_session.startswith("Chat") or current_session.startswith("Imported")) and len(st.session_state.chat_sessions[current_session]) == 0:
            new_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
            if new_title not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_session)
                st.session_state.active_session = new_title
                current_session = new_title  # Update reference to avoid KeyError

        answer = qa_chain.run(user_input)
        st.session_state.chat_sessions[current_session].append({"role": "user", "content": user_input})
        st.session_state.chat_sessions[current_session].append({"role": "assistant", "content": answer})
        st.session_state.user_input = ""
        save_sessions()

# Chat Styling
st.markdown("""
    <style>
        body { background-color: black; color: white; }
        .stApp { background-color: black; }
        .chat-box { background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .user { text-align: right; color: cyan; }
        .assistant { text-align: left; color: white; }
    </style>
""", unsafe_allow_html=True)

# Title and Logo
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://raw.githubusercontent.com/5298479/college-GPT/main/logo.jpg", width=80)
with col2:
    st.title("CollegeGPT")

# Chat Display
current_session = st.session_state.active_session
for chat in st.session_state.chat_sessions.get(current_session, []):
    role_class = "user" if chat["role"] == "user" else "assistant"
    st.markdown(f"<div class='chat-box {role_class}'><b>{chat['role'].capitalize()}:</b> {chat['content']}</div>", unsafe_allow_html=True)

# Input Field
st.text_input("Ask another question:", key="user_input", on_change=send_query)

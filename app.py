import streamlit as st
import fitz
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.chat-container {max-width: 800px; margin: auto;}
.user-msg {
    background-color: #2b313e;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: right;
}
.bot-msg {
    background-color: #1f6feb;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: left;
}
.upload-box {
    text-align: center;
    padding: 30px;
    border: 2px dashed #4CAF50;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🤖 Chat with Your PDF</h1>", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="gpt2")
    return embed_model, generator

embed_model, generator = load_models()

# ------------------ FUNCTIONS ------------------
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)

def create_faiss(chunks):
    embeddings = embed_model.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def generate_answer(query, chunks, index):
    q_embed = embed_model.encode([query]).astype("float32")
    D, I = index.search(q_embed, 3)

    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
    Answer based on the context below:

    {context}

    Question: {query}
    Answer:
    """

    result = generator(prompt, max_length=200)
    return result[0]["generated_text"]

# ------------------ SESSION ------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ UPLOAD UI ------------------
if not st.session_state.processed:

    st.markdown("<div class='upload-box'>📂 Upload your PDF to start</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file:
        st.success("✅ PDF uploaded successfully!")

        with st.spinner("Processing PDF..."):
            raw_text = extract_text(uploaded_file)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned)
            index = create_faiss(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.processed = True

        st.rerun()

# ------------------ CHAT UI ------------------
import streamlit as st
import fitz
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.chat-container {max-width: 800px; margin: auto;}
.user-msg {
    background-color: #2b313e;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: right;
}
.bot-msg {
    background-color: #1f6feb;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: left;
}
.upload-box {
    text-align: center;
    padding: 30px;
    border: 2px dashed #4CAF50;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🤖 Chat with Your PDF</h1>", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="gpt2")
    return embed_model, generator

embed_model, generator = load_models()

# ------------------ FUNCTIONS ------------------
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)

def create_faiss(chunks):
    embeddings = embed_model.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def generate_answer(query, chunks, index):
    q_embed = embed_model.encode([query]).astype("float32")
    D, I = index.search(q_embed, 3)

    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
    Answer based on the context below:

    {context}

    Question: {query}
    Answer:
    """

    result = generator(prompt, max_length=200)
    return result[0]["generated_text"]

# ------------------ SESSION ------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ UPLOAD UI ------------------
if not st.session_state.processed:

    st.markdown("<div class='upload-box'>📂 Upload your PDF to start</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file:
        st.success("✅ PDF uploaded successfully!")

        with st.spinner("Processing PDF..."):
            raw_text = extract_text(uploaded_file)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned)
            index = create_faiss(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.processed = True

        st.rerun()

# ------------------ CHAT UI ------------------
else:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Show history
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"<div class='user-msg'>👤 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg}</div>", unsafe_allow_html=True)

    # Chat input
    query = st.chat_input("Ask something about your PDF...")

    if query:
        st.session_state.history.append(("user", query))

        with st.spinner("Thinking..."):
            answer = generate_answer(query, st.session_state.chunks, st.session_state.index)

        st.session_state.history.append(("bot", answer))

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Show history
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"<div class='user-msg'>👤 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg}</div>", unsafe_allow_html=True)

    # Chat input
    query = st.chat_input("Ask something about your PDF...")

    if query:
        st.session_state.history.append(("user", query))

        with st.spinner("Thinking..."):
            answer = generate_answer(query, st.session_state.chunks, st.session_state.index)

        st.session_state.history.append(("bot", answer))

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

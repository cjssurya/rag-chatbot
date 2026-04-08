import streamlit as st
import fitz
from PIL import Image
import io
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")

# ------------------ UI ------------------
st.markdown("""
<style>
.main {background-color: #0E1117; color: white;}
.big-title {font-size: 40px; text-align:center; color:#4CAF50;}
.subtitle {text-align:center; color:#aaa;}
.chat-box {background:#1E1E1E; padding:10px; border-radius:10px; margin:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🤖 Chat with Your PDF</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload PDF & Ask Questions</div>', unsafe_allow_html=True)
st.divider()

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # FIXED: use supported pipeline
    generator = pipeline(
        "text-generation",
        model="gpt2"
    )

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

    result = generator(prompt, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"]

# ------------------ LAYOUT ------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose file", type=["pdf"])

    if uploaded_file:
        st.success("PDF uploaded!")

        with st.spinner("Processing PDF..."):
            raw_text = extract_text(uploaded_file)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned)
            index = create_faiss(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index

with col2:
    st.subheader("💬 Ask Question")

    query = st.text_input("Enter your question")

    if query and "index" in st.session_state:
        st.markdown(f"<div class='chat-box'><b>👤 You:</b> {query}</div>", unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            answer = generate_answer(query, st.session_state.chunks, st.session_state.index)

        st.markdown(f"<div class='chat-box'><b>🤖 AI:</b> {answer}</div>", unsafe_allow_html=True)

import streamlit as st

st.set_page_config(page_title="Chat with PDF", layout="wide")

# ---------- CSS ----------
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

# ---------- TITLE ----------
st.markdown("<h1 style='text-align:center;'>🤖 Chat with Your PDF</h1>", unsafe_allow_html=True)

# ---------- SESSION ----------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- UPLOAD UI ----------
if not st.session_state.processed:

    st.markdown("<div class='upload-box'>Upload your PDF to start</div>", unsafe_allow_html=True)

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

# ---------- CHAT UI ----------
else:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Show chat history
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"<div class='user-msg'>👤 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg}</div>", unsafe_allow_html=True)

    # Input
    query = st.chat_input("Ask something about your PDF...")

    if query:
        st.session_state.history.append(("user", query))

        with st.spinner("Thinking..."):
            answer = generate_answer(query, st.session_state.chunks, st.session_state.index)

        st.session_state.history.append(("bot", answer))

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

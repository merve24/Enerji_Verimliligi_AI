import streamlit as st
from data import prepare_rag_data, simple_query_streamlit

st.set_page_config(page_title="Enerji AI", page_icon="💡", layout="wide")

# -----------------------------
# 🎨 DARK THEME + WHITE TEXT FIX
# -----------------------------
st.markdown("""
<style>

/* Arka plan */
body {
    background-color: #0e1117;
    color: #ffffff;
}

/* Genel yazı rengi */
html, body, [class*="css"]  {
    color: #ffffff !important;
}

/* Başlık */
h1, h2, h3, p {
    color: #ffffff !important;
}

/* Kullanıcı mesajı */
.user-message {
    background-color: transparent;
    color: #ffffff;
    padding: 10px;
    margin: 10px;
    text-align: right;
    font-weight: 500;
}

/* Asistan mesajı */
.assistant-message {
    background-color: transparent;
    color: #ffffff;
    padding: 10px;
    margin: 10px;
    text-align: left;
    line-height: 1.6;
    font-size: 15px;
}

/* Input yazı rengi */
.stChatInput input {
    color: #ffffff !important;
}

/* Input kutu arka plan */
.stChatInput {
    background-color: #1c1f26;
}

/* Scrollbar daha temiz */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# 🔑 API KEY
# -----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("API key eksik")
    st.stop()

API_KEY = st.secrets["GEMINI_API_KEY"]


# -----------------------------
# 🧠 DATA LOAD
# -----------------------------
if "collection" not in st.session_state:
    st.session_state.collection = None

if st.session_state.collection is None:
    with st.spinner("Sistem hazırlanıyor..."):
        st.session_state.collection = prepare_rag_data(API_KEY)


# -----------------------------
# 🏷️ HEADER
# -----------------------------
st.title("💡 Enerji Verimliliği AI Chatbot")
st.caption("Sizin için sorularınızı profesyonel şekilde cevaplamaya hazırım.")


# -----------------------------
# 💬 STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# 💬 MESAJ GÖSTER
# -----------------------------
for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message">{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="assistant-message">{msg["content"]}</div>',
            unsafe_allow_html=True
        )


# -----------------------------
# ✍️ INPUT
# -----------------------------
if prompt := st.chat_input("Sorunuzu yazın..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    st.markdown(
        f'<div class="user-message">{prompt}</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Yanıt hazırlanıyor..."):
        response, _ = simple_query_streamlit(
            prompt,
            st.session_state.collection,
            API_KEY
        )

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown(
        f'<div class="assistant-message">{response}</div>',
        unsafe_allow_html=True
    )
import streamlit as st
from data import prepare_rag_data, simple_query_streamlit

st.set_page_config(page_title="Enerji AI", page_icon="💡", layout="wide")

# -----------------------------
# 🎨 CSS
# -----------------------------
st.markdown("""
<style>

/* Kullanıcı mesajı (sağ - şeffaf) */
.user-message {
    background-color: transparent;
    color: #222;
    padding: 10px;
    margin: 10px;
    text-align: right;
    font-weight: 500;
}

/* Asistan mesajı (sol - düz metin) */
.assistant-message {
    background-color: transparent;
    color: #111;
    padding: 10px;
    margin: 10px;
    text-align: left;
    line-height: 1.6;
    font-size: 15px;
}

/* Başlık */
.header {
    text-align: center;
    margin-bottom: 20px;
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
st.markdown('<div class="header"><h1>💡 Enerji Verimliliği AI Chatbot</h1><p>Sizin için sorularınızı profesyonel şekilde cevaplamaya hazırım.</p></div>', unsafe_allow_html=True)


# -----------------------------
# 💬 MESAJLAR
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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
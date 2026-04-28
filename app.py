import streamlit as st
from data import prepare_rag_data, simple_query_streamlit

st.set_page_config(page_title="Enerji AI", page_icon="💡", layout="wide")

# -----------------------------
# 🎨 CUSTOM CSS (CHAT UI)
# -----------------------------
st.markdown("""
<style>

/* Genel arka plan */
body {
    background-color: #f7f7f7;
}

/* Kullanıcı mesajı (sağ) */
.user-message {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 15px;
    margin: 8px;
    width: fit-content;
    margin-left: auto;
    max-width: 70%;
    font-size: 15px;
}

/* Asistan mesajı (sol) */
.assistant-message {
    background-color: #ffffff;
    padding: 12px;
    border-radius: 15px;
    margin: 8px;
    width: fit-content;
    margin-right: auto;
    max-width: 70%;
    font-size: 15px;
    border: 1px solid #e6e6e6;
}

/* Input kutusu */
.stChatInput {
    border-top: 1px solid #ddd;
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
# 🧠 DATA LOAD (LAZY)
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
# 💬 MESAJ STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# 💬 MESAJLARI GÖSTER
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

    # kullanıcı mesajı ekle
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.markdown(
        f'<div class="user-message">{prompt}</div>',
        unsafe_allow_html=True
    )

    # cevap üret
    with st.spinner("Yanıt hazırlanıyor..."):
        response, _ = simple_query_streamlit(
            prompt,
            st.session_state.collection,
            API_KEY
        )

    # cevap ekle
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown(
        f'<div class="assistant-message">{response}</div>',
        unsafe_allow_html=True
    )
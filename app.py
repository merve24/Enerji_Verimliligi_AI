st.markdown("""
<style>
/* Kullanıcı mesajı (sağ) */
.user-message {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 12px;
    margin: 5px;
    width: fit-content;
    margin-left: auto;
    max-width: 70%;
}

/* Asistan mesajı (sol) */
.assistant-message {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 12px;
    margin: 5px;
    width: fit-content;
    margin-right: auto;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)
import streamlit as st
from data import prepare_rag_data, simple_query_streamlit

st.set_page_config(page_title="Enerji AI", page_icon="💡")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("API key eksik")
    st.stop()

API_KEY = st.secrets["GEMINI_API_KEY"]

if "collection" not in st.session_state:
    st.session_state.collection = None

if st.session_state.collection is None:
    with st.spinner("Sistem hazırlanıyor..."):
        st.session_state.collection = prepare_rag_data(API_KEY)

st.title("💡 Enerji Verimliliği AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Sor..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Yanıt hazırlanıyor..."):
        response, _ = simple_query_streamlit(
            prompt,
            st.session_state.collection,
            API_KEY
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

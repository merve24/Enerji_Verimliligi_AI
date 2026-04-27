import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from data import prepare_rag_data, simple_query_streamlit

st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡")

# API KEY kontrol
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Secrets kısmına GEMINI_API_KEY ekle.")
    st.stop()

API_KEY = st.secrets["GEMINI_API_KEY"]

# RAG data yükleme
collection = prepare_rag_data(API_KEY)

if collection is None:
    st.error("PDF/veri dosyası bulunamadı.")
    st.stop()

st.title("💡 Enerji Verimliliği AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Sorularını yanıtlayabilirim ⚡"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Sorunu yaz..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Cevap hazırlanıyor..."):
        response, _ = simple_query_streamlit(prompt, collection, API_KEY)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
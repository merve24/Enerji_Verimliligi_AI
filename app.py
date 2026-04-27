import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from data import prepare_rag_data, simple_query_streamlit

st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡")

# --- API KEY ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Secrets kısmına 'GEMINI_API_KEY' ekleyin.")
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- DATA ---
collection = prepare_rag_data(GEMINI_API_KEY)

if collection is None:
    st.error("Veri dosyası bulunamadı.")
    st.stop()

st.title("💡 Enerji Verimliliği AI Chatbot")

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Merhaba! Enerji verimliliği hakkında sorularınızı yanıtlayabilirim.⚡"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Cevap üretiliyor..."):
        response, _ = simple_query_streamlit(prompt, collection, GEMINI_API_KEY)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
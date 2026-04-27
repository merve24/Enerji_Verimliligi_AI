import os
# Protobuf hatasını engellemek için en üstte olmalıdır
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from google import genai
from data import prepare_rag_data, simple_query_streamlit 

# --- AYARLAR ---
CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

if "GEMINI_API_KEY" not in st.secrets:
    st.error("API Anahtarı bulunamadı! Secrets kısmına 'GEMINI_API_KEY' ekleyin.")
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Veri yükleme (Cache kullanılarak)
collection = prepare_rag_data(GEMINI_API_KEY) 

if collection is None:
    st.error(f"Veri seti yüklenemedi. '{CORRECT_FILE_NAME}' dosyasını kontrol edin.")
    st.stop()

# --- ARAYÜZ ---
st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡")
st.title("💡 Enerji Verimliliği AI Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{'role': 'assistant', 'content': 'Merhaba! Sorularınızı bekliyorum.⚡'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Cevap üretiliyor..."):
        response, _ = simple_query_streamlit(prompt, collection, GEMINI_API_KEY)

    st.chat_message("assistant").write(response)
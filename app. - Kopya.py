import os
# Protobuf hata kontrolünü esnetmek için en üste eklenmelidir
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from google import genai
from data import prepare_rag_data, simple_query_streamlit 

# --- UYGULAMA BAŞLANGICI ---
CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    st.error("API Anahtarı bulunamadı!")
    st.stop()

def load_rag_data(api_key):
    return prepare_rag_data(api_key)

collection = load_rag_data(GEMINI_API_KEY) 

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

    with st.spinner("Cevap hazırlanıyor..."):
        response, source = simple_query_streamlit(prompt, collection, GEMINI_API_KEY)

    st.chat_message("assistant").write(response)
import os
# DİKKAT: Bu ayar protobuf hatasını engellemek için tüm importlardan önce gelmelidir.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from google import genai
from data import prepare_rag_data, simple_query_streamlit 

# --- 1. AYARLAR ---
CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

# API Anahtarı Kontrolü
if "GEMINI_API_KEY" not in st.secrets:
    st.error("API Anahtarı bulunamadı! Lütfen Streamlit Secrets kısmına 'GEMINI_API_KEY' ekleyin.")
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- 2. VERİ YÜKLEME ---
# Veriyi cache kullanarak yüklüyoruz
collection = prepare_rag_data(GEMINI_API_KEY) 

if collection is None:
    st.error(f"Veri seti yüklenemedi. '{CORRECT_FILE_NAME}' dosyasının ana dizinde olduğundan emin olun.")
    st.stop()

# --- 3. ARAYÜZ ---
st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡", layout="wide")
st.title("💡 Enerji Verimliliği AI Chatbot")
st.markdown("Enerji Verimliliği Eğitim Kitabı tabanlı asistanınız hazır. ⚡")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{'role': 'assistant', 'content': 'Merhaba! Enerji verimliliği hakkında ne öğrenmek istersiniz?'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Kitap taranıyor..."):
        response, _ = simple_query_streamlit(prompt, collection, GEMINI_API_KEY)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
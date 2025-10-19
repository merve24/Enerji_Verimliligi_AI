# Streamlit web uygulamasının ana dosyası
import streamlit as st
from google import genai
import os
# data.py dosyasındaki fonksiyonları içe aktarıyoruz
# NOT: simple_query_streamlit fonksiyonunun data.py'den geldiğini varsayıyorum.
from data import prepare_rag_data, simple_query_streamlit 

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---

CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

# 1. API Anahtarı Kontrolü
try:
    # API key bulunamazsa uygulama durur.
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta 'GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Veri yükleme fonksiyonu
def load_rag_data(api_key):
    # data.py dosyasındaki ana fonksiyonu çağırır ve metin parçalarını yükler
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
text_chunks, index_placeholder = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if not text_chunks:
    st.error(f"Veri seti yüklenemedi. Lütfen '{CORRECT_FILE_NAME}' dosyasının GitHub'da olduğundan ve data.py ile aynı dizinde bulunduğundan emin olun.")
    st.stop()

# --- 2. UYGULAMA MANTIĞI VE ARAYÜZ ---
st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡", layout="wide")
st.title("💡 Enerji Verimliliği AI Chatbot")
st.markdown("Enerji dünyasındaki bilgelik parmaklarınızın ucunda. Kitabın **tüm içeriğine** erişimim var. ⚡")

if "messages" not in st.session_state:
    # KRİTİK DÜZELTME YAPILDI (HATA OLUŞTURAN SATIR): Dize tanımlaması için dışta TEK TIRNAK kullanıldı.
    st.session_state["messages"] = [{'role': 'assistant', 'content': 'Merhaba! Enerji verimliliği konularında sorularınızı yanıtlamaya hazırım. Artık **tüm kitaptaki bilgilere** erişimim var. ⚡'}]

for msg in st.session_state.messages:
    avatar = "💡" if msg["role"] == "assistant" else "👤" 
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Enerji verimliliği hakkında bir soru sorun... 📝"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    with st.spinner("Kitabın tamamı taranıyor ve cevap üretiliyor... 🧠"):
        # Sorgu fonksiyonuna text_chunks parametresi eklendi (önceki düzeltme)
        # Not: simple_query_streamlit'in data.py'de güncel parametrelerle tanımlandığından emin olun.
        response, source = simple_query_streamlit(prompt, text_chunks, GEMINI_API_KEY)

    st.chat_message("assistant", avatar="💡").write(response)
    if source:
        st.markdown(f"**Kaynak Metin:** {source[:300]}...")
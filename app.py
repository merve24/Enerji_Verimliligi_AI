# Streamlit web uygulamasının ana dosyası
import streamlit as st
from google import genai
import os
# data.py dosyasındaki fonksiyonları içe aktarıyoruz
from data import prepare_rag_data, simple_query_streamlit 

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---

CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

# API Anahtarı Kontrolü
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta 'GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Veri yükleme fonksiyonu
# prepare_rag_data artık bir ChromaDB 'collection' nesnesi döndürüyor.
def load_rag_data(api_key):
    return prepare_rag_data(api_key)

# 💡 Güncelleme 1: Dönüş değerini collection olarak değiştiriyoruz.
collection = load_rag_data(GEMINI_API_KEY) 

# 💡 Güncelleme 2: Hata kontrolünü collection nesnesine göre yapıyoruz.
if collection is None:
    st.error(f"Veri seti yüklenemedi. Lütfen '{CORRECT_FILE_NAME}' dosyasının doğru konumda olduğundan emin olun ve ChromaDB hatası için konsolu kontrol edin.")
    st.stop()

# --- 2. UYGULAMA MANTIĞI VE ARAYÜZ ---
st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡", layout="wide")
st.title("💡 Enerji Verimliliği AI Chatbot")
st.markdown("Bu asistan, **Enerji Verimliliği Eğitim Kitabı** temel alınarak geliştirilmiştir. Enerji yönetimi, sürdürülebilirlik ve verimlilik konularında size rehberlik edebilirim. ⚡")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{'role': 'assistant', 'content': 'Merhaba! Enerji verimliliği konularında sorularınızı yanıtlamaya hazırım.⚡'}]

for msg in st.session_state.messages:
    avatar = "💡" if msg["role"] == "assistant" else "👤" 
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Enerji verimliliği hakkında bir soru sorun... 📝"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    with st.spinner("Kitabın tamamı taranıyor ve cevap üretiliyor... 🧠"):
        # 💡 Güncelleme 3: Sorgu fonksiyonunu güncel argümanlarla çağır
        # text_chunks ve embeddings yerine 'collection' nesnesini iletiyoruz.
        response, source = simple_query_streamlit(prompt, collection, GEMINI_API_KEY)

    # Kullanıcıya yalnızca cevabı göster
    st.chat_message("assistant", avatar="💡").write(response)

    # 👇 Bu kısım artık yalnızca loglarda (terminalde) görünecek
    if source:
        # Kaynak metin başarıyla çekildiyse
        print("\n🧩 Kullanılan kaynak metin (ilk 300 karakter):\n", source[:300], "...\n")
    else:
        # Kaynak metin boş geldiyse (örn. ChromaDB hatası)
        print("\nKaynak
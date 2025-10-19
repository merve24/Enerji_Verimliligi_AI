# Streamlit web uygulamasının ana dosyası
import streamlit as st
from google import genai
# import numpy kaldırıldı
import os
# data.py dosyasındaki fonksiyonları içe aktarıyoruz
from data import prepare_rag_data, simple_query_streamlit 
# NOT: simple_query_streamlit fonksiyonunun data.py'den geldiğini varsayıyorum.
# Eğer bu fonksiyon app.py içinde ise, lütfen app.py'de tutmaya devam edin.

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---

CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

# 1. HATA ÇÖZÜMÜ: API Anahtarı Kontrolü
# Streamlit API Anahtarını Streamlit Secrets'tan alacak
try:
    # API key bulunamazsa uygulama durur. Bu kontrol doğru bırakıldı.
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    # Kullanıcıya hata mesajı göster ve uygulamayı durdur.
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta 'GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Cache kaldırıldı.
def load_rag_data(api_key):
    # data.py dosyasındaki ana fonksiyonu çağırır ve metin parçalarını yükler
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
text_chunks, index_placeholder = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if not text_chunks:
    st.error(f"Veri seti yüklenemedi. Lütfen '{CORRECT_FILE_NAME}' dosyasının GitHub'da olduğundan ve data.py ile aynı dizinde bulunduğundan emin olun.")
    st.stop()

# --- 2. MODEL VE SORGULAMA FONKSİYONU (app.py'de kalmalıysa) ---
# Eğer simple_query_streamlit fonksiyonunuz burada ise, buraya ekleyin.
# DÜZELTME: Bu fonksiyonun artık text_chunks'ı parametre olarak alması gerekir.
# Eğer app.py'de ise:
# def simple_query_streamlit(prompt):
#     # ... (Mevcut kodunuzu koruyun)
#     # Düzeltme: Sorgulama yaparken text_chunks değişkenini kullanmayı unutmayın
#     pass
    
# --- 3. UYGULAMA MANTIĞI VE ARAYÜZ ---
st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡", layout="wide")
st.title("💡 Enerji Verimliliği AI Chatbot")
# DÜZELTME: Sınırlı içerik uyarısı kaldırıldı, artık tüm kitaba erişim var.
st.markdown("Enerji dünyasındaki bilgelik parmaklarınızın ucunda. Kitabın **tüm içeriğine** erişimim var. ⚡")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{\"role\": \"assistant\", \"content\": \"Merhaba! Enerji verimliliği konularında sorularınızı yanıtlamaya hazırım. Artık **tüm kitaptaki bilgilere** erişimim var. ⚡\"}]

for msg in st.session_state.messages:
    avatar = "💡" if msg["role"] == "assistant" else "👤" 
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Enerji verimliliği hakkında bir soru sorun... 📝"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    with st.spinner("Kitabın tamamı taranıyor ve cevap üretiliyor... 🧠"):
        # KRİTİK DÜZELTME: Sorgu fonksiyonuna text_chunks parametresi eklendi
        response, source = simple_query_streamlit(prompt, text_chunks, GEMINI_API_KEY)

    # Cevap, spinner bittikten sonra chat_message içinde yazdırılıyor...
    st.chat_message("assistant", avatar="💡").write(response)
    if source:
        st.markdown(f"**Kaynak Metin:** {source[:300]}...") # Kaynağın ilk 300 karakterini göster
# Streamlit web uygulamasının ana dosyası
import streamlit as st
from google import genai
# import numpy kaldırıldı
import os
# data.py dosyasındaki fonksiyonu içe aktarıyoruz
from data import prepare_rag_data 

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---

CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

# Streamlit API Anahtarını Streamlit Secrets'tan alacak
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Cache kaldırıldı.
def load_rag_data(api_key):
    # data.py dosyasındaki ana fonksiyonu çağırır ve metin parçalarını yükler
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
# index_placeholder, data.py'den gelen None değerini tutar.
text_chunks, index_placeholder = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if not text_chunks:
    st.error(f"Veri seti yüklenemedi. Lütfen '{CORRECT_FILE_NAME}' dosyasının GitHub'da olduğundan ve data.py'nin güncel olduğundan emin olun.")
    st.stop()

# Client'ı sadece Gemini modelini çağırmak için bir kez tanımlıyoruz.
client = genai.Client(api_key=GEMINI_API_KEY) 

# --- 2. TEMEL SORGULAMA FONKSİYONU (RAG Kaldırıldı) ---
def simple_query_streamlit(query): 
    
    # FAISS ARAMASI VE VEKTÖRLEŞTİRME KALDIRILDI.
    # Sadece ilk 1000 karakteri (text_chunks[0]) bağlam olarak kullanılır.
    retrieved_text = text_chunks[0] 
    
    # Prompt oluşturma
    prompt = f"""
    Sen, "Enerji Verimliliği Eğitim Kitabı"ndan bilgi alan uzman bir danışmansın. 
    Senin tek bilgi kaynağın aşağıdaki Bağlam (Context) içinde yer alan metinlerdir.

    1. **AKIL YÜRÜTME:** Kitaptan Çekilen Kaynaklar ışığında, kapsamlı bir cevap üret. Cevabının doğruluğu, sadece bu kaynaklara dayanmalıdır.
    2. **KAYNAK KISITLAMASI:** Bağlamda soruya net ve doğrudan cevap verecek bilgi **bulunmuyorsa**, yanıtın: "Bu konuda eğitim kitabımda yeterli ve güncel bilgi bulunmamaktadır." şeklinde olmalıdır.

    Cevabını anlaşılır, profesyonel bir dille ve ilgili emojilerle sun.
    
    ---
    
    Bağlam (Kitaptan Çekilen Kaynaklar - Sadece İlk 1000 Karakter):
    {retrieved_text}

    ---
    
    Soru:
    {query}
    """
    
    # Cevap Üretme (Generation)
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt
    )
    
    return response.text, retrieved_text

# --- 3. UYGULAMA MANTIĞI VE ARAYÜZ ---
st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡", layout="wide")
st.title("💡 Enerji Verimliliği AI Chatbot")
st.markdown("Enerji dünyasındaki bilgelik parmaklarınızın ucunda. Şu an sadece kitabın **giriş bölümüne** erişim vardır. ⚡")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Enerji verimliliği konularında sorularınızı yanıtlamaya hazırım. Şu an sadece **sınırlı bir içeriğe** erişimim var. ⚡"}]

for msg in st.session_state.messages:
    avatar = "💡" if msg["role"] == "assistant" else "👤" 
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Enerji verimliliği hakkında bir soru sorun... 📝"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    # Spinner'ı, chat_message'ın dışına taşıyarak DOM hatasını çözüyoruz.
    with st.spinner("Kitabın sınırlı bölümü taranıyor ve cevap üretiliyor... 🧠"):
        response, source = simple_query_streamlit(prompt)

    # Cevap, spinner bittikten sonra chat_message içinde yazdırılıyor
    with st.chat_message("assistant", avatar="💡"):
        st.markdown(response)

        with st.expander("🔍 Kullanılan Kaynak (Sadece İlk 1000 Karakter)"):
            st.code(source, language='text')

    st.session_state.messages.append({"role": "assistant", "content": response})
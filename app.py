import streamlit as st
from google import genai
import numpy as np
import os
from data import prepare_rag_data 

# --- Sayfa Konfigürasyonu (YENİ) ---
st.set_page_config(
    page_title="Enerji Verimliliği RAG Uzmanı",
    page_icon="💡",
    layout="wide"
)

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Cache kullanarak verinin sadece BİR KEZ yüklenmesini sağlıyoruz
@st.cache_resource(ttl=900) 
def load_rag_data(api_key):
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
text_chunks, index = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if index is None:
    file_name = "Enerji_verimliligi_eğitim_kitabi.txt" 
    st.error(f"Veri seti yüklenemedi. Lütfen '{file_name}' dosyasının GitHub'da bulunduğundan emin olun.")
    st.stop()

# Client'ı sadece Gemini modelini çağırmak için bir kez tanımlıyoruz.
client = genai.Client(api_key=GEMINI_API_KEY) 


# --- 2. RAG Sorgu Fonksiyonu ---
def rag_query_streamlit(query, k=5):
    
    # 2A. Sorguyu Vektörleştirme (Retrieval)
    query_embedding = client.models.embed_content(
        model='text-embedding-004',
        contents=[query]
    ).embeddings[0].values
    
    # 2B. FAISS'te en yakın parçaları arama
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)
    
    # 2C. Parçaları metin olarak birleştirme
    retrieved_text = "\n---\n".join([text_chunks[i] for i in I[0]])
    
    # 2D. Prompt oluşturma (System Prompt ile)
    prompt = f"""
    Sen, "Enerji Verimliliği Eğitim Kitabı"ndan bilgi alan uzman bir danışmansın.
    Görevin, öncelikli olarak **aşağıdaki Bağlam (Context)** içinde yer alan bilgilere dayanarak cevap vermektir.
    
    Talimatlar:
    1. Kitaptaki bilgiyi anlaşılır, akıcı ve profesyonel bir dille **özetle ve yorumla**. Asla metni olduğu gibi kopyalama.
    2. Cevabın, kitaptan gelen bilgiyle tutarlı olmalıdır.
    3. **Çok önemli:** Eğer kitaptan gelen Bağlam (Context) cevabı desteklemiyorsa, kendi **genel bilgini ve mantığını** kullanarak en iyi tahmini veya bilgiyi sun, ancak cevabına şunu ekle: "(Kitapta bu konuyla ilgili net bir bilgi bulunmamaktadır, bu genel bir bilgidir.)"
    4. Cevabını Soruya özel olarak kişiselleştir.

    ---
    
    Bağlam:
    {retrieved_text}

    ---
    
    Soru:
    {query}
    """
    
    # 2E. Cevap Üretme (Generation)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    return response.text, retrieved_text

# --- 3. UYGULAMA MANTIĞI VE ARAYÜZ ---
st.title("💡 Enerji Verimliliği RAG Uzmanı") 
st.markdown("Merhaba! Bu uzman chatbot, 'Enerji Verimliliği Eğitim Kitabı'ndan alınan **güncel bilgilere** 📚 dayanarak sorularınızı yanıtlar. **Enerji tasarrufu** 🌍 ve **verimlilik** konusunda hemen soru sormaya başlayın! 👇") # EMOJİLER EKLENDİ

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Enerji verimliliği hakkında sorularınızı yanıtlamak için buradayım."}]

for msg in st.session_state.messages:
    # Emoji eklendi
    avatar = "💡" if msg["role"] == "assistant" else "👤" 
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Enerji tasarrufu ipuçları mı arıyorsunuz? 🔍 Bir soru sorun..."): # SORU KUTUSUNA EMOJİ EKLENDİ
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    with st.chat_message("assistant", avatar="💡"):
        with st.spinner("Enerji Verimliliği Kitabı taranıyor... ⏳"): # SPINNER'A EMOJİ EKLENDİ
            response, source = rag_query_streamlit(prompt)

        st.markdown(response)

        with st.expander("Kullanılan Kaynak (RAG Retrieval) 📖"): # EXPANDER BAŞLIĞINA EMOJİ EKLENDİ
            st.code(source, language='text')

    st.session_state.messages.append({"role": "assistant", "content": response})

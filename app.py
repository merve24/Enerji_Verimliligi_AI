import streamlit as st
from google import genai
import numpy as np
import os
# data.py dosyasındaki fonksiyonu içe aktarıyoruz
from data import prepare_rag_data 

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---

# Streamlit API Anahtarını Streamlit Secrets'tan alacak
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Cache kullanarak verinin sadece BİR KEZ yüklenmesini sağlıyoruz
@st.cache_resource(ttl=900) 
def load_rag_data(api_key):
    # data.py dosyasındaki ana fonksiyonu çağır
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
text_chunks, index = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if index is None:
    # data.py dosyasındaki güncel dosya adını kontrol etmek için
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
    Aşağıdaki Bağlam (Context) içinde yer alan bilgilere dayanarak ver. Eğer bağlamda cevap yoksa,
    "Bu sorunun cevabına sahip olduğum Enerji Verimliliği Kitabında net bir bilgi bulunmamaktadır." diye cevap ver.
    Cevabını anlaşılır ve profesyonel bir dille sun.

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
st.title("💡 Enerji Verimliliği RAG Chatbot'u")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Enerji verimliliği hakkında sorularınızı yanıtlamak için buradayım."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Enerji verimliliği veya sürdürülebilirlik hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):        with st.spinner("Enerji Verimliliği Kitabı taranıyor..."):            response, source = rag_query_streamlit(prompt)

        st.markdown(response)

        with st.expander("Kullanılan Kaynak (RAG Retrieval)"):            st.code(source, language='text')

    st.session_state.messages.append({"role": "assistant", "content": response})

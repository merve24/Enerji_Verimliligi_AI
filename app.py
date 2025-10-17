# Streamlit web uygulamasının ana dosyası
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
# ttl=900, 15 dakika boyunca önbellekte tutulur.
@st.cache_resource(ttl=900) 
def load_rag_data(api_key):
    # data.py dosyasındaki ana fonksiyonu çağırır ve FAISS indexini yükler
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
    # D: Uzaklıklar, I: İndeksler
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)
    
    # 2C. Parçaları metin olarak birleştirme
    retrieved_text = "\n---\n".join([text_chunks[i] for i in I[0]])
    
    # 2D. Prompt oluşturma (Akıl Yürütmeye İzin Veren System Prompt)
    prompt = f"""
    Sen, "Enerji Verimliliği Eğitim Kitabı"ndan bilgi alan uzman bir danışmansın. 
    Öncelikle, aşağıdaki Bağlam (Context) içinde yer alan bilgilere dayanarak kapsamlı ve kişiselleştirilmiş bir cevap üret.
    
    Eğer bağlamda soruya net ve doğrudan cevap verecek bilgi bulunmuyorsa, KENDİ GENEL BİLGİNİ kullanarak konuya açıklık getir (Kitaba bağlı kalmak zorunda değilsin, ancak bilgiyi teyit et).
    
    Cevabını anlaşılır, profesyonel bir dille ve ilgili emojilerle sun.
    
    ---
    
    Bağlam (Kitaptan Çekilen Kaynaklar):
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

# Sayfa Yapılandırması (st.set_page_config'i ilk komut olarak kullanmak önemlidir)
st.set_page_config(
    page_title="Enerji Verimliliği AI", 
    page_icon="💡", 
    layout="wide"
)

# Yeni isim ve açıklama buraya eklendi
st.title("💡 Enerji Verimliliği AI Chatbot")
st.markdown("Enerji dünyasındaki 1000 sayfalık 📚 bilgelik parmaklarınızın ucunda. Bu uzman 🤖 AI, size en güncel ve güvenilir bilgileri anında sunar.")

# Session State ile mesaj geçmişini yönetme
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Enerji verimliliği ve sürdürülebilirlik konularında sorularınızı yanıtlamaya hazırım. Size nasıl yardımcı olabilirim? ⚡"}]

# Mesaj geçmişini arayüze yazdırma
for msg in st.session_state.messages:
    # Emojilerle rol belirleme
    avatar = "💡" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# Kullanıcı girişi ve cevap üretme
if prompt := st.chat_input("Enerji verimliliği, sürdürülebilirlik veya çevre hakkında bir soru sorun... 📝"):
    # Kullanıcı mesajını geçmişe ekle ve yazdır
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    # Asistan cevabını üretme
    with st.chat_message("assistant", avatar="💡"):
        with st.spinner("Enerji Verimliliği Kitabı taranıyor ve akıl yürütülüyor... 🧠"):
            response, source = rag_query_streamlit(prompt)

        st.markdown(response)

        # RAG kaynağını gösterme (Şeffaflık için)
        with st.expander("🔍 Kullanılan Kaynak (RAG Retrieval)"):
            st.code(source, language='text')

    # Asistan cevabını geçmişe ekle
    st.session_state.messages.append({"role": "assistant", "content": response})
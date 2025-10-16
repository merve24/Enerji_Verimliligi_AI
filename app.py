import streamlit as st
from google import genai
import numpy as np
import os
from data import prepare_rag_data # data.py dosyasını içe aktardık

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---

# Streamlit API Anahtarını Streamlit Secrets'tan alacak
try:
    # 💥 Streamlit Cloud'da secret olarak tanımlanacak
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    # Lokal test veya hata durumunda boş bırakılırsa hata verecektir.
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Cache kullanarak verinin sadece BİR KEZ yüklenmesini sağlıyoruz
@st.cache_resource
def load_rag_data(api_key): # API KEY'İ PARAMETRE OLARAK ALIYOR
    # prepare_rag_data'ya API key'i gönderiyoruz
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
text_chunks, index = load_rag_data(GEMINI_API_KEY) 

# Client'ı sadece Gemini modelini çağırmak için bir kez tanımlıyoruz.
# Embedding için ayrı, Generation için ayrı Client kullanmak yerine, Generation için Client'ı burada tutuyoruz.
client = genai.Client(api_key=GEMINI_API_KEY) 


# --- 2. RAG FONKSİYONUNU TANIMLAMA ---
# RAG fonksiyonu artık client'ı globalden (veya parametre olarak) alabilir.
def rag_query_streamlit(query, top_k=3, text_chunks=text_chunks, index=index):
    
    # 5A. Sorguyu Vektörleştirme
    # Bu client, yukarıda tanımlanan GEMINI_API_KEY ile oluşturulmuştur.
    query_embedding_response = client.models.embed_content(
        model='text-embedding-004',
        contents=[query] 
    )
    query_embedding = np.array(query_embedding_response.embeddings[0].values, dtype='float32').reshape(1, -1)

    # 5B. Veritabanından Alakalı Bilgiyi Çekme (Retrieval)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_text = "".join(text_chunks[i] + "\n---\n" for i in indices[0])

    # 5C. Prompt Oluşturma
    prompt = f"""
    Sen, 'Sürdürülebilir İşletme Enerji Danışmanı' adlı RAG temelli bir yapay zeka chatbot'usun.
    Aşağıdaki 'Bağlam' kısmında sana verilen bilgileri kullanarak, 'Soru' kısmındaki kullanıcı sorusunu yanıtla.
    Cevabını yalnızca sana verilen bağlamdaki bilgilere dayanarak ver. Eğer bağlamda cevap yoksa,
    "Bu sorunun cevabına sahip olduğum Enerji Verimliliği Kitabında net bir bilgi bulunmamaktadır." diye cevap ver.
    Cevabını anlaşılır ve profesyonel bir dille sun.

    ---
    
    Bağlam:
    {retrieved_text}

    ---
    
    Soru:
    {query}
    """
    
    # 5D. Cevap Üretme (Generation)
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

    with st.chat_message("assistant"):
        with st.spinner("Enerji Verimliliği Kitabı taranıyor..."):
            response, source = rag_query_streamlit(prompt)
        
        st.markdown(response)
        
        with st.expander("Kullanılan Kaynak (RAG Retrieval)"):
            st.code(source, language='text')

    st.session_state.messages.append({"role": "assistant", "content": response})

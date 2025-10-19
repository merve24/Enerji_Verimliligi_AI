# Streamlit web uygulamasının ana dosyası
import streamlit as st
from google import genai
import numpy as np
import os
# data.py dosyasındaki fonksiyonu içe aktarıyoruz
from data import prepare_rag_data 

# --- 1. SABİT VERİLERİ VE BAĞLANTILARI TANIMLAMA ---

# Dosya adı, hata mesajlarında doğru adın görünmesi için burada tanımlanır.
CORRECT_FILE_NAME = "Enerji_verimliligi_eğitim_kitabi.txt"

# Streamlit API Anahtarını Streamlit Secrets'tan alacak
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    st.error(f"API Anahtarı Yüklenemedi: Lütfen Streamlit Cloud Secrets'ta GEMINI_API_KEY'i tanımladığınızdan emin olun. Hata: {e}")
    st.stop()
    
# Cache kullanarak verinin sadece BİR KEZ yüklenmesini sağlıyoruz
@st.cache_resource(ttl=900) 
def load_rag_data(api_key):
    # data.py dosyasındaki ana fonksiyonu çağırır ve FAISS indexini yükler
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
text_chunks, index = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if index is None:
    st.error(f"Veri seti yüklenemedi. Lütfen '{CORRECT_FILE_NAME}' dosyasının GitHub'da app.py ve data.py ile aynı dizinde bulunduğundan emin olun.")
    st.stop()

# Client'ı sadece Gemini modelini çağırmak için bir kez tanımlıyoruz.
client = genai.Client(api_key=GEMINI_API_KEY) 

# --- 2. RAG Sorgu Fonksiyonu ---
def rag_query_streamlit(query, k=8): 
    
    # 2A. Sorguyu Vektörleştirme (Retrieval)
    response_embedding = client.models.embed_content(
        model='text-embedding-004',
        contents=[query] # Sorguyu liste olarak gönderiyoruz
    )
    
    # 2B. Gömülü Vektöre Doğru Erişim
    try:
        # API yanıtı bir sözlüktür ve vektör 'embedding' anahtarı altındadır.
        # Sorguyu liste içinde [query] gönderdiğimiz için, dönen embedding de bir listedir.
        # Bu yüzden ilk elemanı [0] ile alıyoruz.
        query_embedding_list = response_embedding['embedding'][0]
        
        # NumPy array'ine dönüştürme (FAISS için gerekli)
        query_embedding = np.asarray(query_embedding_list, dtype='float32')
        
    except (AttributeError, KeyError, IndexError) as e:
        st.error(f"Vektörleştirme API yanıtı hatası: Gömülü vektör verisine erişilemiyor. Detay: {e}")
        raise ValueError("API'den gelen gömülü vektör verisine erişilemiyor.")

    # FAISS'te en yakın parçaları arama
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)
    
    # 2C. Parçaları metin olarak birleştirme
    retrieved_text = "\n---\n".join([text_chunks[i] for i in I[0]])
    
    # 2D. Prompt oluşturma
    prompt = f"""
    Sen, "Enerji Verimliliği Eğitim Kitabı"ndan bilgi alan uzman bir danışmansın. 
    Senin tek bilgi kaynağın aşağıdaki Bağlam (Context) içinde yer alan metinlerdir.

    1. **AKIL YÜRÜTME:** Kitaptan Çekilen Kaynaklar ışığında, akıl yürüterek kapsamlı ve kişiselleştirilmiş bir cevap üret. Cevabının doğruluğu, sadece bu kaynaklara dayanmalıdır.
    2. **KAYNAK KISITLAMASI:** Bağlamda soruya net ve doğrudan cevap verecek bilgi **bulunmuyorsa**, **KESİNLİKLE KENDİ GENEL BİLGİNİ KULLANMA** ve akıl yürütmeye çalışma.
    3. **RED YANITI:** Eğer cevap veremeyeceksen, yanıtın: "Bu konuda eğitim kitabımda yeterli ve güncel bilgi bulunmamaktadır." şeklinde olmalıdır.

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
        model='gemini-1.5-flash', # Model adını güncelledim, '2.5-flash' henüz genel kullanıma açık olmayabilir. '1.5-flash' daha güvenli bir seçimdir.
        contents=prompt
    )
    
    return response.text, retrieved_text

# --- 3. UYGULAMA MANTIĞI VE ARAYÜZ ---
st.set_page_config(page_title="Enerji Verimliliği AI", page_icon="💡", layout="wide")
st.title("💡 Enerji Verimliliği AI Chatbot")
st.markdown("Enerji dünyasındaki 1000 sayfalık 📚 bilgelik parmaklarınızın ucunda. Bu uzman 🤖 AI, size en güncel ve güvenilir bilgileri anında sunar.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Enerji verimliliği ve sürdürülebilirlik konularında sorularınızı yanıtlamaya hazırım. Size nasıl yardımcı olabilirim? ⚡"}]

for msg in st.session_state.messages:
    avatar = "💡" if msg["role"] == "assistant" else "👤" 
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Enerji verimliliği, sürdürülebilirlik veya çevre hakkında bir soru sorun... 📝"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    with st.chat_message("assistant", avatar="💡"):
        with st.spinner("Enerji Verimliliği Kitabı taranıyor ve akıl yürütülüyor... 🧠"):
            response, source = rag_query_streamlit(prompt)

        st.markdown(response)

        with st.expander("🔍 Kullanılan Kaynak (RAG Retrieval)"):
            st.code(source, language='text')

    st.session_state.messages.append({"role": "assistant", "content": response})
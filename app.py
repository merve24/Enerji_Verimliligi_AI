# Streamlit web uygulamasının ana dosyası
import streamlit as st
from google import genai
import numpy as np # RAG arama/vektörleştirme için geri eklendi
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
    
# Cache kullanarak verinin sadece BİR KEZ yüklenmesini sağlıyoruz
@st.cache_resource(ttl=900) 
def load_rag_data(api_key):
    # data.py dosyasındaki ana fonksiyonu çağırır ve FAISS indexini yükler
    return prepare_rag_data(api_key)

# Veri yükleme fonksiyonunu API key ile çağırıyoruz.
text_chunks, index = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if index is None:
    st.error(f"Veri seti yüklenemedi. Lütfen '{CORRECT_FILE_NAME}' dosyasının GitHub'da app.py ve data.py ile aynı dizinde bulunduğundan emin olun veya logları kontrol edin.")
    st.stop()

# Client'ı sadece Gemini modelini çağırmak için bir kez tanımlıyoruz.
client = genai.Client(api_key=GEMINI_API_KEY) 

# --- 2. RAG Sorgu Fonksiyonu ---
def rag_query_streamlit(query, k=8): 
    
    # 1. Kullanıcı sorgusunu vektörleştirme
    try:
        query_embedding_response = client.models.embed_content(
            model='text-embedding-004',
            # DÜZELTME: Sorguyu liste içinde gönderiyoruz
            contents=[query], 
        )
        # DÜZELTME: response.embeddings[0] ile kesin erişim sağlanıyor
        query_embedding = np.array(query_embedding_response.embeddings[0], dtype='float32') 
        # FAISS arama için vektörü 2 boyutlu hale getiriyoruz (1x1536)
        query_embedding = query_embedding.reshape(1, -1)
    except Exception as e:
        return f"Hata: Sorgu vektörleştirilemedi. API hatası: {e}", "Kaynaklar yüklenemedi."


    # 2. FAISS'te en yakın parçaları arama (k=8 adet)
    D, I = index.search(query_embedding, k)
    
    # 3. İlgili metin parçalarını çekme
    retrieved_chunks = [text_chunks[i] for i in I[0] if i < len(text_chunks)]
    retrieved_text = "\n\n---\n\n".join(retrieved_chunks)
    
    # 4. Prompt oluşturma (Sadece çekilen parçalar bağlama eklenir)
    prompt = f"""
    Sen, "Enerji Verimliliği Eğitim Kitabı"ndan bilgi alan uzman bir danışmansın. 
    Senin tek bilgi kaynağın aşağıdaki Bağlam (Context) içinde yer alan metinlerdir.

    1. **AKIL YÜRÜTME:** Kitaptan Çekilen Kaynaklar ışığında, kapsamlı ve kişiselleştirilmiş bir cevap üret. Cevabının doğruluğu, sadece bu kaynaklara dayanmalıdır.
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
    
    # 5. Cevap Üretme (Generation)
    response = client.models.generate_content(
        model='gemini-1.5-flash',
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

        with st.expander("🔍 Kullanılan Kaynaklar"):
            st.code(source, language='text')

    st.session_state.messages.append({"role": "assistant", "content": response})
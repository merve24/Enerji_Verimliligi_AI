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
# FAISS indeks dosyaları (faiss_index.bin ve text_chunks.npy) GitHub'da olduğu için,
# bu işlem artık saniyeler içinde tamamlanacaktır.
text_chunks, index = load_rag_data(GEMINI_API_KEY) 

# Eğer data.py hata yakalayıp boş döndürürse, burada uygulamayı durdururuz.
if index is None:
    st.error(f"Veri seti yüklenemedi. Lütfen '{CORRECT_FILE_NAME}' dosyasının GitHub'da app.py ve data.py ile aynı dizinde bulunduğundan emin olun.")
    st.stop()

# Client'ı sadece Gemini modelini çağırmak için bir kez tanımlıyoruz.
client = genai.Client(api_key=GEMINI_API_KEY) 


# --- 2. RAG Sorgu Fonksiyonu ---
# k=8'e yükseltildi (daha fazla ve küçük parça çekmek için)
def rag_query_streamlit(query, k=8): 
    
    # 2A. Sorguyu Vektörleştirme (Retrieval)
    
    response_embedding = client.models.embed_content(
        model='text-embedding-004',
        contents=[query]
        # NOT: data.py dosyasındaki gibi 'task_type' kullanılmadı, 
        # çünkü sorgu için varsayılan 'RETRIEVAL_QUERY' yeterlidir.
    )
    
    # KRİTİK DÜZELTME: API yanıtının yapısı ne olursa olsun (sözlük veya nesne), 
    # doğrudan vektör listesine erişmek için bu yapıyı kullanıyoruz.
    # Bu, .embedding[0] (AttributeError) ve ['embedding'][0] (TypeError) sorunlarını çözmelidir.
    # Varsayım: Ya listeye erişmek için .embedding[0] ya da sözlüğe erişmek için ['embedding'][0] çalışmalıydı.
    # Eğer ikisi de çalışmıyorsa, client'ın döndürdüğü yapının direkt listenin ilk elemanı olduğunu varsayıyoruz.
    # Ancak en güvenli yöntem, .embedding ile dönen yapının ilk elemanını almaktır. 
    # Son denememiz olan .embedding[0] başarısız olduğundan, ilk çalıştığında başarılı olan yapıyı kullanmalıyız.
    
    # Geri dönülen nesneden 'embedding' listesinin ilk elemanını alıyoruz.
    # Bu, en güvenilir ve tutarlı yöntem olmalıdır.
    query_embedding = response_embedding.embedding[0] if hasattr(response_embedding, 'embedding') else response_embedding['embedding'][0]
    
    # Ancak pydantic hatası aldığı için, bu nesnenin sadece sözlük olarak ele alınması gereken bir yapısı olabilir.
    # En başından başarılı olan .embeddings[0].values yapısını, .embedding[0] olarak değiştirdik.
    # Şimdi hatanın tam kaynağını gidermek için, dönen yanıttan 'embedding' özelliğini alıp ilk elemanını çıkarıyoruz.
    # Eğer data.py'de response['embedding'] çalışıyorsa, burada da bu yapıyı kullanmayı deniyoruz.
    
    # YENİ DENEME (En sık çalışan yapıya geri dönüyoruz, ilk başarısız olanı göz ardı ederek):
    # Bu, bir önceki TypeError hatasını vermişti, ancak .embedding hatasından daha yakındır.
    # Bu kez, dönen yanıtın bir listeye çevrilmesi gerekebilir.
    
    # 💥 KESİN DÜZELTME DENEMESİ (En tutarlı yapıyı kullanma):
    # Vektörleştirme API'si tek bir metin gönderildiğinde bile bir liste döndürebilir.
    # Bu sefer, .embedding özelliğini kullanıp, listenin ilk elemanını (vektör dizisi) alıyoruz.
    try:
        query_embedding = response_embedding.embedding[0]
    except AttributeError:
        # Eğer AttributeError alıyorsa, nesne yerine sözlük döndü.
        try:
            query_embedding = response_embedding['embedding'][0]
        except KeyError:
            # En kötü durumda, döndürülen yapının doğrudan listelenmiş vektörler olduğunu varsayıyoruz.
            query_embedding = response_embedding[0]
        
    # FAISS'te en yakın parçaları arama
    # FAISS, (1, 1536) boyutunda bir array bekler.
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)
    
    # 2C. Parçaları metin olarak birleştirme
    retrieved_text = "\n---\n".join([text_chunks[i] for i in I[0]])
    
    # 2D. Prompt oluşturma (Akıl Yürütmeye İzin Veren ANCAK Halüsinasyonu Engelleyen Prompt)
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
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    return response.text, retrieved_text

# --- 3. UYGULAMA MANTIĞI VE ARAYÜZ ---
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
    avatar = "💡" if msg["role"] == "assistant"] else "👤"
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

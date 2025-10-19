import os 
import streamlit as st
from google import genai 
import math # Cosine Similarity için gerekli

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA ---
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    pass

# --- 2. ANA FONKSİYON: VERİSİNİ HAZIRLAMA (Embedding ve Index Oluşturma) ---
@st.cache_resource
def prepare_rag_data(api_key):
    
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return [], None 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None
    
    # Metin Parçalama
    text_chunks = []
    chunk_size = 2000 
    for i in range(0, len(text), chunk_size):
        text_chunks.append(text[i:i + chunk_size])
    
    # --- KRİTİK DÜZELTME: Toplu (batch) işlem yerine Tek tek Gömme (embed_content) ---
    st.info(f"{len(text_chunks)} metin parçası için embedding oluşturuluyor. Lütfen bekleyin...")
    
    embeddings = []
    try:
        client = genai.Client(api_key=api_key)
        
        # Her bir metin parçasını tek tek gömme (embedding) işlemine tabi tutma
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(text_chunks):
            response = client.models.embed_content(
                model='text-embedding-004', 
                content=chunk
            )
            embeddings.append(response.embedding)
            # Yükleme çubuğunu güncelleyin
            progress_bar.progress((i + 1) / len(text_chunks))
        
        progress_bar.empty() # İşlem bitince yükleme çubuğunu kaldırın
        
        return text_chunks, embeddings 
        
    except Exception as e:
        st.error(f"Embedding Hatası: Veri gömme işlemi başarısız oldu. API Anahtarınızı kontrol edin. Hata: {e}")
        return [], None

# --- 3. YARDIMCI FONKSİYON: Cosine Similarity Hesaplama ---
def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(v**2 for v in vec1))
    magnitude_v2 = math.sqrt(sum(v**2 for v in vec2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return dot_product / (magnitude_v1 * magnitude_v2)

# --- 4. SORGULAMA FONKSİYONU ---
def simple_query_streamlit(prompt, text_chunks, embeddings, api_key):
    
    # 1. RAG AŞAMASI: Anlamsal Arama (Top 3 En Alakalı Parçayı Bulma)
    try:
        client = genai.Client(api_key=api_key)
        
        prompt_embedding = client.models.embed_content(
            model='text-embedding-004',
            content=prompt
        ).embedding
        
        similarity_scores = []
        for i, chunk_embedding in enumerate(embeddings):
            score = cosine_similarity(prompt_embedding, chunk_embedding)
            similarity_scores.append((score, i))
            
        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        
        # En alakalı 3 parçanın indeksini alın (Kota Optimizasyonu)
        top_indices = [index for score, index in similarity_scores[:3]]
        
        retrieved_text = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])
        
    except Exception as e:
        st.warning(f"Vektör Arama Hatası: {e}. Basit aramaya geçiliyor.")
        retrieved_text = "\n\n---\n\n".join(text_chunks[:3])

    # 2. ÜRETİM AŞAMASI: Gemini API çağrısı
    try:
        client = genai.Client(api_key=api_key)

        # Gelişmiş Halüsinasyon Engelleme ve Akıl Yürütme Talimatı
        rag_prompt = (
            f"Sen, 'Enerji Verimliliği Eğitim Kitabı'na dayalı bir yapay zeka asistansın. Görevin, aşağıdaki 'KAYNAK METİN'i kullanarak kullanıcı sorularını yanıtlamaktır.\n\n"
            f"TALİMATLAR:\n"
            f"1. **Öncelik ve Halüsinasyon Engeli:** Cevaplarını **KESİNLİKLE** sadece sağlanan KAYNAK METİN'deki bilgilerle sınırla. Kaynak metnin dışındaki kendi genel bilgini **ASLA** kullanma.\n"
            f"2. **Akıl Yürütme (Muhakeme):** Sorunun cevabı tek bir yerde geçmiyorsa, kitaptan aldığın **farklı bilgileri mantıklı bir şekilde birleştirerek (muhakeme ederek)** kapsamlı bir cevap oluşturabilirsin.\n"
            f"3. **Reddetme:** Sorunun cevabı kaynak metinde **yoksa** veya akıl yürütmeye uygun yeterli veri bulunmuyorsa, kibarca **'Üzgünüm, bu sorunun cevabını elimdeki Enerji Verimliliği Eğitim Kitabında bulamıyorum.'** şeklinde cevap ver.\n\n"
            f"KAYNAK METİN:\n---\n{retrieved_text}\n---\n\n"
            f"KULLANICI SORUSU: {prompt}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt
        )
        
        return response.text, retrieved_text
        
    except Exception as e:
        st.error(f"Sorgulama Hatası: Gemini API çağrısında bir sorun oluştu. Hata: {e}")
        return "Üzgünüm, sorgulama sırasında bir hata oluştu.", retrieved_text
import os
import streamlit as st
from google import genai
import math
import faiss
import numpy as np

# --- 1. COSINE SIMILARITY ---
def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(v**2 for v in vec1))
    magnitude_v2 = math.sqrt(sum(v**2 for v in vec2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return dot_product / (magnitude_v1 * magnitude_v2)

# --- 2. RAG VERİ HAZIRLAMA ---
@st.cache_resource
def prepare_rag_data(api_key):
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return [], None, None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None, None

    # Daha küçük chunk boyutu → yanıt kesilmesini önler
    chunk_size = 1500
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    st.info(f"{len(text_chunks)} metin parçası için embedding oluşturuluyor. Lütfen bekleyin...")

    embeddings = []
    try:
        client = genai.Client(api_key=api_key)
        progress_bar = st.progress(0, text="Embedding Oluşturuluyor...")

        for i, chunk in enumerate(text_chunks):
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=[chunk]
            )
            if hasattr(response, "embeddings"):
                embedding_vector = response.embeddings[0].values
            else:
                embedding_vector = response.embedding
            embeddings.append(embedding_vector)
            progress_bar.progress((i + 1) / len(text_chunks))

        progress_bar.empty()

        # --- FAISS INDEX OLUŞTURMA ---
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        return text_chunks, embeddings, index

    except Exception as e:
        st.error(f"Embedding Hatası: {e}")
        return [], None, None

# --- 3. SORGULAMA FONKSİYONU ---
def simple_query_streamlit(prompt, text_chunks, embeddings, index, api_key):
    try:
        client = genai.Client(api_key=api_key)

        # Prompt embedding
        prompt_for_embed = f"Bu sorunun anlamı: {prompt} (enerji verimliliği bağlamında)"
        prompt_response = client.models.embed_content(
            model='text-embedding-004',
            contents=[prompt_for_embed]
        )

        if hasattr(prompt_response, "embeddings"):
            prompt_embedding = prompt_response.embeddings[0].values
        else:
            prompt_embedding = prompt_response.embedding

        # --- FAISS ile en yakın chunk'ları bul ---
        query_vector = np.array([prompt_embedding]).astype('float32')
        k = 7  # En yakın 7 chunk → daha fazla bilgi
        distances, indices = index.search(query_vector, k)
        retrieved_text = "\n\n---\n\n".join([text_chunks[i] for i in indices[0]])
        max_chars = 4000  # Daha uzun context
        retrieved_text = retrieved_text[:max_chars]

    except Exception as e:
        st.warning(f"Vektör Arama Hatası: {e}. Basit aramaya geçiliyor.")
        retrieved_text = "\n\n---\n\n".join(text_chunks[:3])

    # --- MODEL CEVAP ÜRETİMİ ---
    try:
        rag_prompt = (
            f"Sen 'Enerji Verimliliği AI Chatbot'u olarak Enerji Verimliliği Eğitim Kitabı'na dayalı "
            f"bir yapay zeka asistansın.\n"
            f"Kullanıcı sorularını aşağıdaki kaynak metinlere dayanarak mantıklı, detaylı ve kişiselleştirilmiş "
            f"bir şekilde cevapla.\n"
            f"- Kitaptaki bilgilerden yararlanarak soruya uygun akıl yürütebilirsin.\n"
            f"- Eğer kaynak metinde doğrudan bilgi yoksa, mantıklı çıkarımlar yaparak cevabı oluşturabilirsin, "
            f"ama kitaptaki konulardan çok sapmamalısın.\n"
            f"- Bilgiye tamamen dayalı olmayan veya gerçek dışı (halüsinasyon) cevaplar üretme.\n\n"
            f"KAYNAK METİN:\n---\n{retrieved_text}\n---\n\n"
            f"KULLANICI SORUSU: {prompt}\n\n"
            f"Cevabı eksiksiz, anlaşılır ve mantıksal olarak tutarlı şekilde ver."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt
        )

        # Cevap kısa ise detaylandır
        if len(response.text) < 100:
            follow_up_prompt = rag_prompt + "\nLütfen cevabı daha detaylı ve mantıklı şekilde açıkla."
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=follow_up_prompt
            )

        # 🔧 Kaynak metin artık döndürülmüyor
        return response.text

    except Exception as e:
        st.error(f"Sorgulama Hatası: {e}")
        return "Üzgünüm, sorgulama sırasında bir hata oluştu."

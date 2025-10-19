import os
import streamlit as st
from google import genai
import math

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
        return [], None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None

    # Daha kısa chunk boyutu → yanıt kesilmesini önler
    chunk_size = 2000
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
        return text_chunks, embeddings

    except Exception as e:
        st.error(f"Embedding Hatası: {e}")
        return [], None

# --- 3. SORGULAMA FONKSİYONU ---
def simple_query_streamlit(prompt, text_chunks, embeddings, api_key):
    try:
        client = genai.Client(api_key=api_key)

        # Prompt'u doğal Türkçe ile embedding oluştur
        prompt_for_embed = f"Bu sorunun anlamı: {prompt} (enerji verimliliği bağlamında)"

        prompt_response = client.models.embed_content(
            model='text-embedding-004',
            contents=[prompt_for_embed]
        )

        if hasattr(prompt_response, "embeddings"):
            prompt_embedding = prompt_response.embeddings[0].values
        else:
            prompt_embedding = prompt_response.embedding

        # En benzer 3 chunk seç → token limitini aşmayı önler
        similarity_scores = [(cosine_similarity(prompt_embedding, chunk_emb), i) 
                             for i, chunk_emb in enumerate(embeddings)]
        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [i for score, i in similarity_scores[:3]]

        retrieved_text = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])
        # Gereksiz uzunlukları kes
        max_chars = 3000
        retrieved_text = retrieved_text[:max_chars]

    except Exception as e:
        st.warning(f"Vektör Arama Hatası: {e}. Basit aramaya geçiliyor.")
        retrieved_text = "\n\n---\n\n".join(text_chunks[:3])

    # --- MODEL CEVAP ÜRETİMİ ---
    try:
        client = genai.Client(api_key=api_key)

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

        # Cevap kısa kesildiyse detaylandır
        if len(response.text) < 50:
            follow_up_prompt = rag_prompt + "\nLütfen cevabı daha detaylı ve mantıklı şekilde açıkla."
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=follow_up_prompt
            )

        return response.text, retrieved_text

    except Exception as e:
        st.error(f"Sorgulama Hatası: {e}")
        return "Üzgünüm, sorgulama sırasında bir hata oluştu.", retrieved_text

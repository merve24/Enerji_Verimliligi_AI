import os
import streamlit as st
from google import genai
import math

# --- 1. YARDIMCI FONKSİYON: Cosine Similarity Hesaplama ---
def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(v**2 for v in vec1))
    magnitude_v2 = math.sqrt(sum(v**2 for v in vec2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return dot_product / (magnitude_v1 * magnitude_v2)


# --- 2. ANA FONKSİYON: VERİ HAZIRLAMA (Tekli Embed) ---
@st.cache_resource
def prepare_rag_data(api_key):

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"  # Türkçe karakterleri sadeleştirdik

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

    st.info(f"{len(text_chunks)} metin parçası için embedding oluşturuluyor. Lütfen bekleyin...")

    embeddings = []
    try:
        client = genai.Client(api_key=api_key)

        progress_bar = st.progress(0, text="Embedding Oluşturuluyor...")

        # 🔧 DÜZELTME: 'content' yerine 'contents' kullanılmalı (list olarak)
        for i, chunk in enumerate(text_chunks):
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=[chunk]  # ✅ Doğru parametre
            )
            # Bazı sürümlerde response['embedding'] olabilir, ama genelde böyle döner:
            embedding_vector = response.embeddings[0].values if hasattr(response, 'embeddings') else response.embedding
            embeddings.append(embedding_vector)
            progress_bar.progress((i + 1) / len(text_chunks))

        progress_bar.empty()

        return text_chunks, embeddings

    except Exception as e:
        st.error(f"Embedding Hatası: Veri gömme işlemi başarısız oldu. Hata: {e}")
        return [], None


# --- 3. SORGULAMA FONKSİYONU ---
def simple_query_streamlit(prompt, text_chunks, embeddings, api_key):

    try:
        client = genai.Client(api_key=api_key)

        # 🔧 DÜZELTME: prompt embedding kısmında da aynı parametre düzeltildi
        prompt_response = client.models.embed_content(
            model='text-embedding-004',
            contents=[prompt]  # ✅ burada da 'contents' olmalı
        )

        prompt_embedding = (
            prompt_response.embeddings[0].values
            if hasattr(prompt_response, 'embeddings')
            else prompt_response.embedding
        )

        # --- En benzer 3 metin parçasını bul ---
        similarity_scores = []
        for i, chunk_embedding in enumerate(embeddings):
            score = cosine_similarity(prompt_embedding, chunk_embedding)
            similarity_scores.append((score, i))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [index for score, index in similarity_scores[:3]]
        retrieved_text = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])

    except Exception as e:
        st.warning(f"Vektör Arama Hatası: {e}. Basit aramaya geçiliyor.")
        retrieved_text = "\n\n---\n\n".join(text_chunks[:3])

    # --- RAG ÜRETİM AŞAMASI ---
    try:
        client = genai.Client(api_key=api_key)

        rag_prompt = (
            f"Sen, 'Enerji Verimliliği Eğitim Kitabı'na dayalı bir yapay zeka asistansın. "
            f"Görevin, aşağıdaki 'KAYNAK METİN'i kullanarak kullanıcı sorularını yanıtlamaktır.\n\n"
            f"TALİMATLAR:\n"
            f"1. **Öncelik ve Halüsinasyon Engeli:** Cevaplarını **KESİNLİKLE** sadece sağlanan KAYNAK METİN'deki bilgilerle sınırla. "
            f"Kaynak metnin dışındaki kendi genel bilgini **ASLA** kullanma.\n"
            f"2. **Akıl Yürütme:** Sorunun cevabı tek bir yerde geçmiyorsa, farklı bilgileri birleştirerek kapsamlı bir cevap oluştur.\n"
            f"3. **Reddetme:** Cevap kaynak metinde yoksa, "
            f"'Üzgünüm, bu sorunun cevabını elimdeki Enerji Verimliliği Eğitim Kitabında bulamıyorum.' de.\n\n"
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

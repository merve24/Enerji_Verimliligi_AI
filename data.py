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

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"  # Türkçe karakterleri sadeleştir

    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return [], None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None

    # 🔧 1️⃣ Chunk boyutunu büyüt
    chunk_size = 5000
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    st.info(f"{len(text_chunks)} metin parçası için embedding oluşturuluyor. Lütfen bekleyin...")

    embeddings = []
    try:
        client = genai.Client(api_key=api_key)
        progress_bar = st.progress(0, text="Embedding Oluşturuluyor...")

        # 🔧 2️⃣ Daha kaliteli embedding modeli
        for i, chunk in enumerate(text_chunks):
            response = client.models.embed_content(
                model='text-embedding-004',  # Daha güncel embedding modeli
                contents=[chunk]
            )
            # Embedding format kontrolü
            if hasattr(response, "embeddings"):
                embedding_vector = response.embeddings[0].values
            else:
                embedding_vector = response.embedding
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

        # 🔧 3️⃣ Prompt’u İngilizceye çevirerek embedding kalitesini artır
        prompt_for_embed = f"meaning of: {prompt} (in energy efficiency context)"

        prompt_response = client.models.embed_content(
            model='text-embedding-004',
            contents=[prompt_for_embed]
        )

        if hasattr(prompt_response, "embeddings"):
            prompt_embedding = prompt_response.embeddings[0].values
        else:
            prompt_embedding = prompt_response.embedding

        # 🔧 4️⃣ En benzer 5 metin parçasını al (önceden 3'tü)
        similarity_scores = []
        for i, chunk_embedding in enumerate(embeddings):
            score = cosine_similarity(prompt_embedding, chunk_embedding)
            similarity_scores.append((score, i))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [index for score, index in similarity_scores[:5]]
        retrieved_text = "\n\n---\n\n".join([text_chunks[i] for i in top_indices])

    except Exception as e:
        st.warning(f"Vektör Arama Hatası: {e}. Basit aramaya geçiliyor.")
        retrieved_text = "\n\n---\n\n".join(text_chunks[:3])

    # --- 5️⃣ MODEL CEVAP ÜRETİMİ ---
    try:
        client = genai.Client(api_key=api_key)

        # 🟢 GÜNCELLENMİŞ RAG PROMPT BAŞLANGICI 🟢
        rag_prompt = (
            f"Sen 'Enerji Verimliliği Eğitim Kitabı'na dayalı, uzman bir AI Chatbotsun. "
            f"Kullanıcıya enerji yönetimi, sürdürülebilirlik ve verimlilik konularında pratik, güvenilir ve sentezlenmiş bilgiler sun.\n\n"
            f"TALİMATLAR:\n"
            f"1. YALNIZCA aşağıdaki KAYNAK METİN içeriğine dayanarak cevap oluştur.\n"
            f"2. KAYNAK METİN'deki farklı bilgileri **mantıksal olarak sentezle, analiz et ve akıl yürüterek** kullanıcının sorusuna en kapsamlı cevabı ver.\n"
            f"3. Cevaplarını, kaynak metinlerden doğrudan alıntı yapmak yerine, akıcı ve profesyonel bir üslupla açıkla.\n"
            f"4. **KESİNLİKLE HALÜSİNASYON YAPMA.** Kaynakta olmayan (uydurma, yorum veya tahmine dayalı) hiçbir bilgiyi cevaplarına dahil etme.\n"
            f"5. Eğer sorunun cevabı KAYNAK METİN'de kesinlikle yoksa, cevap olarak şu ifadeyi kullan: "
            f"'Üzgünüm, bu sorunun cevabını elimdeki Enerji Verimliliği Eğitim Kitabı içeriğinde bulamıyorum.'\n\n"
            f"KAYNAK METİN:\n---\n{retrieved_text}\n---\n\n"
            f"KULLANICI SORUSU: {prompt}"
        )
        # 🟢 GÜNCELLENMİŞ RAG PROMPT SONU 🟢

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt
        )

        return response.text, retrieved_text

    except Exception as e:
        st.error(f"Sorgulama Hatası: Gemini API çağrısında bir sorun oluştu. Hata: {e}")
        return "Üzgünüm, sorgulama sırasında bir hata oluştu.", retrieved_text
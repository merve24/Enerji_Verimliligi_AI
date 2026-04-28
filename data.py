import os
import streamlit as st
import chromadb
import google.generativeai as genai

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"


# -----------------------------
# MODEL SEÇ
# -----------------------------
@st.cache_resource
def get_working_model(api_key):

    genai.configure(api_key=api_key)

    try:
        models = genai.list_models()
    except Exception:
        return None

    for m in models:
        if "generateContent" in m.supported_generation_methods:
            return m.name

    return None


# -----------------------------
# RAG VERİ HAZIRLAMA
# -----------------------------
@st.cache_resource
def prepare_rag_data(api_key):

    file_paths = [
        "Enerji_verimliligi_eğitim_kitabi.txt",
        "Enerji Verimliliği Mevzuatı.txt",
        "Binalarda enerji performansı yönetmeliği.txt"
    ]

    all_text = ""

    for path in file_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

    if not all_text.strip():
        return None

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=chromadb.config.Settings(
            anonymized_telemetry=False
        )
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # zaten varsa tekrar indexleme
    if collection.count() > 0:
        return collection

    # chunking
    chunks = [all_text[i:i+1200] for i in range(0, len(all_text), 1200)]

    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )

    return collection


# -----------------------------
# QUERY ENGINE
# -----------------------------
def simple_query_streamlit(prompt, collection, api_key):

    results = collection.query(
        query_texts=[prompt],
        n_results=5
    )

    context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    genai.configure(api_key=api_key)

    model_name = get_working_model(api_key)

    if model_name is None:
        return "Uygun model bulunamadı.", context

    model = genai.GenerativeModel(model_name)

    strict_prompt = f"""
Sen enerji verimliliği ve bina yönetmelikleri konusunda uzman bir asistansın.

KURALLAR:
- Sadece verilen kaynakları kullan
- Kaynaklar: eğitim kitabı + mevzuat + bina enerji performansı yönetmeliği
- Uydurma bilgi ekleme
- Eğer bilgi yoksa "kaynaklarda bulunamadı" de
- Mevzuat maddelerini özellikle belirt
- Açıklamayı rehber gibi yaz

KAYNAK:
{context}

SORU:
{prompt}
"""

    try:
        response = model.generate_content(strict_prompt)
        answer = response.text
    except Exception as e:
        answer = f"Hata oluştu: {e}"

    return answer, context
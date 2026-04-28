import os
import streamlit as st
import chromadb
import google.generativeai as genai

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"


# -----------------------------
# MODEL BUL
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
        "Enerji Verimliliği Mevzuatı.txt"
    ]

    # dosyaları birleştir
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

    # zaten index varsa tekrar yapma
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
# QUERY (GELİŞMİŞ RAG)
# -----------------------------
def simple_query_streamlit(prompt, collection, api_key):

    # context al
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
Sen enerji verimliliği ve mevzuat konusunda uzman bir asistansın.

KURALLAR:
- Sadece verilen KAYNAK metnini kullan
- Kaynak: eğitim kitabı + mevzuat olabilir
- Uydurma bilgi ekleme
- Mevzuat varsa özellikle belirt
- Emin değilsen "kaynakta yok" de
- Açıklamaları rehber gibi yaz

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
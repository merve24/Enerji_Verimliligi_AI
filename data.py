import os
import streamlit as st
import chromadb
import google.generativeai as genai

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"


# -----------------------------
# MODEL BULUCU (EN KRİTİK PARÇA)
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
            return m.name  # örn: models/gemini-1.5-flash

    return None


# -----------------------------
# RAG DATA
# -----------------------------
@st.cache_resource
def prepare_rag_data(api_key):

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
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

    if collection.count() > 0:
        return collection

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]

    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )

    return collection


# -----------------------------
# QUERY (HALÜSİNASYON KONTROLLÜ)
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
        return "Uygun model bulunamadı (API erişim sorunu).", context

    model = genai.GenerativeModel(model_name)

    strict_prompt = f"""
SEN SADECE VERİLEN METNE GÖRE CEVAP VEREN BİR ASİSTANSIN.

KURALLAR:
- SADECE aşağıdaki KAYNAK metni kullan
- Kendi bilginle ekleme yapma
- Tahmin yapma
- Eğer cevap yoksa: "Bu bilgi dokümanda bulunamadı." yaz

KAYNAK:
{context}

SORU:
{prompt}
"""

    try:
        response = model.generate_content(strict_prompt)
        answer = response.text
    except Exception as e:
        answer = f"Model hatası: {e}"

    return answer, context
import os
import streamlit as st
import chromadb
import requests
import google.generativeai as genai

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"


# -----------------------------
# MODEL BULUCU
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
# WEB SEARCH (basit ama iş görür)
# -----------------------------
def search_web(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        res = requests.get(url, timeout=5).json()

        text = ""
        if "AbstractText" in res:
            text += res["AbstractText"]

        if "RelatedTopics" in res:
            for t in res["RelatedTopics"][:3]:
                if isinstance(t, dict) and "Text" in t:
                    text += "\n" + t["Text"]

        return text[:2000]

    except Exception:
        return ""


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

    chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]

    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )

    return collection


# -----------------------------
# QUERY
# -----------------------------
def simple_query_streamlit(prompt, collection, api_key):

    # 1️⃣ TXT
    results = collection.query(
        query_texts=[prompt],
        n_results=5
    )

    local_context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    # 2️⃣ WEB fallback
    web_context = ""
    if len(local_context.strip()) < 200:
        web_context = search_web(prompt)

    # 3️⃣ birleşik context
    context = f"""
[TXT]
{local_context}

[WEB]
{web_context}
"""

    genai.configure(api_key=api_key)

    model_name = get_working_model(api_key)

    if model_name is None:
        return "Uygun model bulunamadı.", context

    model = genai.GenerativeModel(model_name)

    strict_prompt = f"""
Sen enerji verimliliği konusunda uzman bir asistansın.

KURALLAR:
- Önce TXT verisini kullan
- Yetersizse WEB bilgisini kullan
- Asla uydurma bilgi ekleme
- Emin değilsen belirt
- Cevabı öğretici, rehber gibi yaz

- Eğer bilgi yoksa:
  "Bu konuda yeterli bilgi bulunamadı." yaz

- Cevap sonunda kısa maddelerle özet ver

KAYNAK:
{context}

SORU:
{prompt}
"""

    try:
        response = model.generate_content(strict_prompt)
        answer = response.text
    except Exception as e:
        answer = f"Hata: {e}"

    return answer, context
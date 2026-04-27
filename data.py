import os
import streamlit as st
import chromadb

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"


# -----------------------------
# SIMPLE RAG (EMBEDDING YOK)
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

    # ❗ EMBEDDING YOK → direkt text index
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )

    return collection


# -----------------------------
# SIMPLE SEARCH (KEYWORD MATCH)
# -----------------------------
def simple_query_streamlit(prompt, collection, api_key):

    results = collection.query(
        query_texts=[prompt],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    import google.generativeai as genai

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        f"""
Aşağıdaki metinleri kullanarak cevap ver:

METİN:
{context}

SORU:
{prompt}
"""
    )

    return response.text, context
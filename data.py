import os
import streamlit as st
import chromadb
import requests

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"


# -----------------------------
# EMBEDDING (REST - STABIL)
# -----------------------------
def get_embedding(api_key, text):

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/text-embedding-004:embedContent"
    )

    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }

    r = requests.post(url, params={"key": api_key}, json=payload)

    if r.status_code != 200:
        raise Exception(f"Embedding API error: {r.text}")

    return r.json()["embedding"]["values"]


# -----------------------------
# RAG DATA PREP
# -----------------------------
@st.cache_resource
def prepare_rag_data(api_key):

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        return None

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # zaten varsa tekrar embed yapma
    if collection.count() > 0:
        return collection

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # chunking (stabil boyut)
    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]

    embeddings = []
    docs = []
    ids = []

    for i, chunk in enumerate(chunks):

        try:
            emb = get_embedding(api_key, chunk)

            embeddings.append(emb)
            docs.append(chunk)
            ids.append(str(i))

        except Exception as e:
            st.warning(f"Embedding atlandı: {e}")

    # ❗ güvenlik kontrolü
    if len(ids) == 0:
        raise ValueError("Embedding üretilemedi. API çalışmıyor.")

    collection.add(
        embeddings=embeddings,
        documents=docs,
        ids=ids
    )

    return collection


# -----------------------------
# QUERY
# -----------------------------
def simple_query_streamlit(prompt, collection, api_key):

    try:
        q_emb = get_embedding(api_key, prompt)

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3
        )

        context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    except Exception as e:
        context = ""

    # burada Gemini response (Streamlit uyumlu basit kullanım)
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        f"KAYNAK:\n{context}\n\nSORU:\n{prompt}"
    )

    return response.text, context
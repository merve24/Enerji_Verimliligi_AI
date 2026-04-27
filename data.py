import os
import streamlit as st
import chromadb
import google.generativeai as genai

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"

@st.cache_resource
def prepare_rag_data(api_key):

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    if not os.path.exists(file_path):
        return None

    # 🔥 eski ama stabil SDK
    genai.configure(api_key=api_key)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        return collection

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]

    embeddings = []
    docs = []
    ids = []

    for i, chunk in enumerate(chunks):

        try:
            res = genai.embed_content(
                model="models/embedding-001",
                content=chunk
            )

            embeddings.append(res["embedding"])
            docs.append(chunk)
            ids.append(str(i))

        except Exception as e:
            print("Embedding error:", e)

    if not ids:
        raise ValueError("Embedding üretilemedi - API uyumsuz")

    collection.add(
        embeddings=embeddings,
        documents=docs,
        ids=ids
    )

    return collection


def simple_query_streamlit(prompt, collection, api_key):

    genai.configure(api_key=api_key)

    res = genai.embed_content(
        model="models/embedding-001",
        content=prompt
    )

    q_emb = res["embedding"]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
        f"KAYNAK:\n{context}\n\nSORU:\n{prompt}"
    )

    return response.text, context
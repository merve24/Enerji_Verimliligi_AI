import os
import streamlit as st
from google import genai
import chromadb
from chromadb.config import Settings

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"

@st.cache_resource
def prepare_rag_data(api_key):

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    if not os.path.exists(file_path):
        return None

    client = genai.Client(api_key=api_key)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # zaten doluysa tekrar embedding yapma
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
            res = client.models.embed_content(
                model="models/text-embedding-004",  # 🔥 geri döndük (çünkü API bunu istiyor)
                contents=[chunk]
            )

            embeddings.append(res.embeddings[0].values)
            docs.append(chunk)
            ids.append(str(i))

        except Exception as e:
            print("Embedding failed:", e)

    # ❗ KRİTİK KORUMA
    if len(ids) == 0:
        raise ValueError("Hiç embedding oluşturulamadı. API modeli çalışmıyor.")

    collection.add(
        embeddings=embeddings,
        documents=docs,
        ids=ids
    )

    return collection


def simple_query_streamlit(prompt, collection, api_key):

    client = genai.Client(api_key=api_key)

    try:
        res = client.models.embed_content(
            model="models/text-embedding-004",
            contents=[prompt]
        )

        q_emb = res.embeddings[0].values

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3
        )

        context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    except Exception as e:
        context = ""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"KAYNAK:\n{context}\n\nSORU:\n{prompt}"
    )

    return response.text, context
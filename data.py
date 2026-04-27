import os
import streamlit as st
from google import genai
import chromadb
from chromadb.config import Settings

CHROMA_DB_PATH = "chroma_db_cache"
CHROMA_COLLECTION_NAME = "enerji_verimliligi_kitabi"

@st.cache_resource
def prepare_rag_data(api_key):
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        return None

    try:
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )
    except Exception as e:
        st.error(f"ChromaDB Hatası: {e}")
        return None

    if collection.count() == 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunk_size = 2000
        text_chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

        client = genai.Client(api_key=api_key)

        embeddings, ids, docs = [], [], []

        for i, chunk in enumerate(text_chunks):
            res = client.models.embed_content(
                model='text-embedding-004',  # ✅ düzeltildi
                contents=[chunk]
            )

            embeddings.append(res.embeddings[0].values)
            docs.append(chunk)
            ids.append(f"id_{i}")

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
            model='text-embedding-004',  # ✅ düzeltildi
            contents=[prompt]
        )

        query_vector = res.embeddings[0].values

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )

        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""

    except Exception:
        context = ""

    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',  # ✅ daha stabil
            contents=f"KAYNAK:\n{context}\n\nSORU: {prompt}"
        )

        return response.text, context

    except Exception as e:
        return f"Hata: {e}", context
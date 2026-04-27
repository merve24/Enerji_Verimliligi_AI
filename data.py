import os
import streamlit as st
from google import genai
import chromadb
from chromadb.config import Settings

CHROMA_DB_PATH = "chroma_db_cache"
COLLECTION_NAME = "enerji_verimliligi"

# ------------------------
# RAG DATA PREP
# ------------------------
@st.cache_resource
def prepare_rag_data(api_key):

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        return None

    try:
        client_db = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        collection = client_db.get_or_create_collection(name=COLLECTION_NAME)

    except Exception as e:
        st.error(f"ChromaDB hata: {e}")
        return None

    # sadece ilk kurulumda embedding yap
    if collection.count() == 0:

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = [text[i:i+1800] for i in range(0, len(text), 1800)]

        client = genai.Client(api_key=api_key)

        embeddings = []
        docs = []
        ids = []

        for i, chunk in enumerate(chunks):
            try:
                res = client.models.embed_content(
                    model="embedding-001",   # 🔥 STABİL MODEL
                    contents=[chunk]
                )

                embeddings.append(res.embeddings[0].values)
                docs.append(chunk)
                ids.append(str(i))

            except Exception as e:
                st.warning(f"Embedding atlandı: {e}")

        collection.add(
            embeddings=embeddings,
            documents=docs,
            ids=ids
        )

    return collection


# ------------------------
# QUERY
# ------------------------
def simple_query_streamlit(prompt, collection, api_key):

    client = genai.Client(api_key=api_key)

    try:
        res = client.models.embed_content(
            model="embedding-001",   # 🔥 aynı model
            contents=[prompt]
        )

        q_emb = res.embeddings[0].values

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3
        )

        context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    except Exception:
        context = ""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"""
Aşağıdaki bağlamı kullanarak cevap ver.

BAĞLAM:
{context}

SORU:
{prompt}
"""
        )

        return response.text, context

    except Exception as e:
        return f"Hata: {e}", context
import os
import streamlit as st
from google import genai
import chromadb
from chromadb.config import Settings

# --- SABİTLER ---
CHROMA_DB_PATH = "chroma_db_cache"
CHROMA_COLLECTION_NAME = "enerji_verimliligi_kitabi"

@st.cache_resource
def prepare_rag_data(api_key):
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        return None

    try:
        # Telemetriyi kapatmak sürüm hatalarını minimize eder
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        st.error(f"Veritabanı Hatası: {e}")
        return None

    if collection.count() == 0:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except: return None

        # Metni parçalara ayır
        chunk_size = 2000
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        client = genai.Client(api_key=api_key)
        
        embeddings, ids, docs = [], [], []
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(text_chunks):
            # MODEL YOLU DÜZELTİLDİ: 'models/' eklendi
            res = client.models.embed_content(
                model='models/text-embedding-004',
                contents=[chunk]
            )
            embeddings.append(res.embeddings[0].values)
            docs.append(chunk)
            ids.append(f"id_{i}")
            progress_bar.progress((i + 1) / len(text_chunks))
        
        progress_bar.empty()
        collection.add(embeddings=embeddings, documents=docs, ids=ids)
    
    return collection

def simple_query_streamlit(prompt, collection, api_key):
    client = genai.Client(api_key=api_key)
    
    try:
        # Soru için embedding oluştur
        res = client.models.embed_content(
            model='models/text-embedding-004',
            contents=[prompt]
        )
        query_vector = res.embeddings[0].values
        
        # En yakın 3 parçayı getir
        results = collection.query(query_embeddings=[query_vector], n_results=3)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
    except:
        context = ""

    # Yanıt oluşturma
    rag_prompt = f"KAYNAK METİN:\n{context}\n\nSORU: {prompt}\nLütfen yukarıdaki bilgilere göre yanıt ver."
    
    try:
        # Gemini 2.0 Flash kullanıyoruz
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=rag_prompt
        )
        return response.text, context
    except Exception as e:
        return f"Cevap üretilemedi: {e}", context
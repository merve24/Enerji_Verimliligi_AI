import os
import streamlit as st
from google import genai
import chromadb # Yeni import
from chromadb.utils import embedding_functions 

# --- 2. RAG VERİ HAZIRLAMA (Chroma Versiyonu) ---
CHROMA_DB_PATH = "chroma_db_cache"
CHROMA_COLLECTION_NAME = "enerji_verimliligi_kitabi"

@st.cache_resource
def prepare_rag_data(api_key):
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return None # Hata durumunda None döndür

    try:
        # 1. Chroma İstemcisi ve Koleksiyonu Oluştur
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    except Exception as e:
        st.error(f"ChromaDB Başlatma Hatası: {e}")
        return None

    # Eğer koleksiyon boşsa, veriyi oku, parçala ve embedding oluşturup kaydet.
    if collection.count() == 0:
        print(">> [RAG LOG] Veritabanı boş. Embedding'ler oluşturuluyor ve diske kaydediliyor...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
            return None

        chunk_size = 2000
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        print(f">> [RAG LOG] {len(text_chunks)} metin parçası için embedding oluşturuluyor (arka planda).")

        embeddings_to_add = []
        ids_to_add = []
        documents_to_add = []

        try:
            client = genai.Client(api_key=api_key)
            progress_bar = st.progress(0)

            for i, chunk in enumerate(text_chunks):
                # 3. Embedding Oluştur
                response = client.models.embed_content(
                    model='text-embedding-004',
                    contents=[chunk]
                )
                
                if hasattr(response, "embeddings"):
                    embedding_vector = response.embeddings[0].values
                else:
                    embedding_vector = response.embedding

                embeddings_to_add.append(embedding_vector)
                documents_to_add.append(chunk)
                ids_to_add.append(f"doc_{i}")
                
                # İlerleme çubuğunu göster
                progress_bar.progress((i + 1) / len(text_chunks))

            progress_bar.empty() # İş bitince ilerleme çubuğunu kaldır
            
            # 4. Toplu Kayıt İşlemi
            collection.add(
                embeddings=embeddings_to_add,
                documents=documents_to_add,
                ids=ids_to_add
            )
            print(f">> [RAG LOG] {len(documents_to_add)} parça ChromaDB'ye kaydedildi ve kullanıma hazır.")
            
        except Exception as e:
            st.error(f"Embedding/Chroma Kayıt Hatası: {e}")
            return None
    
    else:
        print(f">> [RAG LOG] ChromaDB'den {collection.count()} parça yüklendi (Önbellek kullanıldı).")
        
    return collection # Başarılıysa koleksiyonu döndür

# --- 3. SORGULAMA FONKSİYONU (Chroma Versiyonu) ---
def simple_query_streamlit(
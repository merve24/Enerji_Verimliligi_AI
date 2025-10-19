import os 
import streamlit as st
from google import genai 
import math # Cosine Similarity için gerekli

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA ---
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    pass

# --- 2. ANA FONKSİYON: VERİSİNİ HAZIRLAMA (Embedding ve Index Oluşturma) ---
@st.cache_resource
def prepare_rag_data(api_key):
    # Bu fonksiyon şimdi hem parçaları hem de onların embedding'lerini döndürecek.
    
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return [], None 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None
    
    # Metin Parçalama
    text_chunks = []
    chunk_size = 2000 
    for i in range(0, len(text), chunk_size):
        text_chunks.append(text[i:i + chunk_size])
    
    # --- KRİTİK: Embedding ve Index Oluşturma ---
    st.info(f"{len(text_chunks)} metin parçası için embedding oluşturuluyor. Lütfen bekleyin...")
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Tüm parçaların embedding'lerini tek bir çağrıda alın (Toplu İşlem)
        response = client.models.batch_embed_content(
            model='text-embedding-004', 
            contents=text_chunks
        )
        
        embeddings = [item.embedding for item in response.embeddings]
        
        return text_chunks, embeddings 
        
    except Exception as e:
        st.error(f"Embedding Hatası: Veri gömme işlemi başarısız oldu. API anahtarınızın doğru olduğundan ve kota limitlerinin aşılmadığından emin olun. Hata: {e}")
        return [], None

# --- 3.
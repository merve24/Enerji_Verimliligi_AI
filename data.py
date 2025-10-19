import os 
import faiss
import numpy as np
import streamlit as st
from google import genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.errors import APIError

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA (simple_chunking) ---
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    text_chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            if current_chunk:
                text_chunks.append(current_chunk)
            current_chunk = paragraph
    if current_chunk:
        text_chunks.append(current_chunk)
    return text_chunks

# --- 2. ANA FONKSİYON: RAG VERİSİNİ HAZIRLAMA VE İNDEKSLENDİRME ---

# Streamlit @st.cache_resource kullanarak indeks yüklemesini yalnızca bir kez zorlayın.
@st.cache_resource(show_spinner="RAG Verileri Yükleniyor/Oluşturuluyor...")
def prepare_rag_data(api_key):
    
    def create_empty_rag(dimension=1536):
        empty_index = faiss.IndexFlatL2(dimension)
        return [], empty_index

    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    faiss_index_path = "faiss_index.bin"
    chunks_path = "text_chunks.npy"
    
    # 1. Dosya Var Mı Kontrolü
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı. Lütfen GitHub'a yükleyin.")
        return create_empty_rag()

    # 2. Önbellek Kontrolü (Yükleme)
    if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
        try:
            index = faiss.read_index(faiss_index_path)
            # allow_pickle=True, numpy dizisi içinde Python nesnelerini (metin) saklamak için gereklidir
            text_chunks = np.load(chunks_path, allow_pickle=True).tolist() 
            st.success("Önceden oluşturulmuş FAISS indexi ve metin parçaları başarıyla yüklendi.")
            return text_chunks, index
        except Exception as e:
            # Yükleme hatası olursa, zaman aşımına neden olmamak için boş döndürülür.
            st.error(f"KRİTİK HATA: FAISS indexi veya metin parçaları yüklenemedi. Detay: {e}")
            return create_empty_rag()

    # 3. Vektörleştirme (Önbellek Dosyaları Yoksa)
    st.warning("DİKKAT: FAISS indeksi API ile YENİDEN OLUŞTURULUYOR. Bu işlem ZAMAN AŞIMI riski taşır. Oluşturulduğunda index dosyalarını GitHub'a YÜKLEYİN.")
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return create_empty_rag()
    
    text_chunks = simple_chunking(text)
    
    client = genai.Client(api_key=api_key)
    embeddings_list = []
    
    try:
        with st.status("Metin parçaları vektörleştiriliyor...", expanded=True) as status:
            for i, chunk in enumerate(text_chunks):
                status.update(label=f"Metin parçası {i+1}/{len(text_chunks)} işleniyor...")
                response = client.models.embed_content(
                    model='text-embedding-004',
                    contents=chunk,
                )
                embeddings_list.append(response['embedding'])
            status.update(label="Vektörleştirme tamamlandı.", state="complete")

    except APIError as e:
        st.error(f"KRİTİK HATA: Gemini API'ye bağlanırken sorun oluştu. Detay: {e}")
        return create_empty_rag() 
    except (AttributeError, KeyError, google_exceptions.GoogleAPICallError) as e:
        st.error(f"Vektörleştirme sırasında beklenmeyen bir hata oluştu: {e}")
        return create_empty_rag()
    
    if not embeddings_list:
        st.error("HATA: Vektör dizisi boş. API'den veri alınamadı.")
        return create_empty_rag()
        
    # 4. İndeks Oluşturma ve Kaydetme Adımı
    embeddings_array = np.array(embeddings_list, dtype='float32')
    dimension = embeddings_array.shape[1] 

    index = faiss.IndexFlatL2(dimension)
    try:
        index.add(embeddings_array)

        faiss.write_index(index, faiss_index_path)
        np.save(chunks_path, np.array(text_chunks, dtype=object))
        st.success("FAISS indexi ve metin parçaları başarılı bir şekilde oluşturuldu ve kaydedildi. Lütfen bu iki dosyayı GitHub'a yükleyin.")
    except Exception as e:
        st.error(f"İndeks oluşturma veya kaydetme sırasında KRİTİK HATA oluştu: {e}. Muhtemelen bellek yetersizliği.")
        return create_empty_rag()

    return text_chunks, index
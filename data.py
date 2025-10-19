import os 
import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai.errors import APIError 

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

def prepare_rag_data(api_key):
    # Hata durumunda güvenli dönüş için boş bir indeks oluşturma
    def create_empty_rag():
        empty_index = faiss.IndexFlatL2(1536) 
        return [], empty_index

    # 1. Metin Dosyasını Kontrol Etme
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası bulunamadı: {file_path}")
        return create_empty_rag()

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Kalıcı İndeks Yolu Tanımlama
    faiss_index_path = "faiss_index.bin"
    chunks_path = "text_chunks.npy"
    
    # 3. İndeksi Diskten Yükleme Kontrolü (HIZLI YOL)
    if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
        try:
            index = faiss.read_index(faiss_index_path)
            text_chunks = np.load(chunks_path, allow_pickle=True).tolist()
            st.success("FAISS indexi diskten başarıyla yüklendi (Hızlı Başlangıç).")
            return text_chunks, index # Hızlı ve başarılı dönüş
        except Exception:
            st.warning("Kayıtlı index yüklenirken hata oluştu. Yeniden oluşturuluyor...")
            pass # Hata durumunda yeniden oluşturma sürecine devam et

    # 4. İndeksi Yeniden Oluşturma (YAVAŞ YOL)

    text_chunks = simple_chunking(text)
    embeddings_list = []
    
    try:
        client = genai.Client(api_key=api_key)
        st.info("FAISS indeksi API ile yeniden oluşturuluyor. Bu işlem ZAMAN AŞIMI riski taşıdığından, tamamlandığında oluşan index dosyalarını GitHub'a yükleyin.")
        
        # Vektörleştirme döngüsü
        for chunk in text_chunks:
            # KRİTİK DÜZELTME: 'task_type' parametresi kaldırıldı.
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=chunk,  
            )
            embeddings_list.append(response['embedding'])
        
    except APIError as e:
        st.error(f"KRİTİK HATA: Gemini API'ye bağlanırken sorun oluştu. Anahtarınızı kontrol edin. Detay: {e}")
        return create_empty_rag() # Hata durumunda güvenli dönüş
    except Exception as e:
        st.error(f"Vektörleştirme sırasında beklenmeyen bir hata oluştu: {e}")
        return create_empty_rag() # Hata durumunda güvenli dönüş
    
    # 5. Güvenlik Kontrolü ve FAISS Index Oluşturma
    if not embeddings_list:
        st.error("HATA: Vektör dizisi boş. API'den veri alınamadı.")
        return create_empty_rag()
        
    embeddings_array = np.array(embeddings_list, dtype='float32')
    dimension = embeddings_array.shape[1] 

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # 6. Başarılı Oluşturmanın Ardından İndeksi Kaydetme
    try:
        faiss.write_index(index, faiss_index_path)
        np.save(chunks_path, text_chunks)
        st.success("FAISS indexi ve metin parçaları diske kaydedildi.")
    except Exception as e:
        st.warning(f"Oluşturulan index diske kaydedilemedi: {e}. Lütfen indexi kendiniz oluşturup GitHub'a yükleyin.")

    return text_chunks, index # Başarılı dönüş
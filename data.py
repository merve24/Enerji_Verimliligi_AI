import os 
import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai.errors import APIError # API hatalarını yakalamak için

# --- Diğer yardımcı fonksiyonlar (simple_chunking, vb.) buraya gelir ---

def prepare_rag_data(api_key):
    # 1. Metin Dosyasını Kontrol Etme
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası bulunamadı: {file_path}")
        return [], None # Metin parçaları ve index boş döner

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Kalıcı İndeks Yolu Tanımlama (GitHub'a Yüklediğiniz dosya adları)
    faiss_index_path = "faiss_index.bin"
    chunks_path = "text_chunks.npy"

    # 3. İndeksi Diskten Yükleme Kontrolü (HIZLI YOL)
    if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
        # Index diskten yüklenir. YAVAŞ VE API ÇAĞRISI GEREKMEZ!
        try:
            index = faiss.read_index(faiss_index_path)
            text_chunks = np.load(chunks_path, allow_pickle=True).tolist()
            st.success("FAISS indexi diskten başarıyla yüklendi (Hızlı Başlangıç).")
            return text_chunks, index
        except Exception as e:
            st.warning(f"Kayıtlı index yüklenirken hata: {e}. Yeniden oluşturuluyor...")
            pass # Yükleme başarısız olursa yeniden oluşturma sürecine devam et

    # 4. İndeksi Yeniden Oluşturma (YAVAŞ YOL - API Çağrısı GEREKLİ)

    text_chunks = simple_chunking(text)
    embeddings_list = []
    
    try:
        client = genai.Client(api_key=api_key)
        
        # st.info("FAISS indeksi API ile yeniden oluşturuluyor. Lütfen bekleyin...")
        
        for i, chunk in enumerate(text_chunks):
            # API çağrısı: Vektörleştirme
            response = client.models.embed_content(
                model='text-embedding-004',
                content=chunk,
                task_type='RETRIEVAL_DOCUMENT'
            )
            embeddings_list.append(response['embedding'])
        
    except APIError as e:
        # API anahtarı geçersizse veya kota aşılırsa burası çalışır.
        st.error(f"API HATA: Gemini API'ye bağlanırken sorun oluştu. Anahtarınızı kontrol edin.")
        # Hata durumunda boş döner ve uygulamanın çökmesini engeller.
        return [], None
    except Exception as e:
        st.error(f"Vektörleştirme sırasında beklenmeyen bir hata oluştu: {e}")
        return [], None
    
    # --- BURASI CRITICAL! (NameError: dimension Çözümü) ---
    
    # 5. Güvenlik Kontrolü ve FAISS Index Oluşturma
    
    if not embeddings_list:
        st.error("Vektör dizisi boş. Veri okunamadı veya API'den veri alınamadı.")
        return [], None
        
    embeddings_array = np.array(embeddings_list, dtype='float32')
    
    # Güvenlik Kontrolünden sonra dimension tanımlanır
    dimension = embeddings_array.shape[1] 

    # FAISS Index Oluşturma
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # 6. Başarılı Oluşturmanın Ardından İndeksi Kaydetme
    try:
        faiss.write_index(index, faiss_index_path)
        np.save(chunks_path, text_chunks)
        st.success("FAISS indexi ve metin parçaları başarıyla diske kaydedildi.")
    except Exception as e:
        st.warning(f"Oluşturulan index diske kaydedilemedi: {e}")

    return text_chunks, index
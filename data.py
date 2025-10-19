import os 
import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai.errors import APIError 

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA (simple_chunking) ---
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    """
    Basit metin parçalama (chunking) fonksiyonu.
    Metni paragraf bazında parçalamaya çalışır ve anlam bütünlüğünü korur.
    """
    text_chunks = []
    
    # Paragraflara ayırma
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    current_chunk = ""
    for paragraph in paragraphs:
        # Chunk boyutu kontrolü
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            # Sınır aşıldı, mevcut parçayı kaydet
            if current_chunk:
                text_chunks.append(current_chunk)
            
            # Yeni parçayı başlat
            current_chunk = paragraph

    # Kalan son parçayı ekle
    if current_chunk:
        text_chunks.append(current_chunk)

    return text_chunks


# --- 2. ANA FONKSİYON: RAG VERİSİNİ HAZIRLAMA VE İNDEKSLENDİRME ---

def prepare_rag_data(api_key):
    def create_empty_rag():
        # FAISS'in boş bir indeksle başlatılması (1536 boyutu sabit)
        empty_index = faiss.IndexFlatL2(1536) 
        return [], empty_index

    # 1. Metin Dosyasını Kontrol Etme
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı. Lütfen GitHub'a yükleyin.")
        return create_empty_rag()

    # 2. İndeks ve Parçaları Kontrol Etme (Önceden oluşturulmuş mu?)
    faiss_index_path = "faiss_index.bin"
    chunks_path = "text_chunks.npy"
    
    if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
        st.success("Önceden oluşturulmuş FAISS indexi ve metin parçaları başarıyla yüklendi.")
        try:
            index = faiss.read_index(faiss_index_path)
            text_chunks = np.load(chunks_path, allow_pickle=True).tolist()
            return text_chunks, index
        except Exception as e:
            st.warning(f"FAISS indexi veya metin parçaları yüklenemedi, yeniden oluşturuluyor. Detay: {e}")

    # 3. İndeks Dosyaları Yoksa veya Hatalıysa: Dosyadan Metin Okuma
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return create_empty_rag()
    
    # Metni parçalara ayırma
    text_chunks = simple_chunking(text)
    
    # 4. Vektörleştirme (Embedding) ve İndeks Oluşturma
    client = genai.Client(api_key=api_key)
    embeddings_list = []
    
    st.info("FAISS indeksi API ile yeniden oluşturuluyor. Bu işlem ZAMAN AŞIMI riski taşıdığından, tamamlandığında oluşan index dosyalarını GitHub'a yükleyin.")
        
    try:
        for chunk in text_chunks:
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=chunk,  
            )
            # 💥 DÜZELTME: API'den gelen yanıttan vektör verisini '.values' özelliği ile çekiyoruz.
            # Bu, 'app.py' ile tutarlılık sağlar ve potansiyel 'embedding' özniteliği hatalarını çözer.
            embeddings_list.append(response.values)
        
    except APIError as e:
        st.error(f"KRİTİK HATA: Gemini API'ye bağlanırken sorun oluştu. Anahtarınızı kontrol edin. Detay: {e}")
        return create_empty_rag() 
    except Exception as e:
        st.error(f"Vektörleştirme sırasında beklenmeyen bir hata oluştu: {e}")
        return create_empty_rag()
    
    # 5. Güvenlik Kontrolü ve FAISS Index Oluşturma
    if not embeddings_list:
        st.error("HATA: Vektör dizisi boş. API'den veri alınamadı.")
        return create_empty_rag()
        
    # Not: embeddings_list artık 1536 boyutlu vektörlerin bir listesini tutar.
    embeddings_array = np.array(embeddings_list, dtype='float32')
    dimension = embeddings_array.shape[1] 

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # 6. Başarılı Oluşturmanın Ardından İndeksi Kaydetme
    try:
        faiss.write_index(index, faiss_index_path)
        np.save(chunks_path, text_chunks)
        st.success("FAISS indexi ve metin parçaları başarılı bir şekilde oluşturuldu ve kaydedildi.")
    except Exception as e:
        st.warning(f"Index kaydetme hatası: {e}. Uygulama çalışmaya devam ediyor.")

    return text_chunks, index

import os 
import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai.errors import APIError 

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA (simple_chunking) ---
# NameError hatasını çözmek için bu fonksiyon tanımı CRITICAL'dir.
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    """
    Basit metin parçalama (chunking) fonksiyonu.
    Metni paragraf bazında parçalamaya çalışır ve anlam bütünlüğünü korur.
    """
    text_chunks = []
    
    # Text'i \n\n veya \r\n\r\n gibi paragraflara göre ayır
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    current_chunk = ""
    for paragraph in paragraphs:
        # Mevcut parça ve yeni paragrafın toplam boyutu sınırı aşmıyorsa ekle
        # +2, eklenen boşluk/satır sonu karakterleri için
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            # Sınır aşıldı, mevcut parçayı kaydet
            if current_chunk:
                text_chunks.append(current_chunk)
            
            # Yeni paragrafı yeni bir parça olarak başlat
            current_chunk = paragraph

    # Kalan son parçayı ekle
    if current_chunk:
        text_chunks.append(current_chunk)

    return text_chunks


# --- 2. ANA FONKSİYON: RAG VERİSİNİ HAZIRLAMA VE İNDEKSLENDİRME ---

def prepare_rag_data(api_key):
    # 1. Metin Dosyasını Kontrol Etme
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası bulunamadı: {file_path}")
        return [], None

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Kalıcı İndeks Yolu Tanımlama (GitHub'a yüklenen dosyalar)
    faiss_index_path = "faiss_index.bin"
    chunks_path = "text_chunks.npy"

    # 3. İndeksi Diskten Yükleme Kontrolü (HIZLI YOL - Yalnızca bir kez oluşturulduysa çalışır)
    if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
        try:
            index = faiss.read_index(faiss_index_path)
            # Metin parçalarını yükle
            text_chunks = np.load(chunks_path, allow_pickle=True).tolist()
            st.success("FAISS indexi diskten başarıyla yüklendi (Hızlı Başlangıç).")
            return text_chunks, index
        except Exception:
            st.warning("Kayıtlı index yüklenirken hata oluştu. Yeniden oluşturuluyor...")
            pass # Hata durumunda yeniden oluşturma sürecine devam et

    # 4. İndeksi Yeniden Olu
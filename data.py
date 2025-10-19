import os 
import streamlit as st
from google import genai
# Gerekli olmayan importlar kaldırıldı: faiss, numpy, google_exceptions, APIError

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA (simple_chunking) ---
# Bu fonksiyon artık kullanılmayacak.
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    pass

# --- 2. ANA FONKSİYON: VERİSİNİ HAZIRLAMA (FAISS YOK) ---
# @st.cache_resource kaldırıldı
def prepare_rag_data(api_key):
    
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    
    # Hata durumunda uygulama durur
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı. Lütfen GitHub'a yükleyin.")
        # app.py'nin kontrol etmesi için boş liste ve None döndürülür.
        return [], None 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None
    
    # KRİTİK DÜZELTME: Sadece kitabın İLK BÖLÜMÜNÜ (yaklaşık ilk 1000 karakteri) alıyoruz.
    # Bu, token limitini aşmayı önler.
    text_chunks = [text[:1000]] 
    
    # FAISS indeksi artık kullanılmadığı için index=None olarak döndürülür.
    return text_chunks, None
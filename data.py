import os 
import streamlit as st
# API key, model ve metin parçalama için gerekli importlar
from google import genai 

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA (simple_chunking) ---
# Bu fonksiyon artık kullanılmayacak, ana fonksiyonda manuel chunking yapıldı.
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    pass

# --- 2. ANA FONKSİYON: VERİSİNİ HAZIRLAMA ---
def prepare_rag_data(api_key):
    
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    
    # 1. HATA ÇÖZÜMÜ: Dosya Konumu Kontrolü (Aynı Dizinde Olmalı)
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı. Lütfen '{file_path}' dosyasının Python kodlarınızla (app.py, data.py) aynı dizinde olduğundan emin olun.")
        return [], None 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None
    
    # --- KRİTİK DÜZELTME: METİN PARÇALAMA (CHUNKING) EKLENDİ ---
    
    text_chunks = []
    # Metin parçalarının boyutunu belirleyin (Enerji Kitabı gibi dokümanlar için 2000 karakter iyi bir başlangıçtır.)
    chunk_size = 2000 
    
    # Kitabın tamamını (ilk karakter sınırı olmadan) parçalara ayırın
    for i in range(0, len(text), chunk_size):
        # Basit bir döngü ile çakışma (overlap) olmadan bölme işlemi
        text_chunks.append(text[i:i + chunk_size])
    
    # Önceki versiyondaki hatalı (1000 karakterle sınırlı) kod satırı KALDIRILMIŞTIR.
    
    # --- DÜZELTME SONU ---

    # Parçalanmış metin listesini döndürüyoruz.
    return text_chunks, None 

# --- 3. YARDIMCI FONKSİYON: SORGULAMA ---
# (Bu fonksiyon, app.py içinde basitçe kullanıldığı için burada değişiklik yapılmadı)
def simple_query_streamlit(prompt, text_chunks, api_key):
    # ... (Geri kalan kodunuzu koruyun, bu kısımda mantıksal bir hata gözlemlenmemiştir.)
    # Yalnızca 'text_chunks' değişkenini fonksiyona parametre olarak almasını sağlayın.
    # Bu kısmı app.py'de güncelleyeceğiz.
    
    # ... (Kodunuzun geri kalanını ekleyiniz.)
    pass
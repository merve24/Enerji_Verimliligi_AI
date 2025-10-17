import os 
import faiss
import numpy as np
import time
from google import genai

# --- Veri Hazırlama ve RAG İndeksini Oluşturma Fonksiyonu ---
# Bu fonksiyon Streamlit'te sadece bir kez çalıştırılacaktır (@st.cache_resource sayesinde).
def prepare_rag_data(api_key):
    
    # Dosya yolu, Streamlit/GitHub ortamında doğru yolu bulmak için ayarlanmıştır.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 💥 EN SON VE DOĞRU VERİ DOSYASI ADI
    file_path = os.path.join(base_dir, "Enerji_verimliligi_eğitim_kitabi.txt")
    
    # 1. Dosyayı Okuma
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
    except FileNotFoundError:
        print(f"HATA: Veri dosyası bulunamadı: {file_path}")
        # Hata durumunda boş bir çıktı döndürerek uygulamanın çökmesini engelleriz.
        return [], None
    
    # 2. Metin Parçalama (Chunking) İşlemi
    def simple_chunking(text, chunk_size=4000): # Bellek optimizasyonu için 4000 karakter
        chunks = []
        current_chunk = ""
        for line in text.split('\n'):
            if len(current_chunk) + len(line) + 1 < chunk_size:
                current_chunk += line + '\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # Parçalama işlemini başlat
    text_chunks = simple_chunking(text_content, chunk_size=4000) 
    
    # 3. Vektörleştirme ve Veritabanı Oluşturma
    client = genai.Client(api_key=api_key) 
    embeddings_list = []

    print(f"{len(text_chunks)} adet metin parçası vektörleştiriliyor...")
    
    # Vektörleştirme API limitlerine takılmamak için dikkatli döngü
    for i, chunk in enumerate(text_chunks):
        try:
            # text-embedding-004 modelini kullanarak vektör oluşturma
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=[chunk] 
            )
            embeddings_list.append(response.embeddings[0].values)
            
            # API kısıtlamalarını yönetmek için kısa bekleme
            if (i + 1) % 10 == 0:
                 time.sleep(1) 
        
        except Exception as e:
            print(f"Vektörleştirme sırasında hata oluştu ve durduruldu: {e}")
            break 

    # Vektörleri NumPy dizisine dönüştürme
    embeddings_array = np.array(embeddings_list, dtype='float32')
    dimension = embeddings_array.shape[1] 

    # FAISS Index'ini oluşturma ve vektörleri ekleme (L2 mesafesi)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    print("RAG Veri Hazırlığı Tamamlandı!")
    return text_chunks, index

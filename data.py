import os # YENİ KÜTÜPHANE İÇE AKTARILDI
import faiss
import numpy as np
import time
from google import genai

# --- Veri Hazırlama ve RAG İndeksini Oluşturma Fonksiyonu ---
def prepare_rag_data(api_key):
    
    # 💥 DÜZELTME: Dosya yolu güvenliği artırıldı.
    # __file__, içinde bulunduğu dosyanın yolunu verir. Streamlit root'u buluruz.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "Enerji_verimliligi_eğitim_kitabi-1-200.txt")
    
    # 1. Dosyayı Okuma
    # try/except ile hata yakalamayı ekleyelim.
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı: {file_path}")
        # Hata durumunda boş bir çıktı döndürerek uygulamanın çökmesini engelleriz.
        return [], None
    
    # 2. Metin Parçalama (Chunking) İşlemi (Aynı Kod)
    def simple_chunking(text, chunk_size=2000):
        chunks = []
        current_chunk = ""
        for line in text.split('\n'):
            if len(current_chunk) + len(line) < chunk_size:
                current_chunk += line + '\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    text_chunks = simple_chunking(text_content, chunk_size=2000)
    
    # 3. Vektörleştirme ve Veritabanı Oluşturma (Aynı Kod)
    client = genai.Client(api_key=api_key) 
    embeddings_list = []

    print("Metin parçaları vektörleştiriliyor...")
    
    for i, chunk in enumerate(text_chunks):
        # API Kota aşımı veya diğer hataları yakalamak için
        try:
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=[chunk] 
            )
            embeddings_list.append(response.embeddings[0].values)
            
            if (i + 1) % 10 == 0:
                 print(f"{i + 1} parça işlendi. Bekleniyor...")
                 time.sleep(1) 

        except Exception as e:
            print(f"Vektörleştirme sırasında hata oluştu: {e}")
            break 

    embeddings_array = np.array(embeddings_list, dtype='float32')
    dimension = embeddings_array.shape[1] 

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    print("RAG Veri Hazırlığı Tamamlandı!")
    return text_chunks, index

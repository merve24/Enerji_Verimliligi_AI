import os 
import faiss
import numpy as np
import time
from google import genai
# ... (diğer importlar)

def prepare_rag_data(api_key):
    # ... (file_path tanımlama ve Dosyayı Okuma adımı aynı kalır)

    # Yeni: FAISS ve Chunks dosyalarının yolunu tanımlama
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(base_dir, "faiss_index.bin") # Index dosyası
    chunks_path = os.path.join(base_dir, "text_chunks.npy")    # Chunks dosyası (Vektörlere karşılık gelen metinler)

    # 💥 KONTROL: Eğer FAISS indexi zaten varsa, doğrudan yükle
    if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
        print("FAISS indexi ve metin parçaları diskten yükleniyor...")
        try:
            # 1. FAISS Indexini Yükle
            index = faiss.read_index(faiss_index_path)
            # 2. Metin Parçalarını Yükle
            text_chunks = np.load(chunks_path, allow_pickle=True).tolist()
            return text_chunks, index
        except Exception as e:
            print(f"HATA: Kayıtlı index yüklenirken hata oluştu: {e}")
            # Yükleme başarısız olursa normal sürece devam et.
            pass

    # --- Eğer index yoksa, normal vektörleştirme ve kaydetme sürecini başlat ---

    # 2. Metin Parçalama (Chunking) İşlemi (aynı kalır)
    # ... (simple_chunking fonksiyonu ve text_chunks oluşturma burada yer alır)
    
    # 3. Vektörleştirme ve Veritabanı Oluşturma (Bu kısım YAVAŞ olan kısımdır, SADECE BİR KEZ çalışacak)
    # ... (client oluşturma, API ile vektörleştirme ve embeddings_array oluşturma burada yer alır)

    # --- YENİ: Başarılı Vektörleştirmeden Sonra İndeksi Kaydetme ---
    
    # FAISS Index'ini oluşturma (önceki gibi)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # 💥 Kaydetme işlemi
    try:
        faiss.write_index(index, faiss_index_path)
        np.save(chunks_path, text_chunks)
        print("FAISS indexi ve metin parçaları başarıyla diske kaydedildi.")
    except Exception as e:
        print(f"HATA: Index kaydedilirken sorun yaşandı: {e}")


    return text_chunks, index

# prepare_rag_data fonksiyonunun geri kalanı
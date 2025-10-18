import os 
import faiss
import numpy as np
import time
from google import genai

# --- Veri Hazırlama ve RAG İndeksini Oluşturma Fonksiyonu ---
def prepare_rag_data(api_key):
    
    # 💥 EN BASİT VE STREAMLIT UYUMLU DOSYA YOLU TANIMI
    file_name = "Enerji_verimliligi_eğitim_kitabi.txt"
    
    # 1. Dosyayı Okuma
    try:
        # Doğrudan dosya adını kullanıyoruz
        with open(file_name, 'r', encoding='utf-8') as file:
            text_content = file.read()
    except FileNotFoundError:
        # Hata durumunda, doğru dosya adını içeren hata mesajını yazarız.
        print(f"HATA: Veri dosyası bulunamadı: {file_name}")
        return [], None
    
    # 2. Metin Parçalama (Chunking) İşlemi
    # İYİLEŞTİRME: Boyut 750'ye düşürüldü ve 75 karakter çakışma eklendi
    def simple_chunking(text, chunk_size=750, chunk_overlap=75): 
        chunks = []
        current_chunk = ""
        
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Mevcut parçaya ekle
            if len(current_chunk) + len(line) + 1 < chunk_size:
                current_chunk += line + '\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Yeni parçaya geçmeden önce çakışmayı yönet
                overlap_text = current_chunk[-chunk_overlap:].strip() if len(current_chunk) > chunk_overlap else ""
                
                # Yeni parçayı başlat
                current_chunk = overlap_text + '\n' + line + '\n'
            
            i += 1
            
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # Parçalama işlemini başlat
    text_chunks = simple_chunking(text_content, chunk_size=750, chunk_overlap=75) 
    
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
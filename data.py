import os
import streamlit as st
from google import genai
import chromadb # Yeni import
from chromadb.utils import embedding_functions # Opsiyonel: Kendi embedding fonksiyonunuzu tanımlamak için
import math # Artık gerekli değil, kaldırılabilir

# --- 1. COSINE SIMILARITY ---
# Bu fonksiyon artık gerekli değil, çünkü ChromaDB aramayı dahili olarak yönetiyor.
# Eski sürümden kaldırıldı.

# --- 2. RAG VERİ HAZIRLAMA (Chroma Versiyonu) ---
# Veritabanı dosya yolları ve isimleri tanımlandı
CHROMA_DB_PATH = "chroma_db_cache"
CHROMA_COLLECTION_NAME = "enerji_verimliligi_kitabi"

@st.cache_resource
def prepare_rag_data(api_key):
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return None

    try:
        # 1. Chroma İstemcisi ve Koleksiyonu Oluştur
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Google'ın text-embedding-004 modelini kullanmak için
        # Chroma'ya özel bir Google Embeddings fonksiyonu tanımlanmalıdır.
        # Bu örnekte, embedding'leri harici olarak oluşturup Chroma'ya ekleyeceğiz.
        
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    except Exception as e:
        st.error(f"ChromaDB Başlatma Hatası: {e}")
        return None

    # Eğer koleksiyon boşsa, veriyi oku, parçala ve embedding oluşturup kaydet.
    if collection.count() == 0:
        st.info("Veritabanı boş. Embedding'ler oluşturuluyor ve diske kaydediliyor (Bu biraz sürebilir)...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
            return None

        # Daha kısa chunk boyutu → yanıt kesilmesini önler
        chunk_size = 2000
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        print(f"{len(text_chunks)} metin parçası için embedding oluşturuluyor (arka planda).")

        embeddings_to_add = []
        ids_to_add = []
        documents_to_add = []

        try:
            client = genai.Client(api_key=api_key)
            progress_bar = st.progress(0)

            for i, chunk in enumerate(text_chunks):
                # 3. Embedding Oluştur
                response = client.models.embed_content(
                    model='text-embedding-004',
                    contents=[chunk]
                )
                
                # Embedding'leri doğru şekilde al
                if hasattr(response, "embeddings"):
                    embedding_vector = response.embeddings[0].values
                else:
                    embedding_vector = response.embedding

                embeddings_to_add.append(embedding_vector)
                documents_to_add.append(chunk)
                ids_to_add.append(f"doc_{i}")
                
                progress_bar.progress((i + 1) / len(text_chunks))

            progress_bar.empty()
            
            # 4. Toplu Kayıt İşlemi
            collection.add(
                embeddings=embeddings_to_add,
                documents=documents_to_add,
                ids=ids_to_add
            )
            st.success(f"{len(documents_to_add)} parça ChromaDB'ye kaydedildi ve kullanıma hazır.")
            
        except Exception as e:
            st.error(f"Embedding/Chroma Kayıt Hatası: {e}")
            return None
    
    else:
        # Veritabanı önceden oluşturulmuşsa bilgi ver
        st.info(f"ChromaDB'den {collection.count()} parça yüklendi (Önbellek kullanıldı).")
        
    return collection

# --- 3. SORGULAMA FONKSİYONU (Chroma Versiyonu) ---
# Artık text_chunks ve embeddings yerine 'collection' nesnesini alıyor.
def simple_query_streamlit(prompt, collection, api_key):
    try:
        client = genai.Client(api_key=api_key)

        # 1. Prompt'un Embedding'ini Oluştur
        prompt_for_embed = f"Bu sorunun anlamı: {prompt} (enerji verimliliği bağlamında)"

        prompt_response = client.models.embed_content(
            model='text-embedding-004',
            contents=[prompt_for_embed]
        )

        if hasattr(prompt_response, "embeddings"):
            prompt_embedding = prompt_response.embeddings[0].values
        else:
            prompt_embedding = prompt_response.embedding
            
        # 2. ChromaDB'de En Benzer Chunk'ları Ara
        # Mevcut kodunuzdaki top_k = 3'ü koruduk
        results = collection.query(
            query_embeddings=[prompt_embedding],
            n_results=3 
        )

        # 3. Geri Çekilen Metni Hazırla
        # ChromaDB sonuçları bir dictionary içinde gelir. documents alanını alıyoruz.
        if results and results.get('documents') and results['documents'][0]:
            retrieved_text = "\n\n---\n\n".join(results['documents'][0])
        else:
            st.warning("Vektör araması sonuç vermedi. Boş kaynak metinle devam ediliyor.")
            retrieved_text = ""

        # Gereksiz uzunlukları kes (Token limitini aşmayı önler)
        max_chars = 3000
        retrieved_text = retrieved_text[:max_chars]
        
    except Exception as e:
        st.warning(f"Vektör Arama Hatası: {e}. Basit aramaya geçiliyor (Bu, Chroma entegrasyonu ile artık çok önerilmez).")
        # Basit arama yerine, hatayı rapor edip devam etmek daha iyidir.
        retrieved_text = ""

    # --- MODEL CEVAP ÜRETİMİ ---
    try:
        # RAG Prompt'u (Aynı kaldı)
        rag_prompt = (
            f"Sen 'Enerji Verimliliği AI Chatbot'u olarak Enerji Verimliliği Eğitim Kitabı'na dayalı "
            f"bir yapay zeka asistansın.\n"
            f"Kullanıcı sorularını aşağıdaki kaynak metinlere dayanarak mantıklı, detaylı ve kişiselleştirilmiş "
            f"bir şekilde cevapla.\n"
            f"- Kitaptaki bilgilerden yararlanarak soruya uygun akıl yürütebilirsin.\n"
            f"- Eğer kaynak metinde doğrudan bilgi yoksa, mantıklı çıkarımlar yapabilirsin, "
            f"ama kitaptaki konulardan çok sapmamalısın.\n"
            f"- Bilgiye tamamen dayalı olmayan veya gerçek dışı (halüsinasyon) cevaplar üretme.\n\n"
            f"KAYNAK METİN:\n---\n{retrieved_text}\n---\n\n"
            f"KULLANICI SORUSU: {prompt}\n\n"
            f"Cevabı eksiksiz, anlaşılır ve mantıksal olarak tutarlı şekilde ver."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt
        )

        # Cevap çok kısaysa tekrar dene (Aynı kaldı)
        if len(response.text) < 50 and retrieved_text:
            follow_up_prompt = rag_prompt + "\nLütfen cevabı daha detaylı ve mantıklı şekilde açıkla."
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=follow_up_prompt
            )

        return response.text, retrieved_text

    except Exception as e:
        st.error(f"Sorgulama Hatası: {e}")
        return "Üzgünüm, sorgulama sırasında bir hata oluştu.", retrieved_text
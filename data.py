import os
import streamlit as st
from google import genai
import chromadb
from chromadb.utils import embedding_functions 
import time # Hata durumunda beklemek için eklendi
from google.genai.errors import APIError # API hatalarını yakalamak için eklendi

# --- SABİTLER ---
CHROMA_DB_PATH = "chroma_db_cache"
CHROMA_COLLECTION_NAME = "enerji_verimliligi_kitabi"

# --- 2. RAG VERİ HAZIRLAMA (Chroma Versiyonu) ---
@st.cache_resource
def prepare_rag_data(api_key):
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"

    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return None

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    except Exception as e:
        st.error(f"ChromaDB Başlatma Hatası: {e}")
        return None

    if collection.count() == 0:
        print(">> [RAG LOG] Veritabanı boş. Embedding'ler oluşturuluyor ve diske kaydediliyor...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
            return None

        chunk_size = 2000
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        print(f">> [RAG LOG] {len(text_chunks)} metin parçası için embedding oluşturuluyor.")

        embeddings_to_add = []
        ids_to_add = []
        documents_to_add = []

        try:
            client = genai.Client(api_key=api_key)
            progress_bar = st.progress(0)

            for i, chunk in enumerate(text_chunks):
                response = client.models.embed_content(
                    model='text-embedding-004',
                    contents=[chunk]
                )
                
                embedding_vector = response.embeddings[0].values if hasattr(response, "embeddings") else response.embedding

                embeddings_to_add.append(embedding_vector)
                documents_to_add.append(chunk)
                ids_to_add.append(f"doc_{i}")
                
                progress_bar.progress((i + 1) / len(text_chunks))

            progress_bar.empty()
            
            collection.add(
                embeddings=embeddings_to_add,
                documents=documents_to_add,
                ids=ids_to_add
            )
            print(f">> [RAG LOG] {len(documents_to_add)} parça ChromaDB'ye kaydedildi ve kullanıma hazır.")
            
        except Exception as e:
            st.error(f"Embedding/Chroma Kayıt Hatası: {e}")
            return None
    
    else:
        print(f">> [RAG LOG] ChromaDB'den {collection.count()} parça yüklendi (Önbellek kullanıldı).")
        
    return collection

# --- 3. SORGULAMA FONKSİYONU (Chroma Versiyonu) ---
def simple_query_streamlit(prompt, collection, api_key):
    # Bu fonksiyon artık generator (yield) kullanacak ve app.py'ye akışı döndürecek.
    
    # 1. Vektör Arama Aşaması
    try:
        client = genai.Client(api_key=api_key)

        prompt_for_embed = f"Bu sorunun anlamı: {prompt} (enerji verimliliği bağlamında)"

        prompt_response = client.models.embed_content(
            model='text-embedding-004',
            contents=[prompt_for_embed]
        )

        prompt_embedding = prompt_response.embeddings[0].values if hasattr(prompt_response, "embeddings") else prompt_response.embedding
            
        results = collection.query(
            query_embeddings=[prompt_embedding],
            n_results=3 
        )

        if results and results.get('documents') and results['documents'][0]:
            retrieved_text = "\n\n---\n\n".join(results['documents'][0])
        else:
            st.warning("Vektör araması sonuç vermedi.")
            retrieved_text = ""

        retrieved_text = retrieved_text[:3000] # Maksimum karakter

    except Exception as e:
        # Vektör arama hatası varsa, kullanıcıya hatayı yield et
        yield f"Vektör Arama Hatası: {e}. Kaynak metin bulunamadı."
        return 

    # --- 2. MODEL CEVAP ÜRETİMİ (Streaming ve Retry Logic) ---
    try:
        rag_prompt = (
            "Sen 'Enerji Verimliliği AI Chatbot'u olarak Enerji Verimliliği Eğitim Kitabı'na dayalı "
            "bir yapay zeka asistansın.\n"
            "Kullanıcı sorularını aşağıdaki kaynak metinlere dayanarak mantıklı, detaylı ve kişiselleştirilmiş "
            "bir şekilde cevapla.\n"
            "- Kitaptaki bilgilerden yararlanarak soruya uygun akıl yürütebilirsin.\n"
            "- Eğer kaynak metinde doğrudan bilgi yoksa, mantıklı çıkarımlar yapabilirsin, "
            "ama kitaptaki konulardan çok sapmamalısın.\n"
            "- Bilgiye tamamen dayalı olmayan veya gerçek dışı (halüsinasyon) cevaplar üretme.\n\n"
            "KAYNAK METİN:\n---\n" + retrieved_text + "\n---\n\n"
            "KULLANICI SORUSU: " + prompt + "\n\n"
            "Cevabı eksiksiz, anlaşılır ve mantıksal olarak tutarlı şekilde ver."
        )

        # Otomatik Tekrar Deneme (Retry Logic)
        max_retries = 3
        response_stream = None
        
        for attempt in range(max_retries):
            try:
                # Streaming için generate_content_stream kullanılıyor
                response_stream = client.models.generate_content_stream(
                    model='gemini-2.5-flash',
                    contents=rag_prompt
                )
                # Başarılı akış başlatılırsa döngüden çık
                break
            except APIError as e:
                error_message = str(e)
                # 503 hatasını yakala
                if "503" in error_message and attempt < max_retries - 1:
                    # Kullanıcıya bekleme mesajını yield et
                    yield f"Sunucu Aşırı Yüklendi (503). {attempt + 1}. deneme başarısız. 5 saniye sonra tekrar deneniyor..."
                    time.sleep(5) 
                else:
                    # Başka bir hata türü veya son deneme başarısız
                    raise e

        # Stream'i tüket ve her bir parçayı yield et
        if response_stream:
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                elif chunk.prompt_feedback.block_reason.name != "BLOCK_REASON_UNSPECIFIED":
                    # Güvenlik filtresine takılma durumu
                    yield "⚠️ Üzgünüm, sorunuz güvenlik politikalarımız tarafından filtrelendiği için yanıt üretemiyorum."
                    return
            
            # Orijinal kodunuzdaki 50 karakter kontrolü kaldırıldı (streaming ilkesine aykırı olduğu için)
        
        else:
             # Tüm denemeler başarısız olduysa veya akış hiç oluşmadıysa
             yield "Modelden cevap alınamadı. Tüm denemeler başarısız oldu."


    except Exception as e:
        # Eğer retry logic başarısız olursa bu hata mesajı döner
        yield f"Sorgulama Hatası: {e}"
        return
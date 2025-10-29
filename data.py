import os
import streamlit as st
from google import genai
import chromadb
from chromadb.utils import embedding_functions 
import time

# --- SABİTLER ---
CHROMA_DB_PATH = "chroma_db_cache"
CHROMA_COLLECTION_NAME = "enerji_verimliligi_kitabi"
MAX_RESPONSE_TOKENS = 512 

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
    try:
        client = genai.Client(api_key=api_key)

        prompt_for_embed = f"Bu sorunun anlamı: {prompt} (enerji verimliliği bağlamında)"

        prompt_response = client.models.embed_content(
            model='text-embedding-004',
            contents=[prompt_for_embed]
        )

        prompt_embedding = prompt_response.embeddings[0].values if hasattr(prompt_response, "embeddings") else prompt_response.embedding
            
        # HIZ OPTİMİZASYONU 1: n_results 3'ten 1'e düşürüldü
        results = collection.query(
            query_embeddings=[prompt_embedding],
            n_results=1 # Sadece en alakalı tek parçayı çekerek RAG modelinin okuma süresini kısaltır
        )

        if results and results.get('documents') and results['documents'][0]:
            retrieved_text = "\n\n---\n\n".join(results['documents'][0])
        else:
            st.warning("Vektör araması sonuç vermedi.")
            retrieved_text = ""

        retrieved_text = retrieved_text[:3000] # Maksimum karakter

    except Exception as e:
        st.warning(f"Vektör Arama Hatası: {e}.")
        retrieved_text = ""

    # --- MODEL CEVAP ÜRETİMİ (Retry Logic ve Hız Optimizasyonu) ---
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
        
        # HIZ OPTİMİZASYONU 2: Yanıt uzunluğunu sınırlandıran config oluşturuldu
        config = genai.types.GenerateContentConfig(
            max_output_tokens=MAX_RESPONSE_TOKENS 
        )

        # Otomatik Tekrar Deneme (Retry Logic)
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=rag_prompt,
                    config=config # Yeni config eklendi
                )
                # Cevap başarıyla alındıysa döngüden çık
                break 
            except Exception as e:
                error_message = str(e)
                # Hata kodu 503 (UNAVAILABLE) içeriyorsa ve son deneme değilse tekrar dene
                if "503" in error_message and attempt < max_retries - 1:
                    st.warning(f"Sunucu Aşırı Yüklendi (503). {attempt + 1}. deneme başarısız. 5 saniye sonra tekrar deneniyor...")
                    time.sleep(5) # 5 saniye bekle
                else:
                    # Başka bir hata türü veya son deneme başarısız
                    raise e # Hatayı dışarıya fırlat

        # Tüm denemeler başarısız olduysa hata mesajı döndürülür
        if response is None:
             raise Exception("Modelden cevap alınamadı. Tüm denemeler başarısız oldu.")
        
        # YENİ KONTROL: response.text'in None olup olmadığını kontrol et (Güvenlik filtreleri için)
        if response.text is None:
            # Eğer yanıt filtrelendiyse, NoneType hatası vermeden kullanıcıya bilgi ver
            return "Üzgünüm, sorunuzun içeriği güvenlik politikalarımız tarafından filtrelendiği için yanıt üretemiyorum.", retrieved_text

        
        # Cevap çok kısaysa tekrar dene (Mevcut mantığınız korunuyor)
        if len(response.text) < 50 and retrieved_text:
            follow_up_prompt = rag_prompt + "\nLütfen cevabı daha detaylı ve mantıklı şekilde açıkla."
            
            # İKİNCİ ÇAĞRI İÇİN GÜVENLİK KONTROLÜ
            follow_up_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=follow_up_prompt,
                config=config 
            )
            # İkinci yanıtta filtrelenme olasılığına karşı kontrol
            if follow_up_response.text is not None:
                response = follow_up_response # Yeni yanıtı kullan
            # Aksi takdirde (filtreli ise), ilk yanıtı (kısa olsa bile) kullanmaya devam eder.
            
        return response.text, retrieved_text

    except Exception as e:
        # Eğer retry logic başarısız olursa bu hata mesajı döner
        st.error(f"Sorgulama Hatası: {e}")
        return "Üzgünüm, Gemini sunucuları şu anda çok yoğun veya beklenmedik bir hata oluştu. Lütfen biraz sonra tekrar deneyin.", retrieved_text
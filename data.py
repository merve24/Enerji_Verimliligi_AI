import os 
import streamlit as st
from google import genai 

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA ---
# Bu fonksiyon kullanılmadığı için sadeleştirilmiştir.
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    pass

# --- 2. ANA FONKSİYON: VERİSİNİ HAZIRLAMA ---
# Tüm kitabı parçalara ayırır ve text_chunks listesini döndürür.
def prepare_rag_data(api_key):
    
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    
    # Hata kontrolü
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return [], None 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None
    
    # KRİTİK DÜZELTME: Metin Parçalama (Chunking)
    text_chunks = []
    chunk_size = 2000 
    
    for i in range(0, len(text), chunk_size):
        text_chunks.append(text[i:i + chunk_size])
    
    return text_chunks, None 

# --- 3. SORGULAMA FONKSİYONU (Eksik Olan ve Hata Veren Kısım) ---
def simple_query_streamlit(prompt, text_chunks, api_key):
    
    # 1. RAG AŞAMASI: En alakalı metin parçasını/parçalarını bulma (Basit anahtar kelime eşleştirme)
    # RAG implementasyonu (vektör tabanı olmadığı için basit arama yapılır)
    relevant_chunks = [chunk for chunk in text_chunks if any(word.lower() in chunk.lower() for word in prompt.split() if len(word) > 2)]
    
    # Eğer alakalı parça bulunamazsa, ilk 2-3 parçayı (Giriş Bölümü) kullan.
    if not relevant_chunks:
        retrieved_text = "\n\n---\n\n".join(text_chunks[:3])
    else:
        # Bulunan tüm alakalı parçaları birleştir.
        retrieved_text = "\n\n---\n\n".join(relevant_chunks)

    # 2. ÜRETİM AŞAMASI: Gemini API çağrısı
    try:
        client = genai.Client(api_key=api_key)
        
        # Gemini modeline verilecek RAG talimatı
        rag_prompt = (
            f"Aşağıdaki alıntı, bir enerji verimliliği eğitim kitabından alınmıştır. "
            f"Kullanıcının sorusunu cevaplamak için yalnızca bu alıntıyı (kaynak metni) kullanın. "
            f"Alıntıda cevap yoksa, kibarca 'Bu sorunun cevabını elimdeki metinde bulamıyorum.' deyin.\n\n"
            f"KAYNAK METİN:\n---\n{retrieved_text}\n---\n\n"
            f"KULLANICI SORUSU: {prompt}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt
        )
        
        # KRİTİK: İki değer döndürülmeli: Cevap ve Kaynak Metin
        return response.text, retrieved_text
        
    except Exception as e:
        st.error(f"Sorgulama Hatası: Gemini API çağrısında bir sorun oluştu. Hata: {e}")
        # Hata durumunda bile iki değer döndürülerek TypeError önlenir.
        return "Üzgünüm, sorgulama sırasında bir hata oluştu.", retrieved_text
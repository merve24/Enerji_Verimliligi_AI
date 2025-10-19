import os 
import streamlit as st
from google import genai 

# --- 1. YARDIMCI FONKSİYON: METİN PARÇALAMA ---
def simple_chunking(text, chunk_size=2000, chunk_overlap=200):
    pass

# --- 2. ANA FONKSİYON: VERİSİNİ HAZIRLAMA (RAG Verisi) ---
def prepare_rag_data(api_key):
    
    file_path = "Enerji_verimliligi_eğitim_kitabi.txt"
    
    # Dosya Konumu Kontrolü
    if not os.path.exists(file_path):
        st.error(f"HATA: Veri dosyası '{file_path}' bulunamadı.")
        return [], None 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return [], None
    
    # Metin Parçalama (Chunking)
    text_chunks = []
    chunk_size = 2000 # Parça boyutu
    
    for i in range(0, len(text), chunk_size):
        text_chunks.append(text[i:i + chunk_size])
    
    return text_chunks, None 

# --- 3. SORGULAMA FONKSİYONU (Kota Optimizasyonu Yapıldı) ---
def simple_query_streamlit(prompt, text_chunks, api_key):
    
    # 1. RAG AŞAMASI: En alakalı metin parçasını/parçalarını bulma
    # Basit anahtar kelime eşleştirme
    relevant_chunks = [
        chunk for chunk in text_chunks 
        if any(word.lower() in chunk.lower() for word in prompt.split() if len(word) > 2)
    ]
    
    # --- KRİTİK KOTA OPTİMİZASYONU ---
    # Gönderilen token miktarını azaltmak için sadece ilk 3 alakalı parçayı kullanın.
    if not relevant_chunks:
        # Eğer alakalı parça bulunamazsa, ilk 3 parçayı kullan (Giriş bölümü)
        retrieved_text = "\n\n---\n\n".join(text_chunks[:3])
    else:
        # Sadece en alakalı İLK ÜÇ parçayı birleştir
        retrieved_text = "\n\n---\n\n".join(relevant_chunks[:3]) 
    # --- OPTİMİZASYON SONU ---

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
        
        # Beklenen iki değeri döndür: Cevap ve Kaynak Metin
        return response.text, retrieved_text
        
    except Exception as e:
        # Kota aşımı durumunda (429) yakalanan hata.
        st.error(f"Sorgulama Hatası: Gemini API çağrısında bir sorun oluştu. Hata: {e}")
        return "Üzgünüm, sorgulama sırasında bir hata oluştu.", retrieved_text
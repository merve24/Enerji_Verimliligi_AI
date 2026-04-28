# 💡 Enerji Verimliliği AI Chatbot

Bu proje, enerji verimliliği, mevzuat ve bina enerji performansı yönetmelikleri üzerine geliştirilmiş yapay zekâ destekli bir bilgi asistanıdır.  
Sistem, yalnızca güvenilir dokümanlara dayanarak cevap üretir ve halüsinasyon riskini minimize eder.

---

##  Özellikler

-  Çoklu kaynaklı bilgi sistemi (RAG)
  - Enerji verimliliği eğitim kitabı
  - Enerji verimliliği mevzuatı
  - Binalarda enerji performansı yönetmeliği

-  Google Gemini AI ile akıllı yanıt üretimi
-  ChromaDB tabanlı semantik arama sistemi
-  Halüsinasyon azaltılmış, kaynak odaklı cevaplama
-  Modern ve sade chat arayüzü (Streamlit)
-  Çoklu dokümandan bağlam oluşturma

---

##  Sistem Nasıl Çalışır?

1. Kullanıcı soru sorar  
2. Sistem soruyla ilgili en alakalı metin parçalarını üç farklı kaynaktan bulur:
   - Eğitim kitabı
   - Mevzuat
   - Bina enerji performansı yönetmeliği  

3. ChromaDB üzerinden en uygun içerikler getirilir (RAG sistemi)  
4. Google Gemini modeli sadece bu kaynaklara dayanarak cevap üretir  
5. Eğer yeterli bilgi yoksa sistem:
   > "Kaynaklarda bulunamadı" yanıtını verir

---

##  Kullanılan Veri Kaynakları

Proje aşağıdaki dosyalardan beslenir:

- `Enerji_verimliligi_eğitim_kitabi.txt`
- `Enerji Verimliliği Mevzuatı.txt`
- `Binalarda enerji performansı yönetmeliği.txt`

---

## ⚙️ Kurulum

### 1. Projeyi klonla
```bash
git clone https://github.com/kullanici/enerji_verimliligi_ai.git
cd enerji_verimliligi_ai
---

## Kullanılan Teknolojiler

Proje, modern **GenAI**, **RAG (Retrieval-Augmented Generation)** ve **vektör tabanlı arama** teknolojileri üzerine inşa edilmiştir:

| Bileşen | Görev | Teknoloji |
|----------|--------|-----------|
| Yapay Zekâ Modeli | Soru anlama, bağlamdan cevap üretme ve açıklama | **Google Gemini (Flash modeli)** |
| Embedding Modeli | Metinleri sayısal vektörlere dönüştürme (semantik arama için) | **text-embedding-004** |
| Vektör Veritabanı | Benzer içerikleri hızlı şekilde bulma ve indeksleme | **ChromaDB (Persistent Vector Store)** |
| Web Arayüzü | Kullanıcı ile etkileşimli sohbet arayüzü | **Streamlit** |
| Veri Kaynağı | Enerji verimliliği ve bina yönetmelikleri içeriği | **Enerji Verimliliği Eğitim Kitabı + Mevzuat + Bina Enerji Performansı Yönetmeliği (.txt)** |

---

## Proje Yapısı

Projenin temel dizin ve dosya yapısı aşağıdaki gibidir:


enerji-verimliligi-ai-chatbot/
│
├── app.py # Streamlit tabanlı kullanıcı arayüzü
├── data.py # RAG pipeline, veri işleme ve LLM sorguları
│
├── Enerji_verimliligi_eğitim_kitabi.txt # Ana bilgi kaynağı (eğitim içeriği)
├── Enerji Verimliliği Mevzuatı.txt # Mevzuat dokümanı
├── Binalarda enerji performansı yönetmeliği.txt # Yönetmelik veri kaynağı
│
├── requirements.txt # Python bağımlılıkları
│
├── .streamlit/ # Streamlit gizli yapılandırma klasörü
│ └── secrets.toml # API anahtarları (GEMINI_API_KEY)
│
└── README.md # Proje dokümantasyonu                     

---

## Kurulum ve Çalıştırma Adımları

### 1. Depoyu Klonlayın
git clone https://github.com/enerji-verimliligi-ai-chatbot.git
cd enerji-verimliligi-ai-chatbot

### 2. Sanal Ortam Oluşturun
python -m venv venv

### 3. Sanal Ortamı Etkinleştirin
Windows için:
venv\Scripts\activate
Mac/Linux için:
source venv/bin/activate

### 4. Bağımlılıkları Yükleyin
pip install -r requirements.txt

### 5. API Anahtarını Tanımlayın
 .env dosyası oluşturun ve içine ekleyin:
echo 'GEMINI_API_KEY="SİZİN_GEMINI_API_ANAHTARINIZ"' > .env

### 6. Uygulamayı Başlatın
streamlit run app.py

 Uygulamayı Çalıştırın:
 http://localhost:8501

---

## Geliştirici Bilgileri

- **Geliştirici:** Merve Nur Öztürk  
- **E-posta:** [mervenurozturk24@gmail.com](mailto:mervenurozturk24@gmail.com)  
- **LinkedIn:** [linkedin.com/in/merve-nur-ozturk](https://www.linkedin.com/in/merve-nur-ozturk)




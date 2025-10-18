# 💡 Enerji Verimliliği RAG Uzmanı Chatbot

Enerji Verimliliği AI Chatbot, 1000+ sayfalık **"Enerji Verimliliği Eğitim Kitabı"** içeriğini temel alan, **Retrieval-Augmented Generation (RAG)** mimarisiyle geliştirilmiş özel bir **yapay zekâ danışmanıdır**.  
Model, enerji yönetimi, HVAC sistemleri, sürdürülebilirlik, ölçme ve doğrulama gibi konularda güvenilir, kaynak temelli yanıtlar üretir.
  
**Retrieval-Augmented Generation (RAG)** mimarisi sayesinde, model genel bilgi yerine kitaptan aldığı güvenilir verilere dayanarak akıl yürütür ve kişiselleştirilmiş, derinlemesine cevaplar üretir.
Amaç, enerji sektöründe çalışan mühendisler, danışmanlar ve öğrenciler için teknik bilgilere **doğrudan, hızlı ve doğrulanabilir erişim** sağlamaktır.

**Temel hedef:** 

“Enerji verimliliği bilincini artırmak ve sürdürülebilir enerji uygulamalarını destekleyen akıllı bir bilgi asistanı oluşturmak.”

---

## ✨ Özellikler ve Kullanım Durumları

Bu uzman chatbot, **Enerji Verimliliği** alanında hızlı, doğru ve bağlamsal bilgi sağlamak üzere tasarlanmıştır.

### 🔹 Ana Özellikler

- **Bağlamsal Güvenilirlik:** Yüksek doğruluk için cevaplarını sadece 1000+ sayfalık teknik dokümantasyon (Kitap) ile sınırlar.  
- **Derinlemesine Uzmanlık:** Karmaşık teknik terimleri, formülleri ve sistem analizlerini (HVAC, Aydınlatma, Motorlar, Kojenerasyon vb.) kitaptaki verilere göre açıklar.  
- **Hızlı Erişim:** Enerji etüt raporları veya ekonomik analiz yöntemleri (Net Bugünkü Değer, İç Karlılık Oranı) gibi kritik bilgilere saniyeler içinde ulaşım sağlar.

---

## 👥 Kimler Kullanabilir?

- **Enerji Yöneticileri ve Mühendisler:** Uygulayacakları Enerji Verimliliği Önlemleri (EVÖ) hakkında hızlı teknik detaylara ve fizibilite bilgilerine ihtiyaç duyan profesyoneller.  
- **Enerji Danışmanlık Şirketleri (ESCO):** Proje teklifleri hazırlarken veya sözleşme detaylarını (Ölçme ve Doğrulama - Ö&D) netleştirirken uzman bağlam arayan danışmanlar.  
- **Üniversite Öğrencileri ve Akademisyenler:** Enerji verimliliği dersleri ve akademik çalışmaları için güvenilir birincil kaynak bilgisine erişmek isteyenler.

---

## 🚀 Kullanılan Teknolojiler

Projenin çekirdeğini oluşturan **Retrieval-Augmented Generation (RAG)** mimarisi, aşağıdaki modern **GenAI** ve veri işleme araçları üzerine inşa edilmiştir:

| Bileşen | Görev | Teknoloji |
|----------|--------|-----------|
| **Model** | Akıl Yürütme ve Cevap Üretme | Gemini 2.5 Flash |
| **Vektörleştirme (Embedding)** | Metinleri sayısal vektörlere dönüştürme | text-embedding-004 |
| **Veri Deposu** | Vektörlerin hızlı aranması | FAISS (Facebook AI Similarity Search) |
| **Arayüz** | Kullanıcı etkileşimli web uygulaması | Streamlit |
| **Veri Kaynağı** | 1000 sayfalık "Enerji Verimliliği Eğitim Kitabı" | enerji_verimliligi_eğitim_kitabi.txt |

---

## 📂 Proje Yapısı

Projenin temel dizin ve dosya yapısı aşağıdaki gibidir:

enerji-verimliligi-ai-chatbot/│
├── app.py                        
├── data.py                       
├── Enerji_verimliligi_eğitim_kitabi.txt 
├── requirements.txt              
├── .streamlit/                 
│   └── secrets.toml              
└── README.md                      

---

## 🚀 Kurulum ve Çalıştırma Adımları

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

 Uygulama genellikle şu adreste açılır:
 http://localhost:8501



----



İletişim

Geliştirici: Merve Nur Öztürk
E-posta: mervenurozturk24@gmail.com

LinkedIn: linkedin.com/in/merve-nur-ozturk

💚 “Enerjini Bilgiyle Verimli Kullan!”
Made with ☁️ by Merve Nur Öztürk

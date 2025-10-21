# 💡 Enerji Verimliliği RAG Uzmanı Chatbot

**Enerji Verimliliği AI Chatbot**, 1000+ sayfalık *"Enerji Verimliliği Eğitim Kitabı"* verisine dayanan, **Retrieval-Augmented Generation (RAG)** mimarisiyle geliştirilmiş özel bir yapay zekâ aracıdır.  
Model; enerji yönetimi, HVAC sistemleri, sürdürülebilirlik, ölçümleme ve arızalanma gibi konularda **güvenilir, kaynak tabanlı ve tutarlı** yanıtlar üretir.

RAG mimarisi sayesinde, model genel bilgiye değil; doğrudan **kitaptan alınan doğrulanabilir verilere** dayanarak akıl yürütme ve kişiselleştirilmiş yanıtlar oluşturur.  
Amaç, enerji sektöründe çalışan mühendisler, danışmanlar ve teknik uzmanlar için **doğru, hızlı ve kaynak temelli bilgiye erişim** sağlamaktır.

---

## 🎯 Temel Hedef

> **“Enerji verimliliği bilincini artırmak ve sürdürülebilir enerji yönetimini destekleyen akıllı bir bilgi altyapısı oluşturmak.”**

---
## 🎬 Uygulama Önizlemesi
![Enerji Verimliliği Chatbot Demo](demo.gif)
### Canlı Uygulama (Deployment) Linki:https://genaibootcampprojesi-nhh2nty3oesoj9yzhrzqyn.streamlit.app/

---

## ✨ Özellikler ve Kullanım Durumları

Bu uzman chatbot, enerji verimliliği alanında **hızlı, doğru ve bağlamsal bilgi** sağlamak için tasarlanmıştır.

### 🔹 Ana Özellikler

- **Bağlamsal Güvenilirlik:** Yanıtlarını yalnızca 1000+ sayfalık *Enerji Verimliliği Eğitim Kitabı* verisinden üretir.  
- **Derinlemesine Uzmanlık:** HVAC, aydınlatma, motorlar, kojenerasyon gibi teknik konuları kitap içeriğine dayanarak açıklar.  
- **Hızlı Erişim:** Enerji etüt raporları veya ekonomik analiz yöntemleri (Net Bugünkü Değer, İç Karlılık Oranı vb.) saniyeler içinde erişilebilir.  
- **Halüsinasyonsuz Yanıtlar:** Kaynak dışı, doğrulanmamış bilgiler üretilmez.

---

## 👥 Kimler Kullanabilir?

- **Enerji Yöneticileri ve Mühendisler:** Enerji Verimliliği Önlemleri (EVÖ) ve fizibilite analizleri hakkında teknik bilgiye ihtiyaç duyan profesyoneller.  
- **Enerji Danışmanlık Şirketleri (ESCO):** Ölçme ve Doğrulama (Ö&D) süreçleri veya teklif hazırlığı sırasında hızlı bilgiye erişmek isteyen ekipler.  
- **Üniversite Öğrencileri ve Akademisyenler:** Enerji verimliliği ve sürdürülebilirlik konularında güvenilir kaynak arayan araştırmacılar.

---

## 🚀 Kullanılan Teknolojiler

Proje, modern **GenAI** ve **veri işleme** teknolojileri üzerine inşa edilmiştir:

| Bileşen | Görev | Teknoloji |
|----------|--------|-----------|
| Model | Akıl yürütme ve içerik üretimi | **Gemini 2.5 Flash** |
| Vektörleştirme (Embedding) | Metinleri dijital vektörlere dönüştürme | **text-embedding-004** |
| Veri Deposu | Vektörlerin hızlı aranması | **FAISS (Facebook AI Similarity Search)** |
| Arayüz | Etkileşimli web uygulaması | **Streamlit** |
| Veri Kaynağı | 1000+ sayfalık *Enerji Verimliliği Eğitim Kitabı* | **Enerji_verimliligi_eğitim_kitabi.txt** |

---

## 📂 Proje Yapısı

Projenin temel dizin ve dosya yapısı aşağıdaki gibidir:


enerji-verimliligi-ai-chatbot/
│
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

 Uygulamayı Çalıştırın:
 http://localhost:8501

---

İletişim

Geliştirici: Merve Nur Öztürk
E-posta: mervenurozturk24@gmail.com
LinkedIn: linkedin.com/in/merve-nur-ozturk



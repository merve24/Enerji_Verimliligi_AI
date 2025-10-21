# Enerji Verimliliği RAG Uzmanı Chatbot

**Enerji Verimliliği AI Chatbot**, 1000+ sayfalık *"Enerji Verimliliği Eğitim Kitabı"* verisine dayanan, **Retrieval-Augmented Generation (RAG)** mimarisiyle geliştirilmiş özel bir yapay zekâ aracıdır.  
Model; enerji yönetimi, HVAC sistemleri, sürdürülebilirlik, yenilenebilir enerji kaynakları ve çevre–enerji ilişkisi gibi konularda **güvenilir, kaynak tabanlı ve tutarlı** yanıtlar üretir.

RAG mimarisi sayesinde, model genel bilgiye değil; doğrudan **kitaptan alınan doğrulanabilir verilere** dayanarak akıl yürütme ve kişiselleştirilmiş yanıtlar oluşturur.  
Amaç, enerji sektöründe çalışan mühendisler, danışmanlar ve teknik uzmanlar için **doğru, hızlı ve kaynak temelli bilgiye erişim** sağlamaktır.

---

## Temel Hedef

> **“Enerji verimliliği bilincini artırmak ve sürdürülebilir enerji yönetimini destekleyen akıllı bir bilgi altyapısı oluşturmak.”**

---
## 🎬 Uygulama Önizlemesi
![Enerji Verimliliği Chatbot Demo](demo.gif)
### Canlı Uygulama (Deployment) Linki:https://genaibootcampprojesi-nhh2nty3oesoj9yzhrzqyn.streamlit.app/

---

## Ana Özellikler

**Kaynak Temelli Güvenilirlik:**  
Yanıtlarını yalnızca 1000+ sayfalık *Enerji Verimliliği Eğitim Kitabı* verisinden üretir.

**Uzmanlık Alanları:**  
- Sürdürülebilir kalkınma, çevre–enerji ilişkisi, ekosistem güvenliği  
- Yenilenebilir enerji kaynakları (güneş, rüzgâr, hidroelektrik, biyokütle, jeotermal)  
- Enerji verimliliği uygulamaları (binalarda, sanayide, su ve atık yönetiminde)  
- İklim değişikliği, karbon emisyonları ve çevresel etkiler konularında özet ve yönlendirici bilgiler sunar  

**Hızlı Bilgi Erişimi:**  
Enerji etüt raporları, sürdürülebilir kalkınma hedefleri, döngüsel ekonomi ve enerji politikalarına ilişkin bilgilere saniyeler içinde erişim sağlar.  

**Doğrulanabilir Yanıtlar:**  
Yanıtlarını yalnızca verilen veri setinden üretir; kaynak dışı, doğrulanmamış bilgiler oluşturmaz.  

---

## Kimler Kullanabilir?

**Enerji Yöneticileri ve Mühendisler:**  
Enerji verimliliği, emisyon azaltımı ve sürdürülebilir enerji politikalarıyla ilgili bilgilere hızlı erişmek isteyen profesyoneller.  

**Enerji Danışmanlık Şirketleri (ESCO):**  
Ölçme, doğrulama, enerji etütleri ve yeşil dönüşüm planlarında kaynak tabanlı bilgilerle çalışan ekipler.  

**Üniversite Öğrencileri ve Akademisyenler:**  
Enerji verimliliği, sürdürülebilir kalkınma, çevre yönetimi ve iklim değişikliği konularında güvenilir Türkçe kaynaklara dayalı araştırmalar yapmak isteyenler.

---

## Kullanılan Teknolojiler

Proje, modern **GenAI** ve **veri işleme** teknolojileri üzerine inşa edilmiştir:

| Bileşen | Görev | Teknoloji |
|----------|--------|-----------|
| Model | Akıl yürütme ve içerik üretimi | **Gemini 2.5 Flash** |
| Vektörleştirme (Embedding) | Metinleri dijital vektörlere dönüştürme | **text-embedding-004** |
| Veri Deposu | Vektörlerin hızlı aranması | **Kosinüs Benzerliği (Manuel Hesaplama)** |
| Arayüz | Etkileşimli web uygulaması | **Streamlit** |
| Veri Kaynağı | 1000+ sayfalık *Enerji Verimliliği Eğitim Kitabı* | **Enerji_verimliligi_eğitim_kitabi.txt** |

---

## Proje Yapısı

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

İletişim

Geliştirici: Merve Nur Öztürk
E-posta: mervenurozturk24@gmail.com
LinkedIn: linkedin.com/in/merve-nur-ozturk



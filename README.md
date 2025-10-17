🚀 Sürdürülebilir İşletme Enerji Danışmanı RAG Temelli Chatbot'u

Bu proje, Akbank GenAI Bootcamp kapsamında, işletmelerin ve bireylerin sürdürülebilirlik ve enerji verimliliği konularındaki sorularını hızlı ve doğru bir şekilde yanıtlamak amacıyla geliştirilmiş, Retrieval-Augmented Generation (RAG) mimarisine dayalı bir yapay zeka sohbet robotudur.

🌟 Ana Özellikler ve Kullanım Senaryoları

Bu chatbot, enerji verimliliği alanındaki karmaşık teknik bilgilere hızlı, güvenilir ve etkileşimli erişim sağlar.

Akıl Yürütme ve Kişiselleştirme: Model, sadece alıntı yapmak yerine, kitaptaki bilgileri analiz eder, yorumlar ve kullanıcıya özel senaryolara uyarlayarak akıl yürütmeli cevaplar sunar.

Güvenilir Kaynak (Grounding): Cevaplar, Enerji Verimliliği Eğitim Kitabı içeriğine sıkı sıkıya bağlıdır, bu da genel LLM tahminlerinin önüne geçerek bilgi güvenilirliğini artırır.

Şeffaf Geri Alma: Her cevabın temelini oluşturan kaynak metin parçaları (RAG Retrieval) gösterilerek kullanıcının bilginin kökenini doğrulaması sağlanır.

Hızlı Performans: Önceden hesaplanmış FAISS indeksi ve Streamlit'in önbellekleme mekanizması (@st.cache_resource) sayesinde, büyük veri setine saniyeler içinde erişilir.

İçin İdeal:

Enerji verimliliği denetçileri ve danışmanları.

Sektördeki mevzuatlar ve teknik uygulamalar hakkında hızlı bilgi arayan işletme yöneticileri.

Akademik çalışma yapan öğrenciler ve araştırmacılar.

🧠 Çözüm Mimarisi (RAG İşlem Hattı)

Proje, tam teşekküllü bir RAG (Retrieval-Augmented Generation) işlem hattı uygulamaktadır:

Bileşen

Görev

Teknoloji

Büyük Dil Modeli (LLM)

Cevap Sentezi

Gemini 2.5 Flash

Vektör Gömme

Metin Dönüşümü

text-embedding-004

Vektör Depolama

Hızlı Arama

FAISS (CPU)

Arayüz

Dağıtım

Streamlit

Veri

Temel Bilgi Kaynağı

Enerji_verimliligi_eğitim_kitabi.txt

RAG Akışı

Veri Hazırlama: Enerji_verimliligi_eğitim_kitabi.txt dosyası okunur ve data.py tarafından anlam bütünlüğünü koruyan parçalara ayrılır (Chunking).

Vektörleştirme: Bu parçalar, Google'ın gömme modeli ile sayısal vektörlere dönüştürülür.

İndeksleme: Vektörler, hızlı arama için FAISS indeksi olarak diske kaydedilir.

Sorgulama: Kullanıcı sorusu vektörleştirilir ve FAISS'te en alakalı metin parçaları (kaynak bağlam) çekilir.

Cevap Üretimi: Çekilen bağlam ve kullanıcı sorusu, Gemini 2.5 Flash modeline gönderilerek nihai, güvenilir cevap üretilir.

⚙️ Yerel Kurulum ve Çalıştırma

1. Dosya Yapısı

Proje yapısı, modüler ve temiz bir mimari sunar:

/enerji-verimliligi-rag/
├── app.py                      # Streamlit arayüzü ve RAG sorgu döngüsü.
├── data.py                     # Veri hazırlama, vektörleştirme ve FAISS indeksi oluşturma.
├── requirements.txt            # Gerekli tüm Python kütüphaneleri.
└── Enerji_verimliligi_eğitim_kitabi.txt # Bilgi kaynağı dosyası.


2. Adımlar

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları sırasıyla izleyin:

a. Klonlama ve Bağımlılıklar

# Depoyu klonlayın ve klasöre geçin
git clone <DEPO_ADRESİ>
cd <PROJE_KLASÖRÜ>

# Gerekli paketleri yükleyin
pip install -r requirements.txt


b. API Anahtarını Tanımlama

Chatbot'un Gemini API'ye erişimi için anahtarınızı ortam değişkeni olarak ayarlayın:

# Linux/macOS
export GEMINI_API_KEY="SİZİN_API_ANAHTARINIZ_BURAYA"

# Windows (CMD)
set GEMINI_API_KEY="SİZİN_API_ANAHTARINIZ_BURAYA"


c. Uygulamayı Başlatma

Anahtar tanımlandıktan sonra Streamlit uygulamasını başlatın:

streamlit run app.py


Tarayıcınızda otomatik olarak açılan adrese gidin.

🔗 Dağıtım ve İletişim

Canlı Dağıtım

Uygulamanın çalışan, canlı demosu Streamlit Cloud üzerinden erişilebilir:
🔗 Canlı Uygulama Linki

CHATBOT ARAYÜZÜ: https://genaibootcampprojesi-tvfvbdqspt4mpkuasvszkd.streamlit.app/



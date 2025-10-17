💡 Enerji Verimliliği RAG Uzmanı Chatbot


Bu proje, Google'ın Gemini 2.5 Flash modelini kullanarak, 1000 sayfayı aşkın "Enerji Verimliliği Eğitim Kitabı" içeriği üzerine kurulmuş özel bir Soru-Cevap (Q&A) aracıdır. Retrieval-Augmented Generation (RAG) mimarisi sayesinde, model genel bilgi yerine kitaptan aldığı güvenilir verilere dayanarak akıl yürütür ve kişiselleştirilmiş, derinlemesine cevaplar üretir.

📸 Canlı Demo ve Görünüm

Canlı uygulamayı ziyaret etmek veya projenin arayüzünü görmek için:

[Ekran Görüntüsü veya GIF Yer Tutucusu]

NOT: Buraya projenizin Streamlit arayüzünün bir ekran görüntüsünü veya GIF'ini ekleyiniz.

🌟 Ana Özellikler ve Avantajlar

Bağlama Dayalı Akıl Yürütme: Yalnızca metin parçalarını kopyalamak yerine, gelişmiş Prompt Mühendisliği ile kitaptaki bilgiyi analiz ederek özel durumlar için akıl yürütmeli ve mantıklı cevaplar üretir.

Yüksek Güvenilirlik (Grounding): Cevaplar, genel LLM bilgisi yerine doğrudan resmi eğitim dokümanından alınır, bu da bilginin güvenilirliğini ve doğruluğunu maksimize eder.

Şeffaf Kaynak Gösterimi: Kullanıcılara her cevabın temelini oluşturan kaynak metin parçalarını göstererek bilginin kaynağını doğrulama imkanı sunar.

Hızlı Performans: Önceden oluşturulmuş FAISS indeksi sayesinde, 1000 sayfalık veri içinde saniyeler içinde alakalı bilgiye ulaşır.

🚀 Mimari ve Teknolojiler

Bileşen

Görev

Teknoloji

Büyük Dil Modeli (LLM)

Cevap Üretme ve Akıl Yürütme

Gemini 2.5 Flash

Vektörleştirme (Embedding)

Metinleri Vektöre Çevirme

text-embedding-004

Vektör Veritabanı

Hızlı Arama İndeksi

FAISS (Facebook AI Similarity Search)

Arayüz

Kullanıcı Arayüzü

Streamlit

🛠️ Kurulum ve Çalıştırma Kılavuzu

1. Ön Gereksinimler

Projenin çalıştırılması için bir Google Gemini API Anahtarı gereklidir. Anahtarınızı Google AI Studio üzerinden alabilirsiniz.

2. Bağımlılıklar

Depoyu klonladıktan sonra, gerekli tüm Python paketlerini (requirements.txt üzerinden) yükleyin:

pip install -r requirements.txt


3. API Anahtarının Tanımlanması (Zorunlu)

a. Streamlit Cloud (Canlı Uygulama) İçin

API anahtarınızı Streamlit'in Secrets bölümüne aşağıdaki formatta eklemeniz en güvenli yöntemdir:

GEMINI_API_KEY="SİZİN_API_ANAHTARINIZ_BURAYA"


b. Yerel Çalıştırma (Opsiyonel) İçin

Uygulamayı yerel makinede çalıştırmak için anahtarı ortam değişkeni olarak tanımlayın ve uygulamayı başlatın:

# Linux/macOS
export GEMINI_API_KEY="SİZİN_API_ANAHTARINIZ_BURAYA"
streamlit run app.py


⚙️ Dosya Yapısı

Proje, görevleri ayrılmış modüllerden oluşan temiz bir yapıya sahiptir:

Dosya Adı

Açıklama

app.py

Uygulama Giriş Noktası: Streamlit arayüzünü, sohbet geçmişini ve RAG sorgu döngüsünü yönetir.

data.py

Veri İşleme: Kaynak metni parçalara ayırır, vektörleştirir ve FAISS arama indeksini oluşturur.

requirements.txt

Projenin tüm Python kütüphanesi bağımlılıklarını listeler.

Enerji_verimliligi_eğitim_kitabi.txt

Chatbot'un bilgi aldığı 1000 sayfalık ham metin veri kaynağıdır.

🔗 Canlı Uygulama Linki

Canlı uygulamaya aşağıdaki bağlantıdan erişebilirsiniz:

[CHATBOT ARAYÜZÜNE GİTMEK İÇİN TIKLAYINIZ](https://genaibootcampprojesi-tvfvbdqspt4mpkuasvszkd.streamlit.app/)]

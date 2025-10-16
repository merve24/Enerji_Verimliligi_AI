# 🚀 Sürdürülebilir İşletme Enerji Danışmanı RAG Temelli Chatbot'u

Bu proje, Akbank GenAI Bootcamp kapsamında, işletmelerin ve bireylerin sürdürülebilirlik ve enerji verimliliği konularındaki sorularını hızlı ve doğru bir şekilde yanıtlamak amacıyla geliştirilmiş, RAG (Retrieval Augmented Generation) mimarisine dayalı bir yapay zeka sohbet robotudur.

## 1. Projenin Amacı

Projenin temel amacı, karmaşık ve teknik bilgi içeren bir eğitim dokümanını (Enerji Verimliliği Kitabı) temel alarak, bu bilgileri kullanıcı dostu ve etkileşimli bir arayüz aracılığıyla sunmaktır.

* **Temel Hedef:** Kullanıcının doğal dilde sorduğu sorulara, enerji verimliliği dokümanındaki en alakalı kısımları (kaynakları) referans göstererek bağlamsal ve doğru cevaplar üretmek.
* **Katkı:** Sürdürülebilirlik bilincini ve enerji verimliliği uygulamalarına erişimi kolaylaştırmak.

## 2. Veri Seti Hakkında Bilgi

Chatbot'un bilgi tabanını oluşturan veri seti, Türkiye Cumhuriyeti Enerji ve Tabii Kaynaklar Bakanlığı'nın **Enerji Verimliliği Eğitim Kitabı**'ndan derlenmiştir (`Enerji_verimliligi_eğitim_kitabi-1-200.txt`).

* **Konu Kapsamı:** Veri seti, geniş bir yelpazede sürdürülebilirlik, çevre ve enerji yönetimi konularını kapsamaktadır:
    * Sürdürülebilirlik ve Sürdürülebilir Kalkınma Kavramları.
    * Çevre ve Enerji İlişkisi, Ekosistem Bütünlüğü.
    * Enerji Yönetimi ve Verimliliği (İşletme seviyesinde uygulamalar ve önlemler).
    * Su ve Atık Yönetimi gibi kritik sürdürülebilirlik başlıkları.
* **Amacı:** İşletmelere ve danışmanlara, enerji tasarrufu potansiyellerini belirleme ve verimlilik artırıcı projelere rehberlik etme konusunda bilgi sağlamaktır.

## 3. Kullanılan Yöntemler ve Çözüm Mimariniz

Bu chatbot, **RAG (Retrieval Augmented Generation)** mimarisi üzerine inşa edilmiştir.

### Kullanılan Ana Teknolojiler

| Bileşen | Görevi | Tahmini Araç/API |
| :--- | :--- | :--- |
| **Büyük Dil Modeli (LLM)** | Cevap üretme ve akıllı etkileşim | `<Gemini API Model Adı (Örn: gemini-2.5-flash)>` |
| **RAG Çatısı** | Veri işleme, sorgu yönetimi | `<LangChain veya Haystack veya Benzeri Kütüphane>` |
| **Vektör Veritabanı** | Metin parçalarını depolama (Gömme) | `<ChromaDB, Pinecone, FAISS veya Benzeri>` |
| **Web Arayüzü** | Kullanıcı ile etkileşim | `<Streamlit veya Gradio>` |

### RAG Akışı (Çözüm Mimarisi)

1.  **Veri Hazırlama (Chunking):** Yüklenen büyük doküman, anlam bütünlüğünü koruyacak şekilde küçük parçalara (chunk) ayrılır.
2.  **Vektörleştirme (Embedding):** Bu metin parçaları, bir Vektör Gömme Modeli (Embedding Model) kullanılarak sayısal vektörlere dönüştürülür ve Vektör Veritabanına kaydedilir.
3.  **Sorgulama (Retrieval):** Kullanıcı bir soru sorduğunda, bu soru da vektörleştirilir ve veritabanında en yakın (en alakalı) metin parçaları çekilir.
4.  **Cevap Üretimi (Generation):** Çekilen alakalı metin parçaları, kullanıcının orijinal sorusuyla birlikte **Gemini** büyük dil modeline bir komut (Prompt) olarak gönderilir.
5.  **Sonuç:** Gemini, bu bağlama dayanarak doğru, kaynağa dayalı cevabı üretir ve kullanıcıya sunar.

## 4. Elde Edilen Sonuçlar (Proje Tamamlandıktan Sonra Doldurulacaktır)

* `<Projeniz çalıştıktan sonra elde ettiğiniz en çarpıcı başarıyı/sonucu yazınız. Örn: "Chatbot, enerji verimliliği yatırımlarının geri ödeme süresi hesaplamaları gibi teknik konularda bile yüksek doğrulukla cevap üretebilmiştir.">`
* `<Projenin kaç saniyede cevap ürettiği gibi performans metrikleri ekleyebilirsiniz.>`

## 5. Çalışma Kılavuzu ve Kurulum (Opsiyonel: Detaylar İçin Ayrı Bir Döküman Varsa Link Verilir)

Bu projeyi yerel ortamınızda çalıştırmak için izlenecek adımlar:

1.  ...
2.  ...

## 6. Canlı Demo (Web Arayüzü Linki)

Projenin çalışan, canlı demosu ve arayüzü aşağıdaki linkte mevcuttur.

**🔗 CHATBOT ARAYÜZÜ:** `<Streamlit, Gradio veya Hugging Face Spaces Deployment Linkinizi Buraya Ekleyiniz>`

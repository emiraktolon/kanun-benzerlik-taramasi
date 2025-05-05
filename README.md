# Kanun Taslağı – Önceki Kanunlarla Benzerlik Taraması

Bu proje, kanun taslaklarının içerik benzerliğini önceki kanun metinleriyle karşılaştırmak amacıyla doğal dil işleme (NLP) teknikleriyle gerçekleştirilmiştir. Amaç, yasa taslaklarının mevcut mevzuatla ne kadar örtüştüğünü metin madenciliği ile incelemektir.

## 📌 Proje Kapsamı
- Çoklu `.txt` dosyalarının otomatik işlenmesi (`data/raw/`)
- Ön işleme (tokenization, stopword removal, stemming, lemmatization)
- Zipf yasası görselleştirmesi (isteğe bağlı)
- TF-IDF vektörleştirme
- Word2Vec model eğitimi
- **Kanun metinleri arası benzerlik ölçümü** (Cosine Similarity)

## 📁 Klasör Yapısı

```
kanun-benzerlik-taramasi/
├── data/
│   └── raw/                  # Tüm .txt formatlı kanun metinleri (taslak ve eski metinler)
│
├── outputs/
│   ├── tfidf/                # TF-IDF çıktı dosyaları
│   └── similarity/           # Benzerlik skorları (csv)
│
├── models/                   # Word2Vec modelleri
├── preprocess.py             # Çoklu metinleri işler
├── vectorize.py              # Vektörleştirme + benzerlik analizi
└── README.md
```

## 🔧 Gereksinimler

```bash
pip install nltk pandas gensim scikit-learn matplotlib numpy
```

İlk çalıştırmada gerekli NLTK bileşenlerini indirin:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Kullanım

1. Tüm `.txt` dosyalarını `data/raw/` klasörüne koy (örneğin: `taslak.txt`, `kanun_2001.txt`, `kanun_2010.txt`)
2. `preprocess.py` dosyasını çalıştır → her dosya için lemmatized/stemmed `.csv`'ler oluşur
3. `vectorize.py` dosyasını çalıştır → TF-IDF & Word2Vec vektörleri çıkar, benzerlik analizleri yapılır
4. Çıktılar `outputs/` klasörüne kaydedilir

## 🧠 Kullanılan Yöntemler

- TF-IDF matrisleri ile Cosine Similarity hesaplaması
- Word2Vec model eğitimi (CBOW ve Skip-Gram; window: 2-4, vector_size: 100-300)

## 👤 Hazırlayan
Emir Aktolon – Yönetim Bilişim Sistemleri  
Odak Adası – 2025

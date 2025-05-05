# Kanun Taslağı Benzerlik Taraması

Bu proje, kanun taslakları gibi resmi metinlerdeki içerik benzerliğini analiz etmeyi amaçlamaktadır. Doğal dil işleme (NLP) yöntemleri kullanılarak metin ön işleme, Zipf yasası analizi, TF-IDF ve Word2Vec temelli vektörleştirme adımları gerçekleştirilmiştir.

## 📌 Proje Amacı

Projede kullanılan metin veri seti üzerinde:
- Temizleme ve ön işleme (tokenization, stopword removal, stemming, lemmatization)
- Zipf yasasına göre metin dağılımının incelenmesi
- TF-IDF ve Word2Vec ile vektörleştirme
- İçerik benzerliğinin analiz edilmesi
amaçlanmaktadır.

## 📁 Klasör Yapısı

```
kanun-taslagi-benzerlik/
│
├── data/
│   ├── raw/                    # Ham veri
│   └── processed/              # Stemming & Lemmatization sonrası veri
├── outputs/
│   ├── tfidf/                  # TF-IDF sonuçları (.csv)
│   └── graphs/                 # Zipf grafikleri ve görselleştirmeler
├── models/                     # Word2Vec modelleri (.model/.bin)
├── preprocess.py              # Temizleme ve ön işleme kodu
├── vectorize.py               # TF-IDF ve Word2Vec kodları
└── README.md
```

## 🔧 Gereksinimler

```bash
pip install nltk pandas gensim scikit-learn matplotlib numpy tqdm seaborn
```

### 📥 NLTK Veri İndirme

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

## 🚀 Kullanım Talimatları

1. `data/raw/` klasörüne ham veri dosyasını yerleştirin.
2. `preprocess.py` dosyasını çalıştırarak veri temizleme işlemini yapın.
3. Zipf yasası grafikleri `outputs/graphs/` klasörüne kaydedilir.
4. `vectorize.py` dosyasını çalıştırarak:
   - TF-IDF çıktıları: `tfidf_lemmatized.csv`, `tfidf_stemmed.csv`
   - Word2Vec modelleri: `.model` formatında 16 adet
     oluşturabilirsiniz.
5. Tüm çıktılar ilgili klasörlerde yer alacaktır.

## 📊 Zipf Yasası

Ham ve işlenmiş veri setlerine ait log-log Zipf dağılımı grafikleri çizilir. Grafikler `outputs/graphs/` klasöründe saklanır.

## 🧠 Vektörleştirme

### TF-IDF
- `tfidf_lemmatized.csv`
- `tfidf_stemmed.csv`

### Word2Vec
Toplam 16 model eğitilmiştir:
- 8 model lemmatized veri için
- 8 model stemmed veri için

Parametreler:
- `window`: 2, 4
- `vector_size`: 100, 300
- `model_type`: CBOW / Skip-gram

Model örnek adları:
- `word2vec_lemmatized_cbow_win2_dim100.model`
- `word2vec_stemmed_skipgram_win4_dim300.model`

## 👤 Yazar

Emir Aktolon – 2025  
Gümüşhane Üniversitesi – Yönetim Bilişim Sistemleri

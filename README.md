# Kanun Benzerlik Taraması (TCK Eski-Yeni Madde Analizi)

Bu proje, Türk Ceza Kanunu’nun eski (1926, 765 sayılı) ve yeni (2005, 5237 sayılı) versiyonları arasındaki benzerlikleri belirlemek amacıyla geliştirilmiştir. Projede 100 eski-yeni madde çifti kullanılmıştır. Veri işleme, vektörleştirme, benzerlik analizi gibi tüm aşamalar Türkçe doğal dil işleme yöntemleriyle gerçekleştirilmiştir.

## 1. Amaç

Yapay zeka destekli bir sistemle iki farklı kanun dönemine ait maddeler arasında içeriksel benzerlikleri analiz etmek. Böylece mevzuat dönüşüm süreçlerinin yapısal izlerini ortaya koymak ve otomatik hukuk karşılaştırmaları için altyapı oluşturmak hedeflenmiştir.

## 2. Kullanılan Teknolojiler ve Kütüphaneler

- Python 3
- Pandas, Scikit-learn
- Gensim (Word2Vec)
- Zeyrek (Türkçe çözümleme için)
- SnowballStemmer
- TQDM

## 3. Veri Seti

- `tck_madde_ciftleri_1_100_clean.csv`  
  → 100 adet madde çifti (eski/yeni)  
  → Sütunlar: `Yıl`, `Kanun`, `Madde No`, `İçerik`

## 4. Adımlar

### 4.1. Ön İşleme

`kanun_on_isleme.py`  
- Zeyrek ile lemmatizasyon (kök bulma)
- SnowballStemmer ile gövdeleme (stemming)
- Çıktılar:
  - `lemmatized.csv`
  - `stemmed.csv`

### 4.2. TF-IDF Vektörleştirme

`tfidf_olustur.py`  
- Lemmatize ve stemmed veriler için ayrı ayrı TF-IDF matrisleri oluşturulur.  
- Modeller:  
  - `tfidf_lemmatized.pkl`  
  - `tfidf_stemmed.pkl`

### 4.3. Word2Vec Model Eğitimi

`word2vec_egit.py`  
- Gensim ile 8 farklı Word2Vec modeli eğitildi.  
- Parametre varyasyonları: `sg` (CBOW/Skip-gram), `window`, `size`

### 4.4. Benzerlik Analizi

`benzerlik_tfidf.py`  
- TF-IDF matrislerinden cosine similarity hesaplanır.  
- En yakın 5 madde CSV’ye yazılır.  
→ `benzerlik_sonuclari_tfidf.csv`

`benzerlik_word2vec.py`  
- 8 Word2Vec modeli için ayrı ayrı benzerlik analizi yapılır.  
- Her model için sonuç CSV’si üretilir.  
→ `benzerlik_sonuclari/lemmatized_model_N.csv`

## 5. Kullanım

```bash
pip install pandas scikit-learn gensim zeyrek snowballstemmer tqdm

python kanun_on_isleme.py
python tfidf_olustur.py
python word2vec_egit.py
python benzerlik_tfidf.py
python benzerlik_word2vec.py

## 6. Klasör Yapısı


📂 Proje Klasörü/
├── tck_madde_ciftleri_1_100_clean.csv
├── lemmatized.csv
├── stemmed.csv
├── tfidf_olustur.py
├── word2vec_egit.py
├── benzerlik_tfidf.py
├── benzerlik_word2vec.py
├── tfidf_lemmatized.pkl
├── tfidf_stemmed.pkl
├── models/ (16 Word2Vec modeli)
├── benzerlik_sonuclari/
│   ├── lemmatized_model_1.csv
│   ├── ...
│   └── lemmatized_model_8.csv
└── benzerlik_sonuclari_tfidf.csv

## 7. Yazar

Ad Soyad: Emir Aktolon
Ders: Yapay Zeka
Teslim Türü: Final Ödevi
Tarih: Mayıs 2025

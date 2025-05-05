import os
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Gerekli NLTK bileşenleri
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 📁 Klasör yolları
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Türkçe stopword listesi (NLTK'de eksik olabilir)
turkish_stopwords = set([
    've', 'bir', 'bu', 'şu', 'ile', 'de', 'da', 'ne', 'için', 'ama', 'gibi', 'daha', 'çok',
    'veya', 'ise', 'ki', 'mi', 'mu', 'mü', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'her', 'hiç'
])

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# 📦 Tüm .txt dosyalarını işle
for filename in os.listdir(RAW_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        # Küçük harfe çevir + noktalama temizliği
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenization + Stopword çıkarma
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in turkish_stopwords]

        # Lemmatize ve stem işlemleri
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
        stemmed = [stemmer.stem(word) for word in tokens]

        # Sonuçları kaydet
        base_name = os.path.splitext(filename)[0]
        lemma_df = pd.DataFrame({"lemmatized": lemmatized})
        stem_df = pd.DataFrame({"stemmed": stemmed})

        lemma_df.to_csv(f"{PROCESSED_DIR}/lemmatized_{base_name}.csv", index=False)
        stem_df.to_csv(f"{PROCESSED_DIR}/stemmed_{base_name}.csv", index=False)

        print(f"{filename} işlendi ✅")

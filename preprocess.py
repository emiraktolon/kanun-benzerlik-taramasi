import os
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Gerekli NLTK bileşenlerini indir
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 📥 1. Ham veriyi oku
RAW_PATH = "data/raw/veri.txt"  # Bu dosyayı daha sonra oluşturacağız
with open(RAW_PATH, "r", encoding="utf-8") as file:
    text = file.read()

# 🔧 2. Küçük harfe çevir, noktalama işaretlerini kaldır
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))

# 🧹 3. Tokenize et, stopword’leri çıkar
tokens = word_tokenize(text)
tokens = [word for word in tokens if word not in stopwords.words("turkish")]

# 🌱 4. Lemmatization ve stemming uygula
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
stemmed = [stemmer.stem(word) for word in tokens]

# 💾 5. Sonuçları kaydet
df_lemma = pd.DataFrame({"lemmatized": lemmatized})
df_stem = pd.DataFrame({"stemmed": stemmed})

# Klasörü oluştur
os.makedirs("data/processed", exist_ok=True)

# CSV olarak dışa aktar
df_lemma.to_csv("data/processed/lemmatized.csv", index=False)
df_stem.to_csv("data/processed/stemmed.csv", index=False)

print("Veri temizleme tamamlandı. CSV dosyaları oluşturuldu.")

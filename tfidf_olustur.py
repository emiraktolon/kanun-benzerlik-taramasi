import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# CSV dosyalarını oku
df_lemma = pd.read_csv("lemmatized.csv", encoding="utf-8-sig")
df_stem = pd.read_csv("stemmed.csv", encoding="utf-8-sig")

# TF-IDF modellerini oluştur
vectorizer_lemma = TfidfVectorizer()
X_lemma = vectorizer_lemma.fit_transform(df_lemma["processed_tokens"])

vectorizer_stem = TfidfVectorizer()
X_stem = vectorizer_stem.fit_transform(df_stem["processed_tokens"])

# .pkl dosyalarına kaydet
with open("tfidf_lemmatized.pkl", "wb") as f:
    pickle.dump((vectorizer_lemma, X_lemma), f)

with open("tfidf_stemmed.pkl", "wb") as f:
    pickle.dump((vectorizer_stem, X_stem), f)

print("✅ TF-IDF modelleri başarıyla oluşturuldu ve kaydedildi.")

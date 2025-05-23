import pandas as pd
from zeyrek import MorphAnalyzer
import snowballstemmer
from tqdm import tqdm

df = pd.read_csv("tck_madde_ciftleri_1_100_clean.csv")

analyzer = MorphAnalyzer()
stemmer = snowballstemmer.stemmer("turkish")

lemmatized = []
stemmed = []

for text in tqdm(df["İçerik"]):
    text = str(text).lower()

    # LEMMATIZATION (yalnızca ilk parse ve tekil kök)
    lemma_words = []
    for word_analysis in analyzer.analyze(text):
        if word_analysis:
            lemma_words.append(word_analysis[0].lemma)
    lemmatized.append(" ".join(lemma_words))

    # STEMMING
    stems = stemmer.stemWords(text.split())
    stemmed.append(" ".join(stems))

# Kayıt
df_lemma = df.copy()
df_lemma["processed_tokens"] = lemmatized
df_lemma.to_csv("lemmatized.csv", index=False, encoding="utf-8-sig")

df_stem = df.copy()
df_stem["processed_tokens"] = stemmed
df_stem.to_csv("stemmed.csv", index=False, encoding="utf-8-sig")

print("✅ DÜZELTİLMİŞ dosyalar: lemmatized.csv & stemmed.csv")

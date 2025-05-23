import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 1. TF-IDF modelini yükle
with open("tfidf_lemmatized.pkl", "rb") as f:
    vectorizer, tfidf_matrix = pickle.load(f)

# 2. CSV dosyasını yükle (lemmatized olan)
df = pd.read_csv("lemmatized.csv")

# 3. Giriş cümlesini belirle (örnek olarak 0. satır)
input_sentence = df["processed_tokens"].iloc[0]

# 4. Vektöre dönüştür
input_vector = vectorizer.transform([input_sentence])

# 5. Cosine benzerliğini hesapla
cosine_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

# 6. En benzer 5 cümleyi seç (kendisi hariç)
top5_indices = cosine_scores.argsort()[-6:-1][::-1]

# 7. Terminale yazdır
print("\n📌 Giriş metni:")
print(df["İçerik"].iloc[0])

print("\n🔍 En benzer 5 cümle:")
for idx in top5_indices:
    print(f"{idx}: {df['İçerik'].iloc[idx]} | Skor: {cosine_scores[idx]:.4f}")

# 8. CSV çıktısı olarak kaydet
results = pd.DataFrame({
    "index": top5_indices,
    "İçerik": df["İçerik"].iloc[top5_indices].values,
    "similarity_score": cosine_scores[top5_indices]
})
results.to_csv("benzerlik_sonuclari_tfidf.csv", index=False, encoding="utf-8-sig")

print("\n✅ Sonuç dosyası oluşturuldu: benzerlik_sonuclari_tfidf.csv")

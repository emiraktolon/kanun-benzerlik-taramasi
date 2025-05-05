import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📁 İşlenecek klasör
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs/similarity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📄 Tüm lemmatized CSV dosyalarını topla
documents = []
filenames = []

for file in os.listdir(PROCESSED_DIR):
    if file.startswith("lemmatized_") and file.endswith(".csv"):
        path = os.path.join(PROCESSED_DIR, file)
        df = pd.read_csv(path)
        text = " ".join(df["lemmatized"].astype(str).tolist())
        documents.append(text)
        filenames.append(file.replace("lemmatized_", "").replace(".csv", ""))

# 🔢 TF-IDF Vektörleri oluştur
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 📐 Cosine Similarity matrisi oluştur
similarity_matrix = cosine_similarity(tfidf_matrix)

# 📊 Sonuçları DataFrame olarak kaydet
similarity_df = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)
similarity_df.to_csv(os.path.join(OUTPUT_DIR, "similarity_matrix.csv"))

print("Benzerlik matrisi başarıyla oluşturuldu! ✅")

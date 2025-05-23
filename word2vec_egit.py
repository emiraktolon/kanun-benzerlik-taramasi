import pandas as pd
from gensim.models import Word2Vec
import os

# CSV dosyasını oku
df = pd.read_csv("lemmatized.csv")

# processed_tokens sütununu boşlukla ayır
sentences = [row.split() for row in df["processed_tokens"].dropna()]

# Model parametre kombinasyonları
params = [
    {"sg": 0, "window": 4, "vector_size": 100},
    {"sg": 0, "window": 4, "vector_size": 300},
    {"sg": 0, "window": 8, "vector_size": 100},
    {"sg": 0, "window": 8, "vector_size": 300},
    {"sg": 1, "window": 4, "vector_size": 100},
    {"sg": 1, "window": 4, "vector_size": 300},
    {"sg": 1, "window": 8, "vector_size": 100},
    {"sg": 1, "window": 8, "vector_size": 300},
]

# Klasör oluştur
os.makedirs("models", exist_ok=True)

# Modelleri eğit ve kaydet
for i, p in enumerate(params, 1):
    print(f"Model {i}: sg={p['sg']}, window={p['window']}, size={p['vector_size']}")
    model = Word2Vec(sentences=sentences, sg=p["sg"], window=p["window"], vector_size=p["vector_size"], min_count=1, workers=4, epochs=100)
    model.save(f"models/lemmatized_model_{i}.model")

print("✅ 8 Word2Vec modeli başarıyla eğitildi ve 'models/' klasörüne kaydedildi.")

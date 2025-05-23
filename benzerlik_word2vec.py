import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Veriyi yükle
df = pd.read_csv("lemmatized.csv")
sentences = [row.split() for row in df["processed_tokens"].dropna()]
texts = df["İçerik"].fillna("")

# Her satır için vektör ortalaması hesaplayan fonksiyon
def get_sentence_vector(model, sentence):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Model klasörü
model_dir = "models"
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".model")])

# Çıktı klasörü
os.makedirs("benzerlik_sonuclari", exist_ok=True)

# Her model için işlem yap
for i, model_name in enumerate(model_files, 1):
    print(f"Model {i}: {model_name}")
    model = Word2Vec.load(os.path.join(model_dir, model_name))
    
    # Cümle vektörlerini hesapla
    vectors = np.array([get_sentence_vector(model, s) for s in sentences])
    
    # Cosine benzerlik matrisi
    similarity_matrix = cosine_similarity(vectors)
    
    # Üst üç benzerlik eşleşmesi
    results = []
    for idx, row in enumerate(similarity_matrix):
        similar_idx = row.argsort()[::-1][1:4]  # En yüksek 3 benzer (ilk kendisi)
        for sim_idx in similar_idx:
            results.append({
                "model": model_name,
                "cümle_1": texts[idx],
                "cümle_2": texts[sim_idx],
                "benzerlik_skoru": row[sim_idx]
            })
    
    # CSV'ye kaydet
    out_path = f"benzerlik_sonuclari/benzerlik_{i}_{model_name.replace('.model', '')}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")

print("✅ Tüm modeller için benzerlik analizleri tamamlandı ve CSV dosyaları oluşturuldu.")

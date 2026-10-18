import pickle
from sentence_transformers import SentenceTransformer

with open('paper/artifacts/v1.0/category_centroids.pkl', 'rb') as f:
    centroids = pickle.load(f)

print("Centroid keys:", list(centroids.keys()))
for k, v in centroids.items():
    print(f"Key: {k}, shape: {v.shape}, norm: {round(float((v**2).sum()**0.5), 4)}")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model max_seq_length:", model.max_seq_length)

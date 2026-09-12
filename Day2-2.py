from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love machine learning",
    "I enjoy studying artificial intelligence",
    "The weather is very nice today"
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(embeddings)

print(similarity)
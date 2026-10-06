from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Regularization helps prevent overfitting in machine learning.",
    "Python is a popular programming language.",
    "CNNs are commonly used for image classification.",
    "RNNs are useful for sequential data.",
    "Transformers use attention mechanisms."
]

query = input("How can I reduce overfitting?")

document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

results = list(zip(documents, similarities))

results.sort(key=lambda x: x[1], reverse=True)

print("\nMost relevant documents:\n")

for document, score in results:
    print(f"{score:.4f} -> {document}")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love machine learning",
    "I enjoy studying artificial intelligence",
    "The weather is very nice today"
]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print("\nSentence:", sentence)
    print("Vector size:", len(embedding))
    print("First 10 values:", embedding[:10])
from ollama import chat

documents = [
    """
    Overfitting occurs when a machine learning model learns the training
    data too closely, including noise and random patterns. As a result,
    the model performs very well on training data but poorly on unseen data.
    """,

    """
    Regularization is a technique used to reduce overfitting. L1 and L2
    regularization add a penalty to the model's objective function and
    discourage overly complex models.
    """,

    """
    Cross-validation is a model evaluation technique. It divides the
    available data into multiple parts and trains and evaluates the model
    multiple times using different train-validation splits.
    """,

    """
    Decision trees can easily overfit when they become very deep. Limiting
    the maximum depth, increasing the minimum number of samples required
    for a split, and pruning can help reduce overfitting.
    """
]
query = input("Enter your question: ")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]
top_k = 2
top_indices = similarities.argsort()[-top_k:][::-1]
for index in top_indices:
    print("\nScore:", similarities[index])
    print(documents[index])

context = "\n\n".join(
    documents[index].strip()
    for index in top_indices
)

print("\n===== CONTEXT FOR LLM =====\n")
print(context)

prompt = f"""
You are a helpful machine learning assistant.

Answer the user's question using the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{query}

Answer:
"""

print("\n===== PROMPT =====\n")
print(prompt)

response = chat(
    model="qwen2.5:1.5b-instruct-q4_K_M",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n===== LLM ANSWER =====\n")
print(response.message.content)
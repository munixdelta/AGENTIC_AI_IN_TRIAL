from ollama import chat

documents = [
    """
    Overfitting is a common problem in machine learning where a model learns
    the training data too closely, including noise and random patterns. A model
    that is overfitted performs very well on the training data but performs
    poorly on unseen data. This happens because the model becomes too complex
    and memorizes specific details instead of learning the general patterns.
    Overfitting can be detected by comparing training and validation performance.
    If training accuracy is very high while validation accuracy is significantly
    lower, the model may be overfitting. Overfitting is especially common when
    the dataset is small or when the model has a large number of parameters.
    Increasing model complexity without enough training data can make the problem
    worse. Techniques such as regularization, cross-validation, early stopping,
    and collecting more training data can help improve generalization and reduce
    overfitting.
    """,

    """
    Regularization is a technique used in machine learning to reduce overfitting
    by discouraging a model from becoming unnecessarily complex. It adds a
    penalty term to the model's objective function. L1 regularization encourages
    some model parameters to become exactly zero, which can also perform feature
    selection. L2 regularization penalizes large parameter values and generally
    keeps the parameters small. The strength of regularization is controlled by
    a parameter that determines how strongly the penalty affects training.
    If regularization is too weak, the model may still overfit. If it is too
    strong, the model may become too simple and underfit the training data.
    Choosing an appropriate regularization strength is therefore important.
    Regularization is widely used in linear models, logistic regression, neural
    networks, and many other machine learning algorithms.
    """,

    """
    Cross-validation is a model evaluation technique used to estimate how well
    a machine learning model will perform on unseen data. In k-fold cross-
    validation, the available dataset is divided into k approximately equal
    parts called folds. The model is trained using k-1 folds and evaluated on
    the remaining fold. This process is repeated until every fold has been used
    as the validation set. The individual evaluation scores are then combined,
    often by calculating their average. Cross-validation provides a more reliable
    estimate of model performance than using only one train-validation split.
    It can also help in selecting models and tuning hyperparameters. However,
    cross-validation can require additional computation because the model must
    be trained multiple times. The choice of k affects both the reliability of
    the estimate and the computational cost.
    """,

    """
    Decision trees are machine learning models that can easily overfit when they
    become very deep. A deep decision tree may create many small branches and
    learn very specific patterns and noise from the training data. One way to
    reduce this problem is to limit the maximum depth of the tree. Another
    technique is increasing the minimum number of samples required to split a
    node. Increasing this value prevents the tree from creating branches based
    on very small groups of observations. Pruning is another useful technique
    in which unnecessary branches are removed from a tree after or during
    training. Setting minimum samples for leaf nodes can also prevent overly
    specific predictions. These techniques reduce the complexity of the tree
    and generally help it perform better on unseen data. Choosing appropriate
    tree parameters is therefore important for controlling overfitting.
    """
]

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


all_chunks = []
chunk_metadata = []

for doc_id, document in enumerate(documents):
    chunks = chunk_text(document)

    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_metadata.append({
            "document_id": doc_id + 1
        })

print("\n===== CHUNKS =====\n")

for i, chunk in enumerate(all_chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
    print()






query = input("Enter your question: ")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(all_chunks)
query_embedding = model.encode([query])

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]
top_k = 4
top_indices = similarities.argsort()[-top_k:][::-1]
for index in top_indices:
    print("\nScore:", similarities[index])
    print("Document:", chunk_metadata[index]["document_id"])
    print(all_chunks[index])

context = "\n\n".join(
    f"[Document {chunk_metadata[index]['document_id']} | Similarity Score: {similarities[index]:.4f}]\n"
    f"{all_chunks[index].strip()}"
    for index in top_indices
)

print("\n===== CONTEXT FOR LLM =====\n")
print(context)

prompt = f"""
You are a helpful machine learning assistant.

Answer the user's question using the provided context.
The context is ordered by similarity score. Higher similarity scores indicate that the retrieved chunk is more relevant to the question. Prefer more relevant chunks when forming the answer.

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
    ],
    options={
        "num_predict": 100
    }
)

print("\n===== LLM ANSWER =====\n")
print(response.message.content)
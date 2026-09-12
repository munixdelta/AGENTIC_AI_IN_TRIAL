import chromadb

client = chromadb.Client()

collection = client.create_collection(
    name="ml_documents"
)

documents = [
    "Overfitting is a common problem in machine learning where a model learns the training data too closely, including noise and random patterns. A model that is overfitted performs very well on the training data but performs poorly on unseen data. This happens because the model becomes too complex and memorizes specific details instead of learning the general patterns. Overfitting can be detected by comparing training and validation performance. If training accuracy is very high while validation accuracy is significantly lower, the model may be overfitting. Overfitting is especially common when the dataset is small or when the model has a large number of parameters. Increasing model complexity without enough training data can make the problem worse. Techniques such as regularization, cross-validation, early stopping, and collecting more training data can help improve generalization and reduce overfitting.",

    "Regularization is a technique used in machine learning to reduce overfitting by discouraging a model from becoming unnecessarily complex. It adds a penalty term to the model's objective function. L1 regularization encourages some model parameters to become exactly zero, which can also perform feature selection. L2 regularization penalizes large parameter values and generally keeps the parameters small. The strength of regularization is controlled by a parameter that determines how strongly the penalty affects training. If regularization is too weak, the model may still overfit. If it is too strong, the model may become too simple and underfit the training data. Choosing an appropriate regularization strength is therefore important. Regularization is widely used in linear models, logistic regression, neural networks, and many other machine learning algorithms.",

    "Cross-validation is a model evaluation technique used to estimate how well a machine learning model will perform on unseen data. In k-fold cross-validation, the available dataset is divided into k approximately equal parts called folds. The model is trained using k-1 folds and evaluated on the remaining fold. This process is repeated until every fold has been used as the validation set. The individual evaluation scores are then combined, often by calculating their average. Cross-validation provides a more reliable estimate of model performance than using only one train-validation split. It can also help in selecting models and tuning hyperparameters. However, cross-validation can require additional computation because the model must be trained multiple times. The choice of k affects both the reliability of the estimate and the computational cost.",

    "Decision trees are machine learning models that can easily overfit when they become very deep. A deep decision tree may create many small branches and learn very specific patterns and noise from the training data. One way to reduce this problem is to limit the maximum depth of the tree. Another technique is increasing the minimum number of samples required to split a node. Increasing this value prevents the tree from creating branches based on very small groups of observations. Pruning is another useful technique in which unnecessary branches are removed from a tree after or during training. Setting minimum samples for leaf nodes can also prevent overly specific predictions. These techniques reduce the complexity of the tree and generally help it perform better on unseen data. Choosing appropriate tree parameters is therefore important for controlling overfitting."
]

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
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

print("\n===== TOTAL CHUNKS =====")
print(len(all_chunks))



from sentence_transformers import SentenceTransformer

from sentence_transformers import CrossEncoder

model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = model.encode(all_chunks).tolist()

ids = [
    f"chunk_{i + 1}"
    for i in range(len(all_chunks))
]

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    ids=ids,
    metadatas=chunk_metadata
)

print("All chunks added successfully!")

query = "How can overfitting be prevented?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=4
)

print("\n===== SEARCH RESULT =====")


for i in range(len(results["ids"][0])):
    print("\nResult:", i + 1)
    print("ID:", results["ids"][0][i])
    print("Document:", results["metadatas"][0][i]["document_id"])
    print("Distance:", results["distances"][0][i])
    print("Text:", results["documents"][0][i])

distances = results["distances"][0]
documents_result = results["documents"][0]

pairs = [
    [query, document]
    for document in documents_result
]

rerank_scores = reranker.predict(pairs)

reranked = sorted(
    zip(rerank_scores, distances, documents_result),
    key=lambda x: x[0],
    reverse=True
)
print("\n===== RERANKED RESULTS =====")

for i, (score, distance, document) in enumerate(reranked):
    print("\nRank:", i + 1)
    print("Rerank Score:", score)
    print("Original Distance:", distance)
    print("Text:", document)

    
top_n = 3

context = "\n\n".join(
    document
    for score, distance, document in reranked[:top_n]
)

print("\n===== FINAL CONTEXT =====")
print(context)



from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Question:
{query}

Context:
{context}
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\n===== FINAL ANSWER =====")
print(response.text)
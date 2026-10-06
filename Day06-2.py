import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv
import os


# ==============================
# 1. FIND ALL PDF FILES
# ==============================

documents_folder = "documents"

pdf_files = [
    file
    for file in os.listdir(documents_folder)
    if file.lower().endswith(".pdf")
]

print("PDF files found:", pdf_files)


# ==============================
# 2. EXTRACT TEXT FROM ALL PDFs
# ==============================

pages = []

for pdf_file in pdf_files:

    pdf_path = os.path.join(documents_folder, pdf_file)

    pdf = pymupdf.open(pdf_path)

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        pages.append({
            "text": text,
            "page_number": page_number + 1,
            "source": pdf_file
        })


print("\nTotal Pages:", len(pages))


print("\n===== FIRST PAGE =====\n")
print(pages[0]["text"])

print("\nPage Number:", pages[0]["page_number"])
print("Source:", pages[0]["source"])


# ==============================
# 3. CHUNKING FUNCTION
# ==============================

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


# ==============================
# 4. CREATE CHUNKS + METADATA
# ==============================

all_chunks = []

chunk_metadata = []


for page in pages:

    chunks = chunk_text(page["text"])

    for chunk in chunks:

        all_chunks.append(chunk)

        chunk_metadata.append({
            "page_number": page["page_number"],
            "source": page["source"]
        })


print("\n===== TOTAL CHUNKS =====")
print(len(all_chunks))


print("\n===== FIRST CHUNK =====")
print(all_chunks[0])

print("\nPage:", chunk_metadata[0]["page_number"])
print("Source:", chunk_metadata[0]["source"])


# ==============================
# 5. CREATE EMBEDDINGS
# ==============================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    all_chunks
).tolist()


# ==============================
# 6. CREATE CHROMADB COLLECTION
# ==============================

client = chromadb.Client()

collection = client.create_collection(
    name="multiple_pdf_rag"
)


# ==============================
# 7. STORE CHUNKS IN CHROMADB
# ==============================

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


print("\n===== CHROMADB =====")
print(
    "Total stored chunks:",
    collection.count()
)


# ==============================
# 8. USER QUERY
# ==============================

query = "What is backpropagation?"

query_embedding = model.encode(
    query
).tolist()


# ==============================
# 9. SEMANTIC RETRIEVAL
# ==============================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


print("\n===== SEARCH RESULTS =====")


for i in range(len(results["ids"][0])):

    print("\nResult:", i + 1)

    print(
        "ID:",
        results["ids"][0][i]
    )

    print(
        "Distance:",
        results["distances"][0][i]
    )

    print(
        "Page:",
        results["metadatas"][0][i]["page_number"]
    )

    print(
        "Source:",
        results["metadatas"][0][i]["source"]
    )

    print(
        "Text:",
        results["documents"][0][i]
    )


# ==============================
# 10. CROSSENCODER RERANKING
# ==============================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


retrieved_chunks = []


for i in range(len(results["ids"][0])):

    document = results["documents"][0][i]

    distance = results["distances"][0][i]

    metadata = results["metadatas"][0][i]

    retrieved_chunks.append(
        (
            document,
            distance,
            metadata
        )
    )


pairs = [
    (query, document)
    for document, distance, metadata
    in retrieved_chunks
]


scores = reranker.predict(pairs)


reranked = []


for i in range(len(retrieved_chunks)):

    document, distance, metadata = retrieved_chunks[i]

    reranked.append(
        (
            scores[i],
            distance,
            document,
            metadata
        )
    )


reranked.sort(
    key=lambda x: x[0],
    reverse=True
)


print("\n===== RERANKED RESULTS =====")


for i, (
    score,
    distance,
    document,
    metadata
) in enumerate(reranked):

    print("\nResult:", i + 1)

    print(
        "CrossEncoder Score:",
        score
    )

    print(
        "Original Distance:",
        distance
    )

    print(
        "Page:",
        metadata["page_number"]
    )

    print(
        "Source:",
        metadata["source"]
    )

    print(
        "Text:",
        document
    )


# ==============================
# 11. FINAL CONTEXT
# ==============================

top_n = 3


context = "\n\n".join(

    f"[Source: {item[3]['source']} | "
    f"Page: {item[3]['page_number']}]\n"
    f"{item[2]}"

    for item in reranked[:top_n]
)


print("\n===== FINAL CONTEXT =====")

print(context)


# ==============================
# 12. GEMINI
# ==============================

load_dotenv()


api_key = os.getenv(
    "GEMINI_API_KEY"
)


client = genai.Client(
    api_key=api_key
)


prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Question:
{query}

Context:
{context}

Give a concise and clear answer.
"""


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


print("\n===== FINAL ANSWER =====")

print(response.text)
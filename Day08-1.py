import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv
import os
from rank_bm25 import BM25Okapi


# ============================================================
# 1. FIND ALL PDF FILES
# ============================================================

documents_folder = "documents"

pdf_files = [
    file
    for file in os.listdir(documents_folder)
    if file.lower().endswith(".pdf")
]

print("PDF files found:", pdf_files)


# ============================================================
# 2. EXTRACT TEXT FROM ALL PDFs
# ============================================================

pages = []

for pdf_file in pdf_files:

    pdf_path = os.path.join(
        documents_folder,
        pdf_file
    )

    pdf = pymupdf.open(pdf_path)

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        pages.append({
            "text": text,
            "page_number": page_number + 1,
            "source": pdf_file
        })

    pdf.close()


print("\nTotal Pages:", len(pages))


# ============================================================
# 3. SHOW FIRST PAGE
# ============================================================

print("\n===== FIRST PAGE =====\n")

print(pages[0]["text"])

print(
    "\nPage Number:",
    pages[0]["page_number"]
)

print(
    "Source:",
    pages[0]["source"]
)


# ============================================================
# 4. CHUNKING FUNCTION
# ============================================================

def chunk_text(
    text,
    chunk_size=50,
    overlap=10
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 5. CREATE CHUNKS + METADATA
# ============================================================

all_chunks = []

chunk_metadata = []


for page in pages:

    chunks = chunk_text(
        page["text"]
    )

    for chunk in chunks:

        all_chunks.append(chunk)

        chunk_metadata.append({

            "page_number":
                page["page_number"],

            "source":
                page["source"]

        })


print("\n===== TOTAL CHUNKS =====")

print(
    len(all_chunks)
)


# ============================================================
# 6. SHOW FIRST CHUNK
# ============================================================

print("\n===== FIRST CHUNK =====")

print(
    all_chunks[0]
)

print(
    "\nPage:",
    chunk_metadata[0]["page_number"]
)

print(
    "Source:",
    chunk_metadata[0]["source"]
)


# ============================================================
# 7. GEMINI CLIENT
# ============================================================

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)

gemini_client = genai.Client(
    api_key=api_key
)


# ============================================================
# 8. ORIGINAL USER QUERY
# ============================================================

original_query = "What is backpropagation?"


# ============================================================
# 9. QUERY REWRITING
# ============================================================

rewrite_prompt = f"""
Rewrite the following user query into a clear,
retrieval-friendly query.

The rewritten query should preserve the original
meaning while making important technical terms
explicit.

Do not answer the question.
Only return the rewritten query.

User query:
{original_query}
"""


rewrite_response = gemini_client.models.generate_content(

    model="gemini-3.6-flash",

    contents=rewrite_prompt

)


rewritten_query = rewrite_response.text.strip()


print(
    "\n===== QUERY REWRITING ====="
)

print(
    "\nOriginal Query:"
)

print(
    original_query
)

print(
    "\nRewritten Query:"
)

print(
    rewritten_query
)


# ============================================================
# 10. BM25 SEARCH
# ============================================================

tokenized_chunks = [

    chunk.lower().split()

    for chunk in all_chunks

]


bm25 = BM25Okapi(
    tokenized_chunks
)


# IMPORTANT:
# BM25 now uses REWRITTEN QUERY

bm25_query = rewritten_query.lower().split()


bm25_scores = bm25.get_scores(
    bm25_query
)


# Get Top 5 BM25 results

top_bm25_indices = sorted(

    range(len(bm25_scores)),

    key=lambda i:
        bm25_scores[i],

    reverse=True

)[:5]


print(
    "\n===== BM25 RESULTS USING REWRITTEN QUERY ====="
)


for rank, index in enumerate(
    top_bm25_indices,
    start=1
):

    print(
        f"\nResult: {rank}"
    )

    print(
        f"ID: chunk_{index + 1}"
    )

    print(
        f"BM25 Score: "
        f"{bm25_scores[index]:.4f}"
    )

    print(
        f"Page: "
        f"{chunk_metadata[index]['page_number']}"
    )

    print(
        f"Source: "
        f"{chunk_metadata[index]['source']}"
    )

    print(
        f"Text: "
        f"{all_chunks[index]}"
    )


# ============================================================
# 11. CREATE SENTENCE EMBEDDINGS
# ============================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


embeddings = model.encode(
    all_chunks
).tolist()


# ============================================================
# 12. CREATE CHROMADB CLIENT
# ============================================================

# IMPORTANT:
# Separate variable from Gemini client

chroma_client = chromadb.Client()


# ============================================================
# 13. CREATE COLLECTION
# ============================================================

collection = chroma_client.create_collection(

    name="day8_query_rewriting"

)


# ============================================================
# 14. STORE CHUNKS IN CHROMADB
# ============================================================

ids = [

    f"chunk_{i + 1}"

    for i in range(
        len(all_chunks)
    )

]


collection.add(

    documents=all_chunks,

    embeddings=embeddings,

    ids=ids,

    metadatas=chunk_metadata

)


print(
    "\n===== CHROMADB ====="
)

print(
    "Total stored chunks:",
    collection.count()
)


# ============================================================
# 15. QUERY EMBEDDING
# ============================================================

# IMPORTANT:
# Semantic search also uses REWRITTEN QUERY

query_embedding = model.encode(
    rewritten_query
).tolist()


# ============================================================
# 16. SEMANTIC RETRIEVAL
# ============================================================

results = collection.query(

    query_embeddings=[
        query_embedding
    ],

    n_results=5

)


print(
    "\n===== SEMANTIC SEARCH RESULTS ====="
)


for i in range(
    len(results["ids"][0])
):

    print(
        "\nResult:",
        i + 1
    )

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
        results["metadatas"][0][i][
            "page_number"
        ]
    )

    print(
        "Source:",
        results["metadatas"][0][i][
            "source"
        ]
    )

    print(
        "Text:",
        results["documents"][0][i]
    )


# ============================================================
# 17. SIMILARITY THRESHOLD
# ============================================================

threshold = 0.80

filtered_results = []


for i in range(
    len(results["ids"][0])
):

    distance = results[
        "distances"
    ][0][i]


    if distance <= threshold:

        filtered_results.append(i)


print(
    "\n===== THRESHOLD FILTERING ====="
)


for i in filtered_results:

    print(

        f"ID: "
        f"{results['ids'][0][i]} | "

        f"Distance: "
        f"{results['distances'][0][i]:.4f}"

    )


print(
    "\nTotal results after threshold:",
    len(filtered_results)
)


# ============================================================
# 18. HYBRID SEARCH — RRF FUSION
# ============================================================

k = 60

rrf_scores = {}


# ------------------------------------------------------------
# BM25 RANKING
# ------------------------------------------------------------

for rank, index in enumerate(
    top_bm25_indices,
    start=1
):

    chunk_id = f"chunk_{index + 1}"

    score = 1 / (
        k + rank
    )

    rrf_scores[chunk_id] = \
        rrf_scores.get(
            chunk_id,
            0
        ) + score


# ------------------------------------------------------------
# SEMANTIC RANKING
# ------------------------------------------------------------

semantic_indices = []


for i in range(
    len(results["ids"][0])
):

    distance = results[
        "distances"
    ][0][i]


    if distance <= threshold:

        chunk_id = results[
            "ids"
        ][0][i]

        semantic_indices.append(
            chunk_id
        )


for rank, chunk_id in enumerate(
    semantic_indices,
    start=1
):

    score = 1 / (
        k + rank
    )

    rrf_scores[chunk_id] = \
        rrf_scores.get(
            chunk_id,
            0
        ) + score


# ------------------------------------------------------------
# SORT COMBINED RESULTS
# ------------------------------------------------------------

hybrid_results = sorted(

    rrf_scores.items(),

    key=lambda x: x[1],

    reverse=True

)


print(
    "\n===== HYBRID SEARCH RESULTS ====="
)


for rank, (
    chunk_id,
    score
) in enumerate(
    hybrid_results,
    start=1
):

    print(

        f"Result: {rank} | "

        f"ID: {chunk_id} | "

        f"RRF Score: {score:.6f}"

    )


# ============================================================
# 19. CROSSENCODER RERANKING
# ============================================================

reranker = CrossEncoder(

    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"

)


# ============================================================
# 20. HYBRID RESULTS → ACTUAL CHUNKS
# ============================================================

retrieved_chunks = []


for chunk_id, rrf_score in hybrid_results:

    # Example:
    # chunk_156 → index 155

    index = int(
        chunk_id.split("_")[1]
    ) - 1


    document = all_chunks[index]

    metadata = chunk_metadata[index]


    retrieved_chunks.append(

        (
            document,
            rrf_score,
            metadata
        )

    )


# ============================================================
# 21. CREATE QUERY-DOCUMENT PAIRS
# ============================================================

# IMPORTANT:
# CrossEncoder also uses REWRITTEN QUERY

pairs = [

    (
        rewritten_query,
        document
    )

    for document,
        rrf_score,
        metadata

    in retrieved_chunks

]


# ============================================================
# 22. CROSSENCODER SCORES
# ============================================================

scores = reranker.predict(
    pairs
)


# ============================================================
# 23. CREATE RERANKED LIST
# ============================================================

reranked = []


for i in range(
    len(retrieved_chunks)
):

    document, rrf_score, metadata = \
        retrieved_chunks[i]


    reranked.append(

        (
            scores[i],
            rrf_score,
            document,
            metadata
        )

    )


# Highest CrossEncoder score first

reranked.sort(

    key=lambda x: x[0],

    reverse=True

)


# ============================================================
# 24. SHOW FINAL RERANKED RESULTS
# ============================================================

print(
    "\n===== HYBRID + CROSSENCODER RESULTS ====="
)


for rank, (
    score,
    rrf_score,
    document,
    metadata

) in enumerate(
    reranked,
    start=1
):

    print(
        f"\nResult: {rank}"
    )

    print(
        f"CrossEncoder Score: "
        f"{score}"
    )

    print(
        f"RRF Score: "
        f"{rrf_score:.6f}"
    )

    print(
        f"Page: "
        f"{metadata['page_number']}"
    )

    print(
        f"Source: "
        f"{metadata['source']}"
    )

    print(
        f"Text: "
        f"{document}"
    )


# ============================================================
# 25. FINAL CONTEXT
# ============================================================

top_n = 3


context = "\n\n".join(

    f"[Source: "
    f"{item[3]['source']} | "
    f"Page: "
    f"{item[3]['page_number']}]\n"
    f"{item[2]}"

    for item in reranked[
        :top_n
    ]

)


print(
    "\n===== FINAL CONTEXT ====="
)

print(
    context
)


# ============================================================
# 26. FINAL GEMINI ANSWER
# ============================================================

# IMPORTANT:
# We ask Gemini about the ORIGINAL question.
# Retrieval was done using the rewritten question.

answer_prompt = f"""
Answer the user's original question using only
the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Original User Question:
{original_query}

Retrieval-Friendly Query:
{rewritten_query}

Context:
{context}

Give a concise and clear answer.
"""


response = gemini_client.models.generate_content(

    model="gemini-3.6-flash",

    contents=answer_prompt

)


# ============================================================
# 27. FINAL ANSWER
# ============================================================

print(
    "\n===== FINAL ANSWER ====="
)

print(
    response.text
)
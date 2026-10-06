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
# 9.5. MULTI-QUERY GENERATION
# ============================================================

multi_query_prompt = f"""
Generate 3 different retrieval queries for the following
user question.

Each query should approach the same information need
from a slightly different perspective.

Do not answer the question.
Only return the 3 queries, one per line.

Original user question:
{original_query}

Retrieval-friendly query:
{rewritten_query}
"""


multi_query_response = gemini_client.models.generate_content(

    model="gemini-3.6-flash",

    contents=multi_query_prompt

)


multi_queries = [

    q.strip()

    for q in multi_query_response.text.splitlines()

    if q.strip()

]


print(
    "\n===== MULTI-QUERY GENERATION ====="
)


for i, q in enumerate(
    multi_queries,
    start=1
):

    print(
        f"Query {i}: {q}"
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
# 10.5. BM25 SEARCH FOR MULTIPLE QUERIES
# ============================================================

multi_bm25_results = {}

for query_number, query in enumerate(
    multi_queries,
    start=1
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:5]

    multi_bm25_results[query_number] = {
        "query": query,
        "indices": top_indices,
        "scores": scores
    }


print(
    "\n===== MULTI-QUERY BM25 RESULTS ====="
)


for query_number, data in multi_bm25_results.items():

    print(
        f"\n--- Query {query_number} ---"
    )

    print(
        data["query"]
    )

    for rank, index in enumerate(
        data["indices"],
        start=1
    ):

        print(
            f"\nRank: {rank}"
        )

        print(
            f"ID: chunk_{index + 1}"
        )

        print(
            f"BM25 Score: "
            f"{data['scores'][index]:.4f}"
        )

        print(
            f"Page: "
            f"{chunk_metadata[index]['page_number']}"
        )

        print(
            f"Source: "
            f"{chunk_metadata[index]['source']}"
        )
        

# ============================================================
# 10.6. MULTI-QUERY RRF FUSION
# ============================================================

multi_query_rrf = {}

k = 60

for query_number, data in multi_bm25_results.items():

    for rank, index in enumerate(
        data["indices"],
        start=1
    ):

        chunk_id = f"chunk_{index + 1}"

        score = 1 / (
            k + rank
        )

        multi_query_rrf[chunk_id] = \
            multi_query_rrf.get(
                chunk_id,
                0
            ) + score


# Sort by combined RRF score

multi_query_hybrid = sorted(
    multi_query_rrf.items(),
    key=lambda x: x[1],
    reverse=True
)


print(
    "\n===== MULTI-QUERY RRF RESULTS ====="
)


for rank, (
    chunk_id,
    score
) in enumerate(
    multi_query_hybrid,
    start=1
):

    print(
        f"Rank: {rank} | "
        f"ID: {chunk_id} | "
        f"RRF Score: {score:.6f}"
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
# 15. MULTI-QUERY EMBEDDINGS
# ============================================================

multi_query_embeddings = model.encode(
    multi_queries
).tolist()


# ============================================================
# 16. MULTI-QUERY SEMANTIC RETRIEVAL
# ============================================================

multi_semantic_results = {}


for query_number, query_embedding in enumerate(
    multi_query_embeddings,
    start=1
):

    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=5

    )

    multi_semantic_results[query_number] = results


    print(
        f"\n===== SEMANTIC RESULTS FOR QUERY {query_number} ====="
    )

    print(
        multi_queries[query_number - 1]
    )


    for rank in range(
        len(results["ids"][0])
    ):

        print(
            f"\nRank: {rank + 1}"
        )

        print(
            f"ID: "
            f"{results['ids'][0][rank]}"
        )

        print(
            f"Distance: "
            f"{results['distances'][0][rank]:.4f}"
        )

        print(
            f"Page: "
            f"{results['metadatas'][0][rank]['page_number']}"
        )

        print(
            f"Source: "
            f"{results['metadatas'][0][rank]['source']}"
        )

# ============================================================
# 17. MULTI-QUERY SIMILARITY THRESHOLD
# ============================================================

threshold = 0.80

filtered_semantic_results = {}


for query_number, results in multi_semantic_results.items():

    filtered_indices = []

    for i in range(
        len(results["ids"][0])
    ):

        distance = results["distances"][0][i]

        if distance <= threshold:

            filtered_indices.append(i)


    filtered_semantic_results[query_number] = \
        filtered_indices


    print(
        f"\n===== THRESHOLD RESULTS FOR QUERY {query_number} ====="
    )

    for i in filtered_indices:

        print(
            f"ID: {results['ids'][0][i]} | "
            f"Distance: {results['distances'][0][i]:.4f} | "
            f"Page: {results['metadatas'][0][i]['page_number']} | "
            f"Source: {results['metadatas'][0][i]['source']}"
        )


    print(
        f"Total results after threshold: "
        f"{len(filtered_indices)}"
    )


# ============================================================
# 18. MULTI-QUERY HYBRID RRF FUSION
# ============================================================

k = 60

final_rrf_scores = {}


# ============================================================
# 18.1 BM25 RESULTS FROM ALL QUERIES
# ============================================================

for query_number, data in multi_bm25_results.items():

    for rank, index in enumerate(
        data["indices"],
        start=1
    ):

        chunk_id = f"chunk_{index + 1}"

        score = 1 / (
            k + rank
        )

        final_rrf_scores[chunk_id] = \
            final_rrf_scores.get(
                chunk_id,
                0
            ) + score


# ============================================================
# 18.2 SEMANTIC RESULTS FROM ALL QUERIES
#     USING THRESHOLD-FILTERED RESULTS
# ============================================================

for query_number, results in multi_semantic_results.items():

    filtered_indices = filtered_semantic_results[query_number]

    for rank, i in enumerate(
        filtered_indices,
        start=1
    ):

        chunk_id = results["ids"][0][i]

        score = 1 / (
            k + rank
        )

        final_rrf_scores[chunk_id] = \
            final_rrf_scores.get(
                chunk_id,
                0
            ) + score


# ============================================================
# 18.3 SORT FINAL RRF RESULTS
# ============================================================

final_hybrid_results = sorted(

    final_rrf_scores.items(),

    key=lambda x: x[1],

    reverse=True

)


print(
    "\n===== MULTI-QUERY HYBRID RRF RESULTS ====="
)


for rank, (
    chunk_id,
    score
) in enumerate(
    final_hybrid_results,
    start=1
):

    index = int(
        chunk_id.split("_")[1]
    ) - 1

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {chunk_id}"
    )

    print(
        f"RRF Score: {score:.6f}"
    )

    print(
        f"Page: "
        f"{chunk_metadata[index]['page_number']}"
    )

    print(
        f"Source: "
        f"{chunk_metadata[index]['source']}"
    )


# ============================================================
# 19. CROSSENCODER RERANKING
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# 20. HYBRID RESULTS → ACTUAL CHUNKS
# ============================================================

candidate_chunks = []


for chunk_id, rrf_score in final_hybrid_results:

    index = int(
        chunk_id.split("_")[1]
    ) - 1

    document = all_chunks[index]

    metadata = chunk_metadata[index]

    candidate_chunks.append(
        (
            chunk_id,
            document,
            rrf_score,
            metadata
        )
    )


# ============================================================
# 21. CROSSENCODER: ALL MULTI-QUERIES × CANDIDATES
# ============================================================

pairs = []

pair_information = []


for chunk_id, document, rrf_score, metadata in candidate_chunks:

    for query_number, query in enumerate(
        multi_queries,
        start=1
    ):

        pairs.append(
            (
                query,
                document
            )
        )

        pair_information.append(
            (
                chunk_id,
                document,
                rrf_score,
                metadata,
                query_number
            )
        )


# ============================================================
# 22. CROSSENCODER SCORES
# ============================================================

scores = reranker.predict(
    pairs
)


# ============================================================
# 23. KEEP BEST SCORE FOR EACH CHUNK
# ============================================================

best_chunk_scores = {}


for i, score in enumerate(scores):

    (
        chunk_id,
        document,
        rrf_score,
        metadata,
        query_number
    ) = pair_information[i]


    if (
        chunk_id not in best_chunk_scores
        or score > best_chunk_scores[chunk_id]["score"]
    ):

        best_chunk_scores[chunk_id] = {

            "score": score,

            "document": document,

            "rrf_score": rrf_score,

            "metadata": metadata,

            "query_number": query_number

        }


# ============================================================
# 24. SORT BY BEST CROSSENCODER SCORE
# ============================================================

reranked = sorted(

    best_chunk_scores.items(),

    key=lambda x: x[1]["score"],

    reverse=True

)


# ============================================================
# 25. SHOW FINAL RERANKED RESULTS
# ============================================================

print(
    "\n===== MULTI-QUERY CROSSENCODER RESULTS ====="
)


for rank, (
    chunk_id,
    data
) in enumerate(
    reranked,
    start=1
):

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {chunk_id}"
    )

    print(
        f"CrossEncoder Score: "
        f"{data['score']:.4f}"
    )

    print(
        f"RRF Score: "
        f"{data['rrf_score']:.6f}"
    )

    print(
        f"Best Matching Query: "
        f"Query {data['query_number']}"
    )

    print(
        f"Page: "
        f"{data['metadata']['page_number']}"
    )

    print(
        f"Source: "
        f"{data['metadata']['source']}"
    )

    print(
        f"Text: "
        f"{data['document']}"
    )



context_threshold = 1.0

filtered_reranked = [
    (chunk_id, data)
    for chunk_id, data in reranked
    if data["score"] >= context_threshold
]

print("\n===== CONTEXT FILTERING =====")

for chunk_id, data in filtered_reranked:
    print(
        f"ID: {chunk_id} | "
        f"CrossEncoder Score: {data['score']:.4f}"
    )

print(
    f"Total chunks after context filtering: "
    f"{len(filtered_reranked)}"
)


# ============================================================
# 26. FINAL CONTEXT
# ============================================================

top_n = 3


context = "\n\n".join(
    f"[Source: {data['metadata']['source']} | "
    f"Page: {data['metadata']['page_number']}]\n"
    f"{data['document']}"
    for chunk_id, data in filtered_reranked[:top_n]
)


print(
    "\n===== FINAL CONTEXT ====="
)

print(
    context
)


# ============================================================
# 27. FINAL GEMINI ANSWER
# ============================================================

answer_prompt = f"""
Answer the user's original question using only
the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Original User Question:
{original_query}

Retrieval Queries:
{chr(10).join(
    f"Query {i + 1}: {q}"
    for i, q in enumerate(multi_queries)
)}

Context:
{context}

Give a concise and clear answer.
"""


response = gemini_client.models.generate_content(

    model="gemini-3.6-flash",

    contents=answer_prompt

)


# ============================================================
# 28. FINAL ANSWER
# ============================================================

print(
    "\n===== FINAL ANSWER ====="
)

print(
    response.text
)
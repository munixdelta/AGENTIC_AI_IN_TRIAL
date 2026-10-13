import pymupdf

import chromadb

from sentence_transformers import SentenceTransformer, CrossEncoder

from dotenv import load_dotenv

import os

from rank_bm25 import BM25Okapi

from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq

import time

import re

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:

    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

# Keep the existing variable name to avoid changing the rest of the project.

gemini_client = Groq(api_key=api_key)

GROQ_MODEL = "openai/gpt-oss-120b"

def generate_with_retry(model=None, contents="", retries=3):

    """Send a prompt to Groq with limited retries for temporary failures."""

    model = model or GROQ_MODEL

    for attempt in range(retries):

        try:

            return gemini_client.chat.completions.create(

                model=model,

                messages=[{"role": "user", "content": contents}],

            )

        except Exception as e:

            error_text = str(e).lower()

            is_rate_limit = "429" in error_text or "rate limit" in error_text

            is_temporary = any(

                code in error_text for code in ("500", "502", "503", "504")

            )

            if not (is_rate_limit or is_temporary) or attempt == retries - 1:

                raise

            wait_time = 10 if is_rate_limit else 2 ** (attempt + 1)

            print(f"Groq temporary error. Retrying in {wait_time} seconds...")

            time.sleep(wait_time)

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

if not pdf_files:

    raise FileNotFoundError("No PDF files found in the ./documents folder.")

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

if not pages:

    raise ValueError("No text pages were extracted. Check that the PDFs contain extractable text.")

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

def split_into_sentences(text):
    """Split text on sentence boundaries while preserving complete sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Keep punctuation with its sentence; do not split on every period blindly.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text, chunk_size=100, overlap=20):
    """Create chunks from complete sentences to avoid cutting evidence mid-sentence."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_sentences and current_words + sentence_words > chunk_size:
            chunks.append(" ".join(current_sentences))

            overlap_sentences = []
            overlap_words = 0
            for previous_sentence in reversed(current_sentences):
                word_count = len(previous_sentence.split())
                if overlap_words + word_count > overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, previous_sentence)
                overlap_words += word_count
                if overlap_words >= overlap:
                    break

            current_sentences = overlap_sentences
            current_words = sum(len(item.split()) for item in current_sentences)

        current_sentences.append(sentence)
        current_words += sentence_words

    if current_sentences:
        final_chunk = " ".join(current_sentences)
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

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

            "page_number": page["page_number"],

            "source": page["source"]

        })

print("\n===== TOTAL CHUNKS =====")

print(len(all_chunks))

# ============================================================

# 6. SHOW FIRST CHUNK

# ============================================================

print("\n===== FIRST CHUNK =====")

if not all_chunks:

    raise ValueError("No text chunks were created from the PDFs.")

print(all_chunks[0])

print(

    "\nPage:",

    chunk_metadata[0]["page_number"]

)

print(

    "Source:",

    chunk_metadata[0]["source"]

)

# ============================================================

# 7. GROQ CLIENT

# Groq client is initialized at the top of this file.

# ============================================================

# ============================================================

# 8. ORIGINAL USER QUERY

# ============================================================

original_query = "What is the role of an optimizer in neural networks?"

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

rewrite_response = generate_with_retry(

    model=GROQ_MODEL,

    contents=rewrite_prompt

)

rewritten_query = (

    (rewrite_response.choices[0].message.content or "").strip()

)

print("\n===== QUERY REWRITING =====")

print("\nOriginal Query:")

print(original_query)

print("\nRewritten Query:")

print(rewritten_query)

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

multi_query_response = generate_with_retry(

    model=GROQ_MODEL,

    contents=multi_query_prompt

)

multi_queries = [

    q.strip()

    for q in (multi_query_response.choices[0].message.content or "").splitlines()

    if q.strip()

]

print("\n===== MULTI-QUERY GENERATION =====")

for i, q in enumerate(multi_queries, start=1):

    print(f"Query {i}: {q}")

# ============================================================

# 10. BM25 SEARCH

# ============================================================

tokenized_chunks = [

    chunk.lower().split()

    for chunk in all_chunks

]

bm25 = BM25Okapi(tokenized_chunks)

# IMPORTANT:

# BM25 now uses REWRITTEN QUERY

bm25_query = rewritten_query.lower().split()

bm25_scores = bm25.get_scores(bm25_query)

# Get Top 5 BM25 results

top_bm25_indices = sorted(

    range(len(bm25_scores)),

    key=lambda i: bm25_scores[i],

    reverse=True

)[:5]

print("\n===== BM25 RESULTS USING REWRITTEN QUERY =====")

for rank, index in enumerate(top_bm25_indices, start=1):

    print(f"\nResult: {rank}")

    print(f"ID: chunk_{index + 1}")

    print(f"BM25 Score: {bm25_scores[index]:.4f}")

    print(f"Page: {chunk_metadata[index]['page_number']}")

    print(f"Source: {chunk_metadata[index]['source']}")

    print(f"Text: {all_chunks[index]}")

# ============================================================

# 10.5. BM25 SEARCH FOR MULTIPLE QUERIES

# ============================================================

multi_bm25_results = {}

for query_number, query in enumerate(multi_queries, start=1):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

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

print("\n===== MULTI-QUERY BM25 RESULTS =====")

for query_number, data in multi_bm25_results.items():

    print(f"\n--- Query {query_number} ---")

    print(data["query"])

    for rank, index in enumerate(data["indices"], start=1):

        print(f"\nRank: {rank}")

        print(f"ID: chunk_{index + 1}")

        print(f"BM25 Score: {data['scores'][index]:.4f}")

        print(f"Page: {chunk_metadata[index]['page_number']}")

        print(f"Source: {chunk_metadata[index]['source']}")

# ============================================================

# 10.6. MULTI-QUERY RRF FUSION

# ============================================================

multi_query_rrf = {}

k = 60

for query_number, data in multi_bm25_results.items():

    for rank, index in enumerate(data["indices"], start=1):

        chunk_id = f"chunk_{index + 1}"

        score = 1 / (k + rank)

        multi_query_rrf[chunk_id] = (

            multi_query_rrf.get(chunk_id, 0) + score

        )

# Sort by combined RRF score

multi_query_hybrid = sorted(

    multi_query_rrf.items(),

    key=lambda x: x[1],

    reverse=True

)

print("\n===== MULTI-QUERY RRF RESULTS =====")

for rank, (chunk_id, score) in enumerate(multi_query_hybrid, start=1):

    print(

        f"Rank: {rank} | "

        f"ID: {chunk_id} | "

        f"RRF Score: {score:.6f}"

    )

# ============================================================

# 11. CREATE SENTENCE EMBEDDINGS

# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(all_chunks).tolist()

# ============================================================

# 12. CREATE CHROMADB CLIENT

# ============================================================

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

    for i in range(len(all_chunks))

]

collection.add(

    documents=all_chunks,

    embeddings=embeddings,

    ids=ids,

    metadatas=chunk_metadata

)

print("\n===== CHROMADB =====")

print("Total stored chunks:", collection.count())

# ============================================================

# 15. MULTI-QUERY EMBEDDINGS

# ============================================================

multi_query_embeddings = model.encode(multi_queries).tolist()

# ============================================================

# 16. MULTI-QUERY SEMANTIC RETRIEVAL

# ============================================================

multi_semantic_results = {}

for query_number, query_embedding in enumerate(

    multi_query_embeddings,

    start=1

):

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=5

    )

    multi_semantic_results[query_number] = results

    print(f"\n===== SEMANTIC RESULTS FOR QUERY {query_number} =====")

    print(multi_queries[query_number - 1])

    for rank in range(len(results["ids"][0])):

        print(f"\nRank: {rank + 1}")

        print(f"ID: {results['ids'][0][rank]}")

        print(f"Distance: {results['distances'][0][rank]:.4f}")

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

    for i in range(len(results["ids"][0])):

        distance = results["distances"][0][i]

        # Keep the retrieved top-k candidates for RRF. Chroma distances are
        # model-specific, so a fixed cutoff can discard useful evidence.
        filtered_indices.append(i)

    filtered_semantic_results[query_number] = filtered_indices

    print(f"\n===== THRESHOLD RESULTS FOR QUERY {query_number} =====")

    for i in filtered_indices:

        print(

            f"ID: {results['ids'][0][i]} | "

            f"Distance: {results['distances'][0][i]:.4f} | "

            f"Page: {results['metadatas'][0][i]['page_number']} | "

            f"Source: {results['metadatas'][0][i]['source']}"

        )

    print(f"Total results after threshold: {len(filtered_indices)}")

# ============================================================

# 18. MULTI-QUERY HYBRID RRF FUSION

# ============================================================

k = 60

final_rrf_scores = {}

# ============================================================

# 18.1 BM25 RESULTS FROM ALL QUERIES

# ============================================================

for query_number, data in multi_bm25_results.items():

    for rank, index in enumerate(data["indices"], start=1):

        chunk_id = f"chunk_{index + 1}"

        score = 1 / (k + rank)

        final_rrf_scores[chunk_id] = (

            final_rrf_scores.get(chunk_id, 0) + score

        )

# ============================================================

# 18.2 SEMANTIC RESULTS FROM ALL QUERIES

# USING THRESHOLD-FILTERED RESULTS

# ============================================================

for query_number, results in multi_semantic_results.items():

    filtered_indices = filtered_semantic_results[query_number]

    for rank, i in enumerate(filtered_indices, start=1):

        chunk_id = results["ids"][0][i]

        score = 1 / (k + rank)

        final_rrf_scores[chunk_id] = (

            final_rrf_scores.get(chunk_id, 0) + score

        )

# ============================================================

# 18.3 SORT FINAL RRF RESULTS

# ============================================================

final_hybrid_results = sorted(

    final_rrf_scores.items(),

    key=lambda x: x[1],

    reverse=True

)

print("\n===== MULTI-QUERY HYBRID RRF RESULTS =====")

for rank, (chunk_id, score) in enumerate(final_hybrid_results, start=1):

    index = int(chunk_id.split("_")[1]) - 1

    print(f"\nRank: {rank}")

    print(f"ID: {chunk_id}")

    print(f"RRF Score: {score:.6f}")

    print(f"Page: {chunk_metadata[index]['page_number']}")

    print(f"Source: {chunk_metadata[index]['source']}")

# ============================================================

# 19. CROSSENCODER RERANKING

# ============================================================

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ============================================================

# 20. HYBRID RESULTS → ACTUAL CHUNKS

# ============================================================

candidate_chunks = []

for chunk_id, rrf_score in final_hybrid_results:

    index = int(chunk_id.split("_")[1]) - 1

    document = all_chunks[index]

    metadata = chunk_metadata[index]

    candidate_chunks.append(

        (chunk_id, document, rrf_score, metadata)

    )

# ============================================================

# 21. CROSSENCODER: ALL MULTI-QUERIES × CANDIDATES

# ============================================================

pairs = []

pair_information = []

for chunk_id, document, rrf_score, metadata in candidate_chunks:

    for query_number, query in enumerate(multi_queries, start=1):

        pairs.append((query, document))

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

scores = reranker.predict(pairs)

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

print("\n===== MULTI-QUERY CROSSENCODER RESULTS =====")

for rank, (chunk_id, data) in enumerate(reranked, start=1):

    print(f"\nRank: {rank}")

    print(f"ID: {chunk_id}")

    print(f"CrossEncoder Score: {data['score']:.4f}")

    print(f"RRF Score: {data['rrf_score']:.6f}")

    print(f"Best Matching Query: Query {data['query_number']}")

    print(f"Page: {data['metadata']['page_number']}")

    print(f"Source: {data['metadata']['source']}")

    print(f"Text: {data['document']}")

# CrossEncoder scores are ranking scores, not probabilities. Avoid a hard
# numeric cutoff; keep the best-ranked candidates and let compression select evidence.
context_top_k = 6
filtered_reranked = reranked[:context_top_k]

print("\n===== CONTEXT FILTERING =====")

for chunk_id, data in filtered_reranked:

    print(

        f"ID: {chunk_id} | "

        f"CrossEncoder Score: {data['score']:.4f}"

    )

print(

    "Total chunks after context filtering: "

    f"{len(filtered_reranked)}"

)

def compress_chunk(chunk, query, threshold=0.35, top_n=5):
    """Select complete, relevant sentences without returning broken fragments."""
    sentences = split_into_sentences(chunk)
    if not sentences:
        return ""

    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        sentence_embeddings
    )[0]

    sentence_scores = list(enumerate(zip(sentences, similarities)))
    sentence_scores.sort(key=lambda item: item[1][1], reverse=True)

    # Always retain the best sentence; similarity thresholds vary by embedding model.
    selected_indices = [
        index
        for index, (sentence, score) in sentence_scores[:top_n]
    ]

    # Restore original source order for readable, faithful context.
    selected_indices.sort()
    return " ".join(sentences[index] for index in selected_indices)


# ============================================================

# 25.5. CONTEXT COMPRESSION

# ============================================================

compressed_chunks = []

for chunk_id, data in filtered_reranked:

    compressed_text = compress_chunk(

        data["document"],

        original_query,

        threshold=0.35

    )

    if compressed_text:

        compressed_chunks.append(

            (chunk_id, data, compressed_text)

        )

print("\n===== CONTEXT COMPRESSION =====")

for chunk_id, data, compressed_text in compressed_chunks:

    print(f"\nID: {chunk_id}")

    print(f"Original Text:\n{data['document']}")

    print(f"\nCompressed Text:\n{compressed_text}")

# ============================================================

# 26. FINAL CONTEXT

# ============================================================

top_n = 6

context = "\n\n".join(

    f"[Source: {data['metadata']['source']} | "

    f"Page: {data['metadata']['page_number']}]\n"

    f"{compressed_text}"

    for chunk_id, data, compressed_text in compressed_chunks[:top_n]

)

if not context.strip():
    raise ValueError(
        "No usable context was retrieved. Check the PDFs, query, and retrieval results."
    )

print("\n===== FINAL CONTEXT =====")

print(context)

# Collect sources available in the final context

available_sources = {

    (

        data["metadata"]["source"],

        data["metadata"]["page_number"]

    )

    for chunk_id, data, compressed_text in compressed_chunks[:top_n]

}

print("\n===== AVAILABLE CITATION SOURCES =====")

for source, page_number in sorted(available_sources):

    print(f"[Source: {source}, Page: {page_number}]")

# ============================================================

# 27. FINAL GROQ ANSWER

# ============================================================

answer_prompt = f"""

Answer the user's original question using only

the provided context.

If the answer is not present in the context, say:

"I don't have enough information in the provided context."

For every factual claim, include an inline citation using

the exact source filename and page number provided in context.

Use this citation format:

[Source: filename.pdf, Page: X]

Do not invent source filenames or page numbers.

Do not combine multiple factual claims into one sentence

unless each claim has appropriate supporting citations.

Every factual sentence must have at least one citation.

Use only sources present in the provided context.

If a claim cannot be supported by the context, say that

the provided context is insufficient.

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

response = generate_with_retry(

    model=GROQ_MODEL,

    contents=answer_prompt

)

# ============================================================

# 28. FINAL ANSWER

# ============================================================

print("\n===== FINAL ANSWER =====")

print(response.choices[0].message.content or "")

answer = response.choices[0].message.content or ""

# ============================================================

# 29. CITATION VALIDATION

# ============================================================

# Extract citations from the generated answer.

citations = re.findall(

    r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]",

    answer

)

print("\n===== CITATION VALIDATION =====")

if not citations:

    print("WARNING: No citations found in the answer.")

else:

    for source, page_number in citations:

        citation = (source.strip(), int(page_number))

        if citation in available_sources:

            print(

                f"VALID: {source}, Page {page_number}"

            )

        else:

            print(

                f"INVALID: {source}, Page {page_number}"

            )

# ============================================================

# 30. INVALID CITATION TEST

# ============================================================

# Temporary test: check a source that was not retrieved.

test_citation = (

    "Deep_Learning_Fundamentals.pdf",

    99

)

print("\n===== INVALID CITATION TEST =====")

if test_citation in available_sources:

    print("VALID")

else:

    print("INVALID: Source/page not present in final context")

# ============================================================

# 31. SENTENCE-LEVEL CITATION COVERAGE

# ============================================================

# Check sentence-level citation coverage.

answer = response.choices[0].message.content or ""

sentences = [

    sentence.strip()

    for sentence in re.split(r"(?<=[.!?])\s+", answer)

    if sentence.strip()

]

# Ignore headings and standalone source references.

factual_sentences = [

    sentence

    for sentence in sentences

    if not sentence.startswith("[Source:")

    and not sentence.startswith("#")

]

cited_sentences = [

    sentence

    for sentence in factual_sentences

    if re.search(

        r"\[Source:\s*.*?,\s*Page:\s*\d+\]",

        sentence

    )

]

total = len(factual_sentences)

cited = len(cited_sentences)

print("\n===== SENTENCE-LEVEL CITATION COVERAGE =====")

print(f"Factual sentences: {total}")

print(f"Sentences with citations: {cited}")

if total > 0:

    coverage = (cited / total) * 100

    print(f"Citation coverage: {coverage:.1f}%")

else:

    print("Citation coverage could not be calculated.")

# ============================================================

# DAY 12 - STEP 2: EXTRACT CLAIMS

# ============================================================

print("\n===== CLAIM EXTRACTION =====")

claims = [

    sentence.strip()

    for sentence in re.split(r"(?<=[.!?])\s+", answer)

    if sentence.strip()

    and not sentence.startswith("[Source:")

    and not sentence.startswith("#")

]

for i, claim in enumerate(claims, start=1):

    print(f"Claim {i}: {claim}")

print(f"\nTotal claims extracted: {len(claims)}")

# ============================================================

# DAY 12 - STEP 3: SEPARATE CLAIMS AND CITATIONS

# ============================================================

print("\n===== CLAIMS AND CITATIONS =====")

claim_records = []

for sentence in claims:

    sentence_citations = re.findall(

        r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]",

        sentence

    )

    claim_text = re.sub(

        r"\[Source:\s*.*?,\s*Page:\s*\d+\]",

        "",

        sentence

    ).strip()

    if claim_text:

        claim_records.append({

            "claim": claim_text,

            "citations": [

                (source.strip(), int(page))

                for source, page in sentence_citations

            ]

        })

for i, record in enumerate(claim_records, start=1):

    print(f"\nClaim {i}: {record['claim']}")

    print(f"Citations: {record['citations']}")

print(f"\nTotal claim records: {len(claim_records)}")

# ============================================================

# DAY 12 - STEP 4: CLAIM-SOURCE SUPPORT VERIFICATION

# ============================================================

supported_count = 0

not_supported_count = 0

uncertain_count = 0

print("\n===== CLAIM-SOURCE SUPPORT VERIFICATION =====")

for i, record in enumerate(claim_records, start=1):

    print(f"\nClaim {i}: {record['claim']}")

    if not record["citations"]:

        print("Result: NOT_SUPPORTED")

        print("Reason: No citation provided.")

        not_supported_count += 1

        continue

    for source, page_number in record["citations"]:

        # Find the exact cited page text.

        page_text = next(

            (

                page["text"]

                for page in pages

                if page["source"] == source

                and page["page_number"] == page_number

            ),

            None

        )

        if not page_text:

            print(

                f"Result: NOT_SUPPORTED — "

                f"{source}, Page {page_number}"

            )

            not_supported_count += 1

            continue

        verification_prompt = f"""

You are a claim-source verification evaluator.

Determine whether the source text supports the claim.

Return exactly one label:

SUPPORTED

NOT_SUPPORTED

UNCERTAIN

Claim:

{record['claim']}

Source filename: {source}

Page number: {page_number}

Source text:

{page_text}

Judge only from the provided source text.

Do not use outside knowledge.

Return only the label.

"""

        # Groq replaces the previous Gemini API call.

        verification_response = generate_with_retry(

            model=GROQ_MODEL,

            contents=verification_prompt

        )

        result = (

            verification_response.choices[0].message.content or ""

        ).strip().upper()

        if "NOT_SUPPORTED" in result:

            label = "NOT_SUPPORTED"

        elif "SUPPORTED" in result:

            label = "SUPPORTED"

        else:

            label = "UNCERTAIN"

        print(f"Source: {source}, Page {page_number}")

        if label == "SUPPORTED":

            supported_count += 1

        elif label == "NOT_SUPPORTED":

            not_supported_count += 1

        else:

            uncertain_count += 1

        print(f"Result: {label}")

# ============================================================

# DAY 12 - STEP 6: FINAL VERIFICATION SUMMARY

# ============================================================

print("\n===== FINAL VERIFICATION SUMMARY =====")

total_checks = (

    supported_count

    + not_supported_count

    + uncertain_count

)

print(f"Total verification checks: {total_checks}")

print(f"Supported: {supported_count}")

print(f"Not Supported: {not_supported_count}")

print(f"Uncertain: {uncertain_count}")

if total_checks > 0:

    support_rate = (supported_count / total_checks) * 100

    print(f"Support rate: {support_rate:.1f}%")

else:

    print("No verification checks were completed.")

# ============================================================

# DAY 13 - STEP 3: FAITHFULNESS EVALUATION

# ============================================================

print("\n===== FAITHFULNESS EVALUATION =====")

faithfulness_prompt = f"""

You are evaluating the faithfulness of an AI-generated answer.

Compare the answer only against the provided context.

Classify each factual claim as:

- SUPPORTED: The context supports the claim.

- NOT_SUPPORTED: The context does not support the claim.

- CONTRADICTED: The context directly contradicts the claim.

Context:

{context}

Generated Answer:

{answer}

For each factual claim, provide:

1. The claim

2. One of the three labels

3. A brief reason based only on the context

Do not use outside knowledge.

"""

faithfulness_response = generate_with_retry(

    model=GROQ_MODEL,

    contents=faithfulness_prompt

)

faithfulness_result = (

    faithfulness_response.choices[0].message.content or ""

)

print(faithfulness_result)

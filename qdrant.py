import os
import uuid
import hashlib
import json
from typing import List, Dict, Optional

from qdrant_client import QdrantClient as QC, models
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "data")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- GLOBALS (initialized by initialize()) ---
client: Optional[QC] = None
local_embedder: Optional[SentenceTransformer] = None
generation_model: Optional[genai.GenerativeModel] = None
bm25_indices: Dict[str, BM25Okapi] = {}
book_corpus: Dict[str, List[Dict]] = {}


def initialize():
    """
    Initialize models and Qdrant client. Called once at application startup.
    """
    global client, local_embedder, generation_model

    print("Initializing Qdrant client and models...")

    local_embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize Gemini / generative model (if API key/config available)
    GENERATION_MODEL_NAME = "models/gemini-flash-latest"
    try:
        generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
    except Exception as e:
        print(f"Warning: failed to initialize generation_model: {e}")
        generation_model = None  # type: ignore

    client = QC(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )

    # Ensure collection exists and create payload indexes if new
    model_embedding_dimension = local_embedder.get_sentence_embedding_dimension()
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' does not exist. Attempting to create it...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=model_embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Collection '{COLLECTION_NAME}' created successfully.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")

        for field in ["class_name", "subject", "chapter", "textbook_uuid", "chpstpage", "chpendpage"]: # Added chpstpage, chpendpage
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                print(f"Payload index for '{field}' created/verified.")
            except Exception as e:
                print(f"Warning: Failed to create payload index for field '{field}': {e}")
    except Exception as e:
        print(f"CRITICAL ERROR during Qdrant initialization: {e}")
        raise # Re-raise to prevent app from running with broken Qdrant connection

    print("Qdrant client and models initialized.")


# --- Helper functions ---
def get_book_uuid(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def check_if_book_exists(book_uuid: str) -> bool:
    if not client:
        raise RuntimeError("Qdrant client not initialized.")
    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
        ),
        limit=1,
    )
    return len(response) > 0


def _get_all_chunks_for_book(book_uuid: str) -> List[Dict]:
    """
    Scrolls through all points in the collection for a given textbook_uuid and
    returns a list of payload dicts.
    """
    if not client:
        raise RuntimeError("Qdrant client not initialized.")

    all_points = []
    next_offset = None

    while True:
        response, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
            ),
            limit=250,
            with_payload=True,
            offset=next_offset,
        )
        all_points.extend(response)
        if not next_offset:
            break

    return [point.payload for point in all_points]


def get_or_build_bm25_index(book_uuid: str) -> Optional[BM25Okapi]:
    """
    Returns a cached BM25 index for the book, or builds it if missing.
    Also caches the book corpus (payloads) for BM25 lookups.
    """
    if book_uuid in bm25_indices:
        return bm25_indices[book_uuid]

    print(f"Building BM25 index for book: {book_uuid}")
    corpus_docs = _get_all_chunks_for_book(book_uuid)
    if not corpus_docs:
        return None

    book_corpus[book_uuid] = corpus_docs
    tokenized_corpus = [doc.get("text", "").split(" ") for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_indices[book_uuid] = bm25
    return bm25


# --- Core logic ---
def process_and_embed_book(pdf_path: str, class_name: str, subject: str, chapters: List[Dict]):
    """
    Read PDF, split into chunks using the provided chapter ranges, embed with
    local_embedder, and upload points to Qdrant. If the same book (sha256)
    already exists, delete previous points first.
    """
    if not client or not local_embedder:
        raise RuntimeError("Client or embedder not initialized. Call initialize() first.")

    print(f"\n--- Starting Book Processing ---")
    print(f"File: {os.path.basename(pdf_path)}")
    print(f"Class: {class_name}, Subject: {subject}")

    book_uuid = get_book_uuid(pdf_path)

    # Delete existing points for this book_uuid (if any)
    if check_if_book_exists(book_uuid):
        print(f"Book with UUID {book_uuid} already exists. Deleting old entries...")
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
                )
            ),
        )
        print("Old entries deleted. Proceeding with re-embedding...")

    reader = PdfReader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    total_chapters = len(chapters)
    print(f"Found {total_chapters} chapters to process.")

    for i, chapter in enumerate(chapters):
        chapter_name = chapter.get("chapter_name") or chapter.get("name", f"Untitled Chapter {i+1}")
        
        # Use the specific page numbers from the chapter object
        pdf_start_page_llm = chapter.get("pdf_startpg")
        pdf_end_page_llm = chapter.get("pdf_endpg")
        chp_start_page = chapter.get("chpstpage")
        chp_end_page = chapter.get("chpendpage")

        if pdf_start_page_llm is None or pdf_end_page_llm is None:
            print(f"  Skipping chapter '{chapter_name}' because pdf_startpg/pdf_endpg missing.")
            continue

        chapter_text = ""
        # Iterate using the LLM's identified PDF page numbers
        for page_num in range(pdf_start_page_llm - 1, pdf_end_page_llm):
            if 0 <= page_num < len(reader.pages):
                chapter_text += reader.pages[page_num].extract_text() or ""

        print(f"  --- Chapter Text for '{chapter_name}' (PDF Pages {pdf_start_page_llm}-{pdf_end_page_llm}) ---")
        print(f"  First 200 chars: {chapter_text[:200].strip()}...")
        print(f"  Last 200 chars: ...{chapter_text[-200:].strip() if len(chapter_text) >= 200 else chapter_text.strip()}")
        print(f"  Total length: {len(chapter_text)} characters")
        print("  --------------------------------------------------")

        text_chunks = text_splitter.split_text(chapter_text)
        print(f"Split chapter into {len(text_chunks)} text chunks.")

        points_to_upload = []
        for chunk in text_chunks:
            embedding = local_embedder.encode(chunk).tolist()
            points_to_upload.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "class_name": class_name,
                        "subject": subject,
                        "textbook_uuid": book_uuid,
                        "filename": os.path.basename(pdf_path),
                        "chapter": chapter_name,
                        "pdf_startpg": pdf_start_page_llm,
                        "pdf_endpg": pdf_end_page_llm,
                        "chpstpage": chp_start_page, # Add this
                        "chpendpage": chp_end_page,   # Add this
                        "text": chunk,
                    },
                )
            )

        if points_to_upload:
            print(f"Embedding complete. Uploading {len(points_to_upload)} points to Qdrant...")
            client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)
            print(f"Chapter '{chapter_name}' processed and saved successfully.")

    print("\n--- Book Processing Complete ---")
    return


def get_books(class_name: Optional[str] = None, subject: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Returns a list of unique books (by textbook_uuid) optionally filtered by class_name
    and case-insensitively filtered by subject.
    """
    if not client:
        raise RuntimeError("Qdrant client not initialized.")

    filter_conditions = []
    if class_name:
        filter_conditions.append(models.FieldCondition(key="class_name", match=models.MatchValue(value=class_name)))

    scroll_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=1000,
        with_payload=["textbook_uuid", "subject", "class_name", "filename"],
    )

    unique_books: Dict[str, Dict[str, str]] = {}
    for p in response:
        book_uuid = p.payload.get("textbook_uuid")
        payload_subject = p.payload.get("subject")

        if subject and payload_subject and subject.lower() != payload_subject.lower():
            continue

        if book_uuid and book_uuid not in unique_books:
            unique_books[book_uuid] = {
                "id": book_uuid,
                "subject": payload_subject,
                "class_name": p.payload.get("class_name", "N/A"),
                "filename": p.payload.get("filename"),
            }

    return list(unique_books.values())


def get_book_metadata(book_uuid: str) -> Dict[str, Optional[str]]:
    if not client:
        raise RuntimeError("Qdrant client not initialized.")

    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
        ),
        limit=1,
        with_payload=["class_name", "subject"],
    )

    if response:
        payload = response[0].payload
        return {"class_name": payload.get("class_name"), "subject": payload.get("subject")}
    return {}


def get_chapter_names(book_uuid: str) -> List[str]:
    """
    Return sorted unique chapter names for a given book_uuid.
    """
    if not client:
        raise RuntimeError("Qdrant client not initialized.")

    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
        ),
        limit=1000,
        with_payload=["chapter"],
    )

    unique_names = set()
    for point in response:
        name = point.payload.get("chapter")
        if name:
            unique_names.add(name)

    return sorted(list(unique_names))


def get_chapters_for_book(book_uuid: str) -> List[Dict]:
    """
    For each chapter name return a dict with name and page ranges.
    Uses payload keys that process_and_embed_book writes: pdf_startpg/pdf_endpg, chap_startpg/chap_endpg
    """
    if not client:
        raise RuntimeError("Qdrant client not initialized.")

    chapter_names = get_chapter_names(book_uuid)
    if not chapter_names:
        return []

    chapter_info = []
    for name in chapter_names:
        response, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid)),
                    models.FieldCondition(key="chapter", match=models.MatchValue(value=name)),
                ]
            ),
            limit=1,
            with_payload=["pdf_startpg", "pdf_endpg", "chpstpage", "chpendpage"], # Fetch these
        )

        pdf_start_page = pdf_end_page = None
        chp_start_page = chp_end_page = None
        if response:
            payload = response[0].payload
            pdf_start_page = payload.get("pdf_startpg")
            pdf_end_page = payload.get("pdf_endpg")
            chp_start_page = payload.get("chpstpage") # Get chpstpage
            chp_end_page = payload.get("chpendpage")   # Get chpendpage

        chapter_info.append(
            {
                "chapter_name": name, # Renamed 'name' to 'chapter_name' for consistency with frontend
                "pdf_startpg": pdf_start_page,
                "pdf_endpg": pdf_end_page,
                "chpstpage": chp_start_page, # Add chpstpage
                "chpendpage": chp_end_page,   # Add chpendpage
            }
        )

    # The sort key needs to be updated as well, as 'chap_startpg' is now 'chpstpage'
    chapter_info.sort(key=lambda x: (x.get("chpstpage") or 0))
    return chapter_info


def hybrid_search(book_uuid: str, query: str, keywords: List[Dict], conceptual_score: float, metadata_filters: Optional[Dict] = None):
    """
    Perform hybrid search: semantic (Qdrant) + BM25 keyword, return top results.
    Returns (ranked_list[:10], semantic_results, normalized_bm25_results).
    """
    if not local_embedder:
        raise RuntimeError("Local embedder not initialized.")

    alpha = 0.4 + (conceptual_score * 0.2)
    keyword_query_str = " ".join([item["keyword"] for item in keywords])

    # Semantic search
    must_conditions = [models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
    if metadata_filters:
        for key, value in metadata_filters.items():
            must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    query_embedding = local_embedder.encode(query).tolist()
    semantic_results = []
    try:
        query_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=models.Filter(must=must_conditions),
            limit=10,
            with_payload=True,
        )
        semantic_results = query_response.points
    except Exception:
        semantic_results = []

    # BM25 keyword search
    bm25 = get_or_build_bm25_index(book_uuid)
    normalized_bm25_results = []
    if bm25:
        corpus_docs = book_corpus.get(book_uuid, [])
        tokenized_query = keyword_query_str.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)

        sparse_results_with_scores = []
        for i, doc in enumerate(corpus_docs):
            if metadata_filters and "chapter" in metadata_filters:
                if doc.get("chapter") != metadata_filters["chapter"]:
                    continue
            sparse_results_with_scores.append((bm25_scores[i], doc))

        sparse_results_with_scores.sort(key=lambda x: x[0], reverse=True)
        top_10_sparse = [res for res in sparse_results_with_scores if res[0] > 0][:10]

        if top_10_sparse:
            scores = [score for score, _doc in top_10_sparse]
            min_s, max_s = min(scores), max(scores)
            for score, doc in top_10_sparse:
                norm_score = (score - min_s) / (max_s - min_s) if max_s > min_s else 1.0
                normalized_bm25_results.append((norm_score, doc))

    # Combine results and compute hybrid score
    hybrid_candidates: Dict[str, Dict] = {}
    for res in semantic_results:
        doc_text = res.payload.get("text", "").strip()
        if doc_text not in hybrid_candidates:
            hybrid_candidates[doc_text] = {"semantic": 0, "bm25": 0, "doc": res.payload}
        hybrid_candidates[doc_text]["semantic"] = res.score

    for score, doc in normalized_bm25_results:
        doc_text = doc.get("text", "").strip()
        if doc_text not in hybrid_candidates:
            hybrid_candidates[doc_text] = {"semantic": 0, "bm25": 0, "doc": doc}
        hybrid_candidates[doc_text]["bm25"] = score

    ranked_list = []
    for doc_text, scores in hybrid_candidates.items():
        hybrid_score = alpha * (scores.get("semantic") or 0) + (1 - alpha) * (scores.get("bm25") or 0)
        ranked_list.append((hybrid_score, scores["doc"]))

    ranked_list.sort(key=lambda x: x[0], reverse=True)
    return ranked_list[:10], semantic_results, normalized_bm25_results


def perform_retrieval(raw_query: str, selected_book: Dict):
    """
    Reformulate query via LLM, run semantic + BM25 retrieval, write results to result.txt
    """
    if not local_embedder:
        raise RuntimeError("Local embedder not initialized.")

    book_uuid = selected_book["id"]
    output_filename = "result.txt"

    print("\nProcessing query with Gemini model...")
    processed_data = reformulate_and_classify_query(raw_query)
    reformulated_query = processed_data.get("reformulated_query", raw_query)
    keywords = processed_data.get("keywords", [])
    conceptual_score = processed_data.get("conceptual_score", 0.0)

    alpha = 0.4 + (conceptual_score * 0.2)
    keyword_list = [item["keyword"] for item in keywords]
    keyword_query_str = " ".join(keyword_list)
    keyword_details = ", ".join([f"{item['keyword']} (Score: {item['importance']:.2f})" for item in keywords])

    print("--- Query Details ---")
    print(f"Original Query: {raw_query}")
    print(f"Conceptual Score: {conceptual_score:.2f} (Alpha for Hybrid Search: {alpha:.2f})")
    print(f"Reformulated Semantic Query: {reformulated_query}")
    print(f"Extracted Keywords for BM25: {keyword_details}")
    print("---------------------")

    semantic_results = []
    normalized_bm25_results = []

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Original Query: {raw_query}\n")
        f.write(f"Conceptual Score: {conceptual_score:.2f} (Alpha: {alpha:.2f})\n")
        f.write(f"Reformulated Semantic Query: {reformulated_query}\n")
        f.write(f"Extracted Keywords: {keyword_details}\n")
        f.write(
            f"Book: Class {selected_book.get('class_name', 'N/A')}, Subject {selected_book.get('subject', 'N/A')}\n"
        )
        f.write("=" * 40 + "\n\n")

        # Semantic search
        query_embedding = local_embedder.encode(reformulated_query).tolist()
        try:
            query_response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
                ),
                limit=10,
                with_payload=True,
            )
            semantic_results = query_response.points

            if not semantic_results:
                f.write("No semantic results found.\n")
            else:
                for i, res in enumerate(semantic_results):
                    f.write(f"  {i+1}. Score: {res.score:.4f} | Text: {res.payload.get('text','').strip()}\n\n")
        except Exception as e:
            f.write(f"An error occurred during semantic search: {e}\n")

        # BM25 search
        corpus_docs = _get_all_chunks_for_book(book_uuid)
        if not corpus_docs:
            f.write("Could not retrieve document corpus for the selected book.\n")
        else:
            tokenized_corpus = [doc.get("text", "").split(" ") for doc in corpus_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = keyword_query_str.split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)

            sparse_results_with_scores = sorted(
                [(bm25_scores[i], doc) for i, doc in enumerate(corpus_docs)], key=lambda x: x[0], reverse=True
            )

            top_10_sparse = [res for res in sparse_results_with_scores if res[0] > 0][:10]

            if not top_10_sparse:
                f.write("No keyword matches found with a score > 0.\n")
            else:
                scores = [score for score, _doc in top_10_sparse]
                min_s, max_s = min(scores), max(scores)
                for score, doc in top_10_sparse:
                    if max_s > min_s:
                        norm_score = (score - min_s) / (max_s - min_s)
                    else:
                        norm_score = 1.0
                    normalized_bm25_results.append((norm_score, doc))

                for i, (score, doc) in enumerate(normalized_bm25_results):
                    f.write(f"  {i+1}. Normalized Score: {score:.4f} | Text: {doc.get('text','').strip()}\n\n")

        # Hybrid ranking
        f.write("\n--- 3. Top 10 Hybrid Search Results ---\n\n")
        hybrid_candidates = {}
        for res in semantic_results:
            doc_text = res.payload.get("text", "").strip()
            if doc_text not in hybrid_candidates:
                hybrid_candidates[doc_text] = {"semantic": 0, "bm25": 0, "doc": res.payload}
            hybrid_candidates[doc_text]["semantic"] = res.score

        for score, doc in normalized_bm25_results:
            doc_text = doc.get("text", "").strip()
            if doc_text not in hybrid_candidates:
                hybrid_candidates[doc_text] = {"semantic": 0, "bm25": 0, "doc": doc}
            hybrid_candidates[doc_text]["bm25"] = score

        ranked_list = []
        for doc_text, scores in hybrid_candidates.items():
            hybrid_score = alpha * (scores.get("semantic") or 0) + (1 - alpha) * (scores.get("bm25") or 0)
            ranked_list.append((hybrid_score, scores["doc"]))

        ranked_list.sort(key=lambda x: x[0], reverse=True)

        if not ranked_list:
            f.write("No results to rank.\n")
        else:
            for i, (score, doc) in enumerate(ranked_list[:10]):
                f.write(f"  {i+1}. Hybrid Score: {score:.4f} | Text: {doc.get('text','').strip()}\n\n")

        # 4. Format top chunks for LLM
        f.write("\n--- 4. Formatted Top 10 Chunks for LLM ---\n\n")
        if not ranked_list:
            f.write("No chunks to format.\n")
        else:
            for i, (score, doc) in enumerate(ranked_list[:10]):
                chunk_header = f"--- Chunk {i+1} ---"
                chunk_body = doc.get("text", "").strip()
                f.write(f"{chunk_header}\n{chunk_body}\n\n")


def reformulate_and_classify_query(query: str, class_name: Optional[str] = None, subject: Optional[str] = None, chapter_list: Optional[List] = None) -> Dict:
    """
    Use the generative model to reformulate the query, extract keywords and
    return a conceptual_score. Returns a dict:
    {
      "reformulated_query": str,
      "keywords": [{"keyword": str, "importance": float}, ...],
      "conceptual_score": float,
      "classification": "conceptual"|"factual"
    }
    """
    raw_query = query

    prompt = (
        "You are a search query processing expert. For the given user query, perform the following tasks:\n\n"
        "1. Reformulate the Query: Make it more descriptive and contextually complete for use in a semantic vector search.\n\n"
        "2. Extract Important Keywords: Identify the most relevant keywords or short key phrases from the query. "
        "For each keyword, assign a relevance score between 0 and 1. Include only keywords with importance >= 0.3.\n\n"
        "3. Classify Query Type: Determine whether the query is more conceptual or factual. Provide a 'conceptual_score' between 0 and 1.\n\n"
        f"Return a single valid JSON object with keys: reformulated_query, keywords (array of {{keyword, importance}}), conceptual_score.\n\n"
        f"User Query: \"{raw_query}\"\n\n"
        "Example output:\n"
        '{"reformulated_query":"Detailed...","keywords":[{"keyword":"photosynthesis","importance":0.95}],"conceptual_score":0.85}\n'
    )

    if not generation_model:
        # Fallback: simple deterministic extraction if no model available
        return {
            "reformulated_query": raw_query,
            "keywords": [],
            "conceptual_score": 0.5,
            "classification": "conceptual",
        }

    try:
        response = generation_model.generate_content(prompt)
        json_text = response.text.strip()
        json_start = json_text.find("{")
        json_end = json_text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            clean_json = json_text[json_start:json_end]
            parsed_json = json.loads(clean_json)
            conceptual_score = parsed_json.get("conceptual_score", 0.5)
            parsed_json["classification"] = "conceptual" if conceptual_score > 0.5 else "factual"
            return parsed_json
        else:
            return {
                "reformulated_query": raw_query,
                "keywords": [],
                "conceptual_score": 0.5,
                "classification": "conceptual",
            }
    except Exception as e:
        print(f"Error during query reformulation/classification: {e}")
        return {
            "reformulated_query": raw_query,
            "keywords": [],
            "conceptual_score": 0.5,
            "classification": "conceptual",
        }


def generate_answer(raw_query: str, book_details: Dict, context: str) -> str:
    """
    Use the generative model (Gemini) with a teacher-system prompt to answer the query.
    """
    if not generation_model:
        raise RuntimeError("Generation model not initialized.")

    system_prompt = (
        "# System Prompt for CHADUVU-GURU\n\n"
        "You are an AI teacher assistant. Your role is to answer student queries in a way a real teacher would, "
        "explaining concepts clearly, using examples if needed, and staying within the context of the textbook provided. "
        "Follow these rules:\n\n"
        "1. Teacher Role: Assume you are the teacher for the given class and subject.\n"
        "2. Context Usage: Use the provided textbook chunks as the primary reference. "
        "Do not include outside information unless necessary to explain a concept.\n"
        "3. Answer Style: Explain clearly and concisely. Provide examples or analogies if useful.\n"
        "4. Always Generate an Answer: Even if similarity is low, produce the best possible answer.\n"
        "5. Formatting: Use readable paragraphs, bullet points or numbered lists as appropriate.\n"
    )

    user_prompt = (
        f"**Class:** {book_details.get('class_name', 'N/A')}\n"
        f"**Subject:** {book_details.get('subject', 'N/A')}\n\n"
        f"**Student's Query:** \"{raw_query}\"\n\n"
        f"**Textbook Context:**\n{context}\n"
    )

    response = generation_model.generate_content([system_prompt, user_prompt])
    return response.text


def generate_chapters_from_json(pdf_json: List[Dict]) -> str:
    """
    Build a single prompt string containing the JSON page list for the LLM to extract chapters.
    Returns the prompt (a plain string).
    """
    json_text = json.dumps(pdf_json)
    prompt = (
        "You are an expert assistant tasked with analyzing a textbook to identify its chapters.\n\n"
        "The book content is provided as a JSON array, each element representing a PDF page:\n\n"
        '[{"pdf_page": <integer>, "text": "<page text>"}]\n\n'
        "When identifying chapters and their page numbers, prioritize information found in an 'INDEX' or 'Table of Contents' section if available within the provided text.\n\n"
        "Return a single valid JSON object following this schema:\n\n"
        '{\n'
        '  "pdf_offset": <integer>,\n'
        '  "chapters": [\n'
        '    {"chapter_name": "Full name of the chapter", "pdf_startpg": <integer>, "pdf_endpg": <integer>}\n'
        "  ]\n"
        "}\n\n"
        "- pdf_startpg/pdf_endpg are the real PDF page numbers (including front matter).\n"
        "- Calculate `pdf_offset` as the number of pages of front matter. This is typically (first_chapter_start_page - 1). If an index is available, infer the front matter by the difference between the page number in the index and the actual pdf page number where the chapter starts.\n"
        "Do not include any text outside the JSON object.\n\n"
        "Here is the book content in JSON format:\n\n"
        f"{json_text}\n"
    )
    return prompt


def generate_chapters_from_text(json_path: str) -> str:
    """
    Read the page JSON file (json_path), construct prompt and ask the generative model to extract chapters.
    Returns a JSON-string representation of the parsed LLM output or a safe default.
    """
    if not generation_model:
        print("Warning: generation_model not initialized; returning empty chapters.")
        return json.dumps({"pdf_offset": 0, "chapters": []})

    with open(json_path, "r", encoding="utf-8") as f:
        pdf_pages_data = json.load(f)

    prompt = generate_chapters_from_json(pdf_pages_data)

    try:
        response = generation_model.generate_content(prompt)
        text = response.text.strip()
        print(f"Raw LLM response for chapter extraction: {text[:500]}...")

        json_start = text.find("{")
        if json_start == -1:
            print("LLM response did not contain a JSON object.")
            return json.dumps({"pdf_offset": 0, "chapters": []})

        open_braces = 0
        json_end = -1
        for i, char in enumerate(text[json_start:]):
            if char == "{":
                open_braces += 1
            elif char == "}":
                open_braces -= 1

            if open_braces == 0:
                json_end = json_start + i + 1
                break

        if json_end == -1:
            print("Could not find matching closing brace in LLM response.")
            return json.dumps({"pdf_offset": 0, "chapters": []})

        clean_json_str = text[json_start:json_end]
        try:
            data = json.loads(clean_json_str)
            # The LLM's pdf_startpg/endpg are taken as is.
            return json.dumps(data)
        except json.JSONDecodeError:
            print("Failed to parse JSON from LLM response.")
            return clean_json_str

    except Exception as e:
        print(f"Error during LLM chapter generation: {e}")
        return json.dumps({"pdf_offset": 0, "chapters": []})

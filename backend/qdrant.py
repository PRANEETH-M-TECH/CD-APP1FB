import os
import uuid
import hashlib
import json
from typing import List, Dict, Optional, Any

from qdrant_client import QdrantClient as QC, models
from backend.app.services.llm.openai_client import create_client, OPENAI_MODEL
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
generation_model: Optional[Any] = None
bm25_indices: Dict[str, BM25Okapi] = {}
book_corpus: Dict[str, List[Dict]] = {}


def initialize():
    """
    Initialize models and Qdrant client. Called once at application startup.
    """
    global client, local_embedder, generation_model

    local_embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize OpenAI Client Adapter
    try:
        generation_model = create_client().models
    except Exception as e:
        generation_model = None


    client = QC(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )

    # Ensure collection exists and create payload indexes if new
    model_embedding_dimension = local_embedder.get_sentence_embedding_dimension()
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=model_embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        for field in ["class_name", "subject", "chapter", "textbook_uuid", "chpstpage", "chpendpage"]: # Added chpstpage, chpendpage
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                pass # Silently fail
    except Exception as e:
        raise # Re-raise to prevent app from running with broken Qdrant connection


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
    Also, generates and saves a summary of the book.
    """
    if not client or not local_embedder:
        raise RuntimeError("Client or embedder not initialized. Call initialize() first.")

    # Create chpchunks directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    chpchunks_dir = os.path.join(project_root, "chpchunks")
    if not os.path.exists(chpchunks_dir):
        os.makedirs(chpchunks_dir)
        
    summary_dir = os.path.join(project_root, "summary")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    book_uuid = get_book_uuid(pdf_path)
    print(f"\n--- Starting processing for book: {os.path.basename(pdf_path)} (UUID: {book_uuid}) ---")

    # Delete existing points for this book_uuid (if any)
    if check_if_book_exists(book_uuid):
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
                )
            ),
        )

    reader = PdfReader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    book_data_for_json = {
        "class_name": class_name,
        "subject": subject,
        "chapters": []
    }
    
    summary_data_for_json = {
        "class_name": class_name,
        "subject_name": subject,
        "chapters": []
    }

    for i, chapter in enumerate(chapters):
        chapter_name = chapter.get("chapter_name") or chapter.get("name", f"Untitled Chapter {i+1}")
        
        pdf_start_page_llm = chapter.get("pdf_startpg")
        pdf_end_page_llm = chapter.get("pdf_endpg")
        chp_start_page = chapter.get("chpstpage")
        chp_end_page = chapter.get("chpendpage")

        if pdf_start_page_llm is None or pdf_end_page_llm is None:
            continue

        chapter_text = ""
        for page_num in range(pdf_start_page_llm - 1, pdf_end_page_llm):
            if 0 <= page_num < len(reader.pages):
                chapter_text += reader.pages[page_num].extract_text() or ""

        text_chunks = text_splitter.split_text(chapter_text)
        print(f"  - Processing chapter '{chapter_name}': {len(text_chunks)} raw chunks extracted.")

        # Store chunks for chpchunks.json
        chapter_data_for_json = {
            "chapter_name": chapter_name,
            "number_of_chunks": len(text_chunks),
            "chunks": text_chunks
        }
        book_data_for_json["chapters"].append(chapter_data_for_json)

        # Generate and store summary for summary.json
        print(f"  - Sending {len(text_chunks)} chunks for chapter '{chapter_name}' to LLM for summary generation...")
        summary_text = generate_chapter_summary(class_name, subject, chapter_name, text_chunks)
        summary_chapter_data = {
            "chapter_name": chapter_name,
            "summary": summary_text
        }
        summary_data_for_json["chapters"].append(summary_chapter_data)

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
                        "chpstpage": chp_start_page,
                        "chpendpage": chp_end_page,
                        "text": chunk,
                    },
                )
            )

        if points_to_upload:
            client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)

    # Write chpchunks JSON file
    json_filename_chunks = f"{subject.lower()}{class_name.replace(' ', '')}.json"
    json_filepath_chunks = os.path.join(chpchunks_dir, json_filename_chunks)
    print(f"--- Writing chapter chunks to {json_filepath_chunks} ---")
    try:
        with open(json_filepath_chunks, "w", encoding="utf-8") as f:
            json.dump(book_data_for_json, f, indent=2)
    except IOError as e:
        print(f"ERROR: Could not write chapter chunks file {json_filepath_chunks}: {e}")
        
    # Write summary JSON file
    json_filename_summary = f"{subject.lower()}{class_name.replace(' ', '')}.json"
    json_filepath_summary = os.path.join(summary_dir, json_filename_summary)
    print(f"--- Writing summary to {json_filepath_summary} ---")
    try:
        with open(json_filepath_summary, "w", encoding="utf-8") as f:
            json.dump(summary_data_for_json, f, indent=2)
    except IOError as e:
        print(f"ERROR: Could not write summary file {json_filepath_summary}: {e}")

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

    processed_data = reformulate_and_classify_query(
        raw_query,
        class_name=selected_book.get('class_name'),
        subject=selected_book.get('subject')
    )
    reformulated_query = processed_data.get("reformulated_query", raw_query)
    keywords = processed_data.get("keywords", [])
    conceptual_score = processed_data.get("conceptual_score", 0.0)
    classified_chapter = processed_data.get("classified_chapter", "N/A") # Get the new field

    alpha = 0.4 + (conceptual_score * 0.2)
    keyword_list = [item["keyword"] for item in keywords]
    keyword_query_str = " ".join(keyword_list)
    keyword_details = ", ".join([f"{item['keyword']} (Score: {item['importance']:.2f})" for item in keywords])

    semantic_results = []
    normalized_bm25_results = []

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Original Query: {raw_query}\n")
        f.write(f"Conceptual Score: {conceptual_score:.2f} (Alpha: {alpha:.2f})\n")
        f.write(f"Reformulated Semantic Query: {reformulated_query}\n")
        f.write(f"Extracted Keywords: {keyword_details}\n")
        f.write(f"Classified Chapter: {classified_chapter}\n") # Write the classified chapter
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
      "classification": "conceptual"|"factual",
      "classified_chapter": Optional[str] # New field
    }
    """
    raw_query = query
    summary_context = ""
    classified_chapter = None

    if class_name and subject:
        summary_dir = "summary"
        json_filename_summary = f"{subject.lower()}{class_name.replace(' ', '')}.json"
        json_filepath_summary = os.path.join(summary_dir, json_filename_summary)

        if os.path.exists(json_filepath_summary):
            try:
                with open(json_filepath_summary, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                # Extract chapter summaries to provide as context
                chapter_summaries = []
                for chapter in summary_data.get("chapters", []):
                    chapter_summaries.append(f"Chapter: {chapter.get('chapter_name')}\nSummary: {chapter.get('summary')}\n---")
                summary_context = "\n".join(chapter_summaries)
            except Exception as e:
                print(f"Error reading summary file {json_filepath_summary}: {e}")
                summary_context = ""

    base_prompt = (
        "You are a search query processing expert. For the given user query, perform the following tasks:\n\n"
        "1. Reformulate the Query: Make it more descriptive and contextually complete for use in a semantic vector search.\n\n"
        "2. Extract Important Keywords: Identify the most relevant keywords or short key phrases from the query. "
        "For each keyword, assign a relevance score between 0 and 1. Include only keywords with importance >= 0.3.\n\n"
        "3. Classify Query Type: Determine whether the query is more conceptual or factual. Provide a 'conceptual_score' between 0 and 1.\n\n"
    )

    if summary_context:
        base_prompt += (
            "4. Classify Chapter: Based on the provided chapter summaries, identify which chapter the user's query is most likely related to. "
            "If the query does not clearly relate to any specific chapter, state 'None'.\n\n"
            f"Chapter Summaries:\n{summary_context}\n\n"
            f"Return a single valid JSON object with keys: reformulated_query, keywords (array of {{keyword, importance}}), conceptual_score, classified_chapter.\n\n"
            f'User Query: "{raw_query}"\n\n'
            "Example output:\n"
            '{"reformulated_query":"Detailed...","keywords":[{"keyword":"photosynthesis","importance":0.95}],"conceptual_score":0.85, "classified_chapter": "PLANTS: PARTS AND FUNCTIONS"}\n'
        )
    else:
        base_prompt += (
            f"Return a single valid JSON object with keys: reformulated_query, keywords (array of {{keyword, importance}}), conceptual_score.\n\n"
            f'User Query: "{raw_query}"\n\n'
            "Example output:\n"
            '{"reformulated_query":"Detailed...","keywords":[{"keyword":"photosynthesis","importance":0.95}],"conceptual_score":0.85}\n'
        )

    if not generation_model:
        # Fallback: simple deterministic extraction if no model available
        result = {
            "reformulated_query": raw_query,
            "keywords": [],
            "conceptual_score": 0.5,
            "classification": "conceptual",
        }
        if summary_context:
            result["classified_chapter"] = "None"
        return result

    try:
        response = generation_model.generate_content(base_prompt)
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
            result = {
                "reformulated_query": raw_query,
                "keywords": [],
                "conceptual_score": 0.5,
                "classification": "conceptual",
            }
            if summary_context:
                result["classified_chapter"] = "None"
            return result
    except Exception as e:
        print(f"Error in reformulate_and_classify_query: {e}")
        result = {
            "reformulated_query": raw_query,
            "keywords": [],
            "conceptual_score": 0.5,
            "classification": "conceptual",
        }
        if summary_context:
            result["classified_chapter"] = "None"
        return result


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
        f'**Student\'s Query:** "{raw_query}"\n\n'  # Corrected escaping for apostrophe
        f"**Textbook Context:**\n{context}\n"
    )

    response = generation_model.generate_content([system_prompt, user_prompt])
    return response.text


def generate_teacher_explanation(class_name: str, subject: str, chapter_name: str, summary_text: str) -> str:
    """
    Uses the generative model to create a teacher-like explanation from a chapter summary.
    """
    if not generation_model:
        raise RuntimeError("Generation model not initialized.")

    system_prompt = (
        "You are an expert AI teacher, skilled at explaining complex topics in a simple, "
        "engaging, and easy-to-understand way. Your audience is a student in the specified class."
    )

    user_prompt = f'''
    **Class:** {class_name}
    **Subject:** {subject}
    **Chapter:** {chapter_name}

    **Chapter Summary to Explain:**
    ---
    {summary_text}
    ---

    **Your Task:**
    Based on the summary above, provide a detailed explanation of the chapter's key topics.
    Follow these rules:
    1.  **Act as a Teacher:** Address the student directly in a friendly and encouraging tone.
    2.  **Simplify Concepts:** Break down the main ideas from the summary into simple, digestible points.
    3.  **Use Analogies:** Where appropriate, use simple analogies or real-world examples to make the concepts relatable for a student of this class level.
    4.  **Structure:** Organize the explanation with clear headings for each main topic. Use bullet points or numbered lists to improve readability.
    5.  **Completeness:** Ensure you cover all the main topics mentioned in the summary.
    6.  **Do Not Introduce New Information:** Stick strictly to the concepts presented in the provided summary.

    Begin the explanation now.
    '''

    response = generation_model.generate_content([system_prompt, user_prompt])
    return response.text


def generate_chapter_summary(class_name: str, subject_name: str, chapter_name: str, chapter_chunks: List[str]) -> str:
    """
    Generates a summary for a single chapter using the generative model.
    """
    if not generation_model:
        raise RuntimeError("Generation model not initialized.")

    # Combine chunks into a single text
    full_chapter_text = "\n\n".join(chapter_chunks)

    # Construct the prompt
    prompt = f"""You are an expert educational content summarizer.
Your task is to read the raw textbook chunks from a single chapter and produce a clear, accurate, and well-structured chapter summary.

---

### Input:
You will receive a list of raw text chunks extracted from a chapter of a textbook.
These chunks may be fragmented, repetitive, or discontinuous — your job is to combine them logically.

---

### Your Goal:
Summarize all the given chunks into a **coherent chapter summary** that:
- Covers **all key topics, definitions, and formulas**.
- Explains each major concept in simple and understandable language.
- **Removes redundant or repeated text**.
- Maintains the **natural chapter flow** (introduction → concepts → examples → conclusion).
- Is **concise but complete** enough for a student to study directly from it.

---

### Output Format (JSON):
Return the summarized output in valid JSON format like this:

{{
  "class_name": "{class_name}",
  "subject_name": "{subject_name}",
  "chapter_name": "{chapter_name}",
  "summary": "<clean summarized text covering the full chapter>"
}}

Make sure:
- The JSON is valid and properly formatted.
- Do NOT include the raw chunks or extra commentary.
- Only include the clean summarized text inside the "summary" field.

---

Now read the provided chapter chunks and generate the summarized JSON output as per the format above.

Chapter Chunks:
{full_chapter_text}
"""

    try:
        response = generation_model.generate_content(prompt)
        # Extract the JSON part from the response
        json_text = response.text.strip()
        json_start = json_text.find("{")
        json_end = json_text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            clean_json = json_text[json_start:json_end]
            parsed_json = json.loads(clean_json)
            return parsed_json.get("summary", "")
        else:
            return "Could not generate summary."
    except Exception as e:
        print(f"Error generating summary for chapter {chapter_name}: {e}")
        return "Could not generate summary."


def generate_chapters_from_json(pdf_json: List[Dict]) -> str:
    """
    Builds a prompt string containing the JSON page list for the LLM to extract chapters.
    Returns the prompt as a plain string.
    """
    json_text = json.dumps(pdf_json, ensure_ascii=False)

    prompt = f'''You are an expert assistant tasked with analyzing a textbook to identify its chapters.

The book content is provided as a JSON array, each element representing a PDF page:

[{{"pdf_page": <integer>, "text": "<page text>"}}]

When identifying chapters and their page numbers, prioritize information found in an 'INDEX' or 'Table of Contents' section if available within the provided text.

Return a single valid JSON object following this schema:

{{
  "pdf_offset": <integer>,
  "chapters": [
    {{"chapter_name": "Full name of the chapter", "pdf_startpg": <integer>, "pdf_endpg": <integer>}}
  ]
}}

- pdf_startpg/pdf_endpg are the real PDF page numbers (including front matter).
- Calculate `pdf_offset` as the number of pages of front matter. This is typically (first_chapter_start_page - 1).
- If an index is available, infer the front matter by comparing the index’s chapter start page with the actual PDF page number.
- Do not include any text outside the JSON object.

Here is the book content in JSON format:

{json_text}
'''
    return prompt


def generate_chapters_from_text(json_path: str) -> str:
    """
    Read the page JSON file (json_path), construct prompt and ask the generative model to extract chapters.
    Returns a JSON-string representation of the parsed LLM output or a safe default.
    """
    if not generation_model:
        return json.dumps({"pdf_offset": 0, "chapters": []})

    with open(json_path, "r", encoding="utf-8") as f:
        pdf_pages_data = json.load(f)

    prompt = generate_chapters_from_json(pdf_pages_data)

    try:
        response = generation_model.generate_content(prompt)
        text = response.text.strip()

        json_start = text.find("{")
        if json_start == -1:
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
            return json.dumps({"pdf_offset": 0, "chapters": []})

        clean_json_str = text[json_start:json_end]
        try:
            data = json.loads(clean_json_str)
            # The LLM's pdf_startpg/endpg are taken as is.
            return json.dumps(data)
        except json.JSONDecodeError:
            # If parsing fails, return the safe default, not the broken string.
            return json.dumps({"pdf_offset": 0, "chapters": []})

    except Exception as e:
        return json.dumps({"pdf_offset": 0, "chapters": []})
    


    
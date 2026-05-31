import datetime
import os
import uuid
import hashlib
import json
import logging
from typing import List, Dict, Optional

from qdrant_client import QdrantClient as QC, models
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "data")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- GLOBALS (initialized by initialize()) ---
client: Optional[QC] = None
local_embedder: Optional[SentenceTransformer] = None
gemini_client: Optional[genai.Client] = None
generation_model_name: str = "gemini-2.5-flash"
bm25_indices: Dict[str, BM25Okapi] = {}
book_corpus: Dict[str, List[Dict]] = {}


def initialize():
    """
    Initialize models and Qdrant client. Called once at application startup.
    PRODUCTION MODE: Preserves existing data.
    """
    global client, local_embedder, gemini_client, generation_model_name

    local_embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize Gemini client with new SDK (API key from environment)
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        gemini_client = genai.Client(api_key=api_key)
        from backend.app.utils.gemini_tracker import instrument_client
        gemini_client = instrument_client(gemini_client)
        print("[Qdrant] Gemini client initialized and instrumented successfully in qdrant_service.")
    except Exception as e:
        print(f"[Qdrant Warning] Error initializing Gemini client: {e}")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise ValueError("QDRANT_URL not found in environment variables.")

    try:
        if qdrant_api_key:
            client = QC(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QC(url=qdrant_url)
        print("[Qdrant] Qdrant client initialized successfully.")
    except Exception as e:
        print(f"[Qdrant ERROR] Failed to connect to Qdrant: {e}")
        raise e

    # Create collection and setup payload index if not existing
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            model_embedding_dimension = local_embedder.get_sentence_embedding_dimension()
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=model_embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' created successfully.")

            # Create payload index for keyword filtering
            for field in ["class_name", "subject", "chapter", "book_uuid", "chpstpage", "chpendpage"]:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            print("[Qdrant] Payload indexes created successfully.")
        else:
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists. Preserving existing data.")
    except Exception as e:
        print(f"[Qdrant] Error checking/creating collection: {e}")
        raise e


def get_or_build_bm25_index(book_uuid: str) -> Optional[BM25Okapi]:
    """
    Get cached BM25 index or build one from Qdrant chunks for this book.
    """
    global bm25_indices, book_corpus
    if book_uuid in bm25_indices:
        return bm25_indices[book_uuid]

    corpus_docs = _get_all_chunks_for_book(book_uuid)
    if not corpus_docs:
        print(f"[BM25] Warning: No chunks found in Qdrant for book {book_uuid} to build index.")
        return None

    # Tokenize corpus for BM25
    tokenized_corpus = [doc.get("text", "").split(" ") for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Cache both index and corpus
    bm25_indices[book_uuid] = bm25
    book_corpus[book_uuid] = corpus_docs
    print(f"[BM25] Built and cached BM25 index for book {book_uuid} ({len(corpus_docs)} chunks).")
    return bm25


def _get_all_chunks_for_book(book_uuid: str) -> List[Dict]:
    """
    Helper to fetch ALL payload chunks for a given book_uuid from Qdrant.
    Uses scroll API to handle pagination.
    """
    if not client:
        raise RuntimeError("Qdrant client not initialized.")

    all_docs = []
    offset = None

    while True:
        response, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="book_uuid", match=models.MatchValue(value=book_uuid))]
            ),
            limit=100,
            with_payload=True,
            offset=offset,
        )

        for point in response:
            all_docs.append(point.payload)

        if not next_offset:
            break
        offset = next_offset

    return all_docs


def process_and_embed_book(pdf_path: str, book_uuid: str, class_name: str, subject: str, chapters: List[Dict]) -> bool:
    """
    Process PDF: Parse pages, split into chunks, map chunks to correct chapters,
    generate local embeddings, and upload to Qdrant.
    """
    if not client or not local_embedder:
        raise RuntimeError("Qdrant client or local embedder not initialized.")

    print(f"\n[INGESTION] Start embedding book UUID: {book_uuid}")
    print(f"[INGESTION] File: {pdf_path}")
    print(f"[INGESTION] Class: {class_name}, Subject: {subject}")
    print(f"[INGESTION] Chapters mapped: {len(chapters)}")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"[INGESTION] Total pages in PDF: {total_pages}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    points = []
    point_count = 0

    # Process page-by-page
    for page_idx in range(total_pages):
        page_text = reader.pages[page_idx].extract_text()
        if not page_text or not page_text.strip():
            continue

        real_page_num = page_idx + 1  # 1-indexed

        # Find matching chapter
        chapter_name = "General"
        chpstpage = 0
        chpendpage = 0
        pdf_startpg = 0
        pdf_endpg = 0

        for chapter in chapters:
            start = chapter.get("pdf_startpg")
            end = chapter.get("pdf_endpg")
            if start is not None and end is not None:
                if start <= real_page_num <= end:
                    chapter_name = chapter.get("chapter_name", "General")
                    chpstpage = chapter.get("chpstpage", start)
                    chpendpage = chapter.get("chpendpage", end)
                    pdf_startpg = start
                    pdf_endpg = end
                    break

        # Split text into chunks
        chunks = text_splitter.split_text(page_text)
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Generate embedding vector
            embedding = local_embedder.encode(chunk_text).tolist()

            # Create structured payload
            payload = {
                "text": chunk_text,
                "book_uuid": book_uuid,
                "class_name": class_name,
                "subject": subject,
                "chapter": chapter_name,
                "chapter_name": chapter_name,  # Duplicate for route consistency
                "pdf_page": real_page_num,
                "pdf_startpg": pdf_startpg,
                "pdf_endpg": pdf_endpg,
                "chpstpage": chpstpage,
                "chpendpage": chpendpage,
                "chunk_index": chunk_idx,
                "ingested_at": datetime.datetime.utcnow().isoformat(),
            }

            # Generate unique point UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{book_uuid}_{real_page_num}_{chunk_idx}_{hashlib.md5(chunk_text.encode()).hexdigest()}"))

            # Create Qdrant point object
            points.append(models.PointStruct(id=point_id, vector=embedding, payload=payload))
            point_count += 1

    # Upload to Qdrant in batches
    batch_size = 50
    print(f"[INGESTION] Generated {point_count} chunks. Uploading to Qdrant...")
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    # Invalidate cached BM25 index
    global bm25_indices
    if book_uuid in bm25_indices:
        del bm25_indices[book_uuid]
    if book_uuid in book_corpus:
        del book_corpus[book_uuid]

    print(f"[INGESTION] Successfully uploaded {point_count} chunks to Qdrant.\n")
    return True


def get_chapter_names(book_uuid: str) -> List[str]:
    """
    Get all unique chapter names for a book from Qdrant payload.
    """
    if not client:
        raise RuntimeError("Qdrant client not initialized.")

    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="book_uuid", match=models.MatchValue(value=book_uuid))]
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
                    models.FieldCondition(key="book_uuid", match=models.MatchValue(value=book_uuid)),
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

    # The sort key is based on 'chpstpage'
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
    
    # Robustly handle keywords that might be dictionaries or strings
    keyword_list = []
    for item in keywords:
        if isinstance(item, dict):
            kw = item.get("keyword")
            if isinstance(kw, dict): # Handle nested dict case
                kw = kw.get("keyword")
            if kw:
                keyword_list.append(str(kw))
        elif isinstance(item, str):
            keyword_list.append(item)
            
    keyword_query_str = " ".join(keyword_list)

    # Semantic search
    must_conditions = [models.FieldCondition(key="book_uuid", match=models.MatchValue(value=book_uuid))]
    
    # Add chapter filter if chapter_names provided (for ranking-based filtering)
    if metadata_filters and "chapter_names" in metadata_filters:
        chapter_names = metadata_filters["chapter_names"]
        if chapter_names:  # Only add if list is not empty
            must_conditions.append(
                models.FieldCondition(
                    key="chapter_name",
                    match=models.MatchAny(any=chapter_names)
                )
            )
            print(f"[HYBRID_SEARCH] Filtering to top {len(chapter_names)} chapters: {', '.join(chapter_names[:3])}...")
    
    # Add other metadata filters
    if metadata_filters:
        for key, value in metadata_filters.items():
            if key != "chapter_names":  # Skip chapter_names as it's already handled
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

    # Print the top 5 chunks with their scores
    # print("\n[HYBRID_SEARCH] Top 5 Hybrid Chunks:")
    # for score, doc in ranked_list[:5]:
    #     print(f"  - Score: {score:.4f} | Chunk: {doc.get('text', '')[:100]}...")
    # print()

    return ranked_list[:10], semantic_results, normalized_bm25_results


def embed_query(query: str):
    """
    Encode a query string into an embedding vector using the local embedder.
    """
    if not local_embedder:
        raise RuntimeError("Local embedder not initialized.")
    return local_embedder.encode(query).tolist()


def clear_qdrant_collection():
    """
    Deletes and re-creates the Qdrant collection, effectively clearing all data.
    """
    if not client:
        raise RuntimeError("Qdrant client not initialized.")
    
    try:
        # Delete the collection if it exists
        if client.collection_exists(collection_name=COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
        
        # Re-create the collection
        model_embedding_dimension = local_embedder.get_sentence_embedding_dimension()
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=model_embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        
        # Re-create payload indexes
        for field in ["class_name", "subject", "chapter", "book_uuid", "chpstpage", "chpendpage"]:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                pass  # Silently fail if index already exists
        
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' cleared and re-initialized successfully.")
    except Exception as e:
        print(f"[Qdrant] Error clearing collection: {e}")
        raise

# Helper to fetch metadata (to avoid circular dependency with app)
def get_book_metadata(book_uuid: str) -> Dict:
    """
    Get a single chunk from Qdrant to read the book's class and subject metadata.
    """
    if not client:
        return {}
    try:
        response, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="book_uuid", match=models.MatchValue(value=book_uuid))]
            ),
            limit=1,
            with_payload=["class_name", "subject"],
        )
        if response:
            return response[0].payload
    except:
        pass
    return {}

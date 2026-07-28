from google.cloud import firestore
from backend.app.core.firebase.firebase_init import db
import logging

logger = logging.getLogger(__name__)

def save_summary_document(class_name: str, subject: str, book_uuid: str, chapters: list):
    """
    Creates a single Firestore summary document containing ALL chapter summaries
    for a given class + subject. This document is used ONLY for LLM context.

    Document path:
        summaries/{subject}_{class}
    Example:
        summaries/science_7
    """

    doc_id = f"{subject.strip().lower()}_{class_name.strip().replace(' ', '')}"
    
    # Check if the collection exists, create it if it doesn't
    collections = [col.id for col in db.collections()]
    if "summaries" not in collections:
        # Create a dummy document to create the collection
        db.collection("summaries").document("dummy").set({})
        
    doc_ref = db.collection("summaries").document(doc_id)

    payload = {
        "class": class_name,
        "subject": subject.lower(),
        "book_uuid": book_uuid,
        "chapters": chapters,   # list of chapter dicts
    }

    try:
        logger.info(f"📤 Uploading to Firestore document: summaries/{doc_id}")
        doc_ref.set(payload)
        logger.info(f"✓ Document created/updated successfully")
        logger.info(f"📝 Chapters saved:")
        for chapter in chapters:
            sno = chapter.get('sno', 'N/A')
            ch_name = chapter.get('chapter_name', 'Unknown')
            summary_len = len(chapter.get('summary', ''))
            logger.info(f"   ✓ Chapter {sno}: {ch_name} ({summary_len} chars)")

    except Exception as e:
        logger.error(f"❌ Failed to save summary document summaries/{doc_id}: {e}")
        for chapter in chapters:
            sno = chapter.get('sno', 'N/A')
            logger.error(f"   ✗ Chapter {sno} failed to save")
        raise

# In-memory cache for summaries to avoid repeated Firestore reads
SUMMARY_CACHE = {}

def load_summary_from_firestore(class_name: str, subject: str):
    """
    Loads summaries/{subject}_{class} from Firestore.
    Caches in memory for FAST access (0ms after first load).
    """
    key = f"{subject.strip().lower()}_{class_name.strip().replace(' ', '')}"
    logger.debug(f"Attempting to load summary for key: {key}")

    # Check cached
    if key in SUMMARY_CACHE:
        logger.debug(f"Summary for key '{key}' found in in-memory cache.")
        return SUMMARY_CACHE[key]

    logger.debug(f"Summary for key '{key}' not in cache, fetching from Firestore.")
    # Fetch from Firestore
    doc_ref = db.collection("summaries").document(key)
    doc = doc_ref.get()

    if not doc.exists:
        logger.warning(f"Summary document not found in Firestore for key: {key}")
        # Return None or raise? Let's return None to let caller handle
        return None

    data = doc.to_dict()
    SUMMARY_CACHE[key] = data  # cache it
    logger.debug(f"Summary for key '{key}' fetched from Firestore and cached. Contains {len(data.get('chapters', []))} chapters.")
    return data


def normalize_query_string(q: str) -> str:
    """Normalizes query text to enable robust exact-match caching."""
    import re
    if not q:
        return ""
    q = q.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\s+', ' ', q)
    return q


def check_global_query_cache(raw_query: str, class_name: str, subject: str = None):
    """
    Checks Firestore 'global_query_cache' for a matching query record.
    Returns the cached data if found and valid on disk, else None.
    """
    import os
    normalized = normalize_query_string(raw_query)
    if not normalized:
        return None

    class_str = str(class_name).strip()
    subj_str = str(subject or "").strip().lower()

    logger.info(f"[CACHE] Checking global cache: normalized='{normalized}', class='{class_str}', subject='{subj_str}'")
    try:
        # Search by normalized query and class
        query_ref = db.collection("global_query_cache")\
                      .where("normalized_query", "==", normalized)\
                      .where("class", "==", class_str)
        
        # Only filter by subject if subject is specific and not generic "all"
        if subj_str and subj_str not in ["all", "none", "choose your subject..."]:
            query_ref = query_ref.where("subject", "==", subj_str)
            
        docs = query_ref.limit(1).get()
        
        if not docs:
            logger.info("[CACHE] Global cache miss (no document found)")
            return None

        cached_data = docs[0].to_dict()
        out = cached_data.get("orchestrator_output", {})

        # Verify orchestrator output is complete
        if not out or not out.get("text_narration"):
            logger.warning(f"[CACHE] Cached record for '{raw_query}' is incomplete (empty orchestrator_output). Treating as cache miss.")
            return None

        # If it was a video, verify local files still exist
        if out.get("format_decision") == "VIDEO_REQUIRED":
            interactive_url = cached_data.get("interactive_url", "")
            if not interactive_url:
                logger.info("[CACHE] Cached lesson is video-required but lacks interactive_url. Treating as miss.")
                return None
            
            # Extract lesson_id from path
            parts = interactive_url.split("/")
            if len(parts) >= 3:
                lesson_id = parts[-2]
                
                # Check standard paths
                MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
                PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", ".."))
                expected_index = os.path.join(PROJECT_ROOT, "uploads", "visual_lessons", lesson_id, "index.html")
                fallback_index = os.path.join(PROJECT_ROOT, "hyperframes_engine", "outputs", lesson_id, "index.html")
                
                if not (os.path.exists(expected_index) or os.path.exists(fallback_index)):
                    logger.warning(f"[CACHE] Video files missing on disk for lesson_id {lesson_id}. Forcing cache miss.")
                    return None

        logger.info(f"[CACHE] Global cache hit! Reusing payload for query: '{raw_query}'")
        return cached_data

    except Exception as e:
        logger.error(f"[CACHE] Error checking global cache: {e}")
        return None


def save_to_global_query_cache(raw_query: str, class_name: str, subject: str, orchestrator_output: dict, interactive_url: str = None):
    """
    Saves a query execution result into the 'global_query_cache' collection.
    """
    from datetime import datetime
    if not orchestrator_output or not orchestrator_output.get("text_narration"):
        logger.warning("[CACHE] Rejecting save_to_global_query_cache because orchestrator_output is incomplete.")
        return
    normalized = normalize_query_string(raw_query)
    if not normalized:
        return

    class_str = str(class_name).strip()
    subj_str = str(subject or "").strip().lower()
    
    # Resolve subject from orchestrator output if generic "all"
    if not subj_str or subj_str in ["all", "none", "choose your subject..."]:
        subj_str = str(orchestrator_output.get("matched_subject") or "general knowledge").strip().lower()

    payload = {
        "raw_query": raw_query,
        "normalized_query": normalized,
        "class": class_str,
        "subject": subj_str,
        "orchestrator_output": orchestrator_output,
        "interactive_url": interactive_url,
        "created_at": datetime.now().isoformat()
    }

    try:
        # Create a deterministic document ID to prevent duplicate listings
        doc_id = f"{class_str}_{subj_str}_{normalized}"
        # Firestore document ID cannot exceed 1500 bytes; if query is extremely long, crop or hash it
        if len(doc_id) > 500:
            import hashlib
            doc_id = f"{class_str}_{subj_str}_" + hashlib.md5(normalized.encode()).hexdigest()

        db.collection("global_query_cache").document(doc_id).set(payload)
        logger.info(f"[CACHE] Successfully registered query in global cache: summaries/{doc_id}")
    except Exception as e:
        logger.error(f"[CACHE] Failed to write cache record: {e}")
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
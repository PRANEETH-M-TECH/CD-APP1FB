"""
Analytics Service for CHADUVU-GURU
Handles all analytics logging and aggregation to Firestore.
"""

from google.cloud import firestore
from .firebase.firebase_init import db
import logging
from datetime import datetime, timezone
import pytz
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

def same_local_day(ts_old, ts_new, timezone_str="Asia/Kolkata"):
    tz = pytz.timezone(timezone_str)
    return ts_old.astimezone(tz).date() == ts_new.astimezone(tz).date()

# ============================================
# PHASE 1: FIRESTORE COLLECTION SCHEMAS
# ============================================

"""
Collections created by this service:

1. user_queries - Individual query logs
   Document: auto-generated ID
   Fields: uid, class, subject, chapter_id, chapter_name, query, 
           reformulated_query, mode, llm_action, timestamp, 
           answer_length, ai_difficulty_score

2. user_stats - Aggregated user statistics
   Document ID: {uid}
   Fields: total_queries, last_active, streak, subjects_count,
           chapters_count, weekly_activity, average_difficulty

3. chapter_stats - Chapter analytics
   Document ID: {class}_{subject}_{chapter_id}
   Fields: class, subject, chapter_id, chapter_name, total_queries,
           unique_students, avg_difficulty, last_asked

4. student_mistakes - Learning patterns
   Document ID: {uid}
   Fields: patterns, confusion_topics, recommended_tasks

5. saved_notes - My Bag feature
   Document ID: {uid}
   Fields: notes (array of {title, content, createdAt})
"""

# ============================================
# CORE ANALYTICS FUNCTIONS
# ============================================

def log_query(
    uid: str,
    class_name: str,
    subject: str,
    chapter_id: Optional[int],
    chapter_name: Optional[str],
    query: str,
    reformulated_query: str,
    mode: str,
    llm_action: str,
    answer_length: int,
    ai_difficulty_score: Optional[float] = None
) -> str:
    """
    Logs a single user query to the user_queries collection.
    
    Args:
        uid: User ID
        class_name: Class (e.g., "8")
        subject: Subject name (e.g., "science")
        chapter_id: Chapter ID (optional)
        chapter_name: Chapter name (optional)
        query: Original user query
        reformulated_query: LLM-reformulated query
        mode: "text" or "voice"
        llm_action: Action taken by LLM (e.g., "retrieve_and_answer")
        answer_length: Length of generated answer
        ai_difficulty_score: AI-assessed difficulty (optional)
    
    Returns:
        Document ID of the logged query
    """
    logger.info(f"[ANALYTICS] Attempting to log query for user {uid} in subject {subject}.")
    try:
        # Parse class to integer
        try:
            class_int = int(class_name.replace("Class", "").replace("class", "").strip())
        except:
            class_int = 0
            logger.warning(f"Could not parse class: {class_name}, defaulting to 0")
        
        doc_ref = db.collection("user_queries").document()
        
        query_data = {
            "uid": uid,
            "class": class_int,
            "subject": subject.lower().strip(),
            "chapter_id": chapter_id if chapter_id is not None else 0,
            "chapter_name": chapter_name or "Unknown",
            "query": query,
            "reformulated_query": reformulated_query,
            "mode": mode,
            "llm_action": llm_action,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "answer_length": answer_length,
        }
        
        if ai_difficulty_score is not None:
            query_data["ai_difficulty_score"] = ai_difficulty_score
        
        doc_ref.set(query_data)
        logger.info(f"✅ Query logged: {doc_ref.id} for user {uid}")
        return doc_ref.id
        
    except Exception as e:
        logger.error(f"❌ Failed to log query for user {uid}: {e}")
        raise


def update_user_stats(
    uid: str,
    subject: str,
    chapter_id: Optional[int],
    class_name: str
) -> None:
    """
    Updates aggregated user statistics with atomic operations.
    Stats are strictly separated by class.
    
    Args:
        uid: User ID
        subject: Subject name
        chapter_id: Chapter ID (optional)
        class_name: Class name
    """
    try:
        # Parse class to integer for consistent key generation
        try:
            class_int = int(class_name.replace("Class", "").replace("class", "").strip())
        except:
            class_int = 0

        # STRICT ISOLATION: Use composite key {uid}_{class}
        # This ensures a user in Class 8 has separate stats from Class 9
        stats_doc_id = f"{uid}_{class_int}"
        doc_ref = db.collection("user_stats").document(stats_doc_id)
        doc = doc_ref.get()

        today_str = datetime.now().date().strftime("%Y-%m-%d")
        subject_key = f"subjects_count.{subject.lower()}"
        chapter_key = f"chapters_count.{subject.lower()}_{chapter_id}" if chapter_id else None

        if not doc.exists:
            # Document doesn't exist, create it
            new_data = {
                "uid": uid,
                "class": class_int,
                "total_queries": 1,
                "last_active": firestore.SERVER_TIMESTAMP,
                "streak": 1,
                "subjects_count": {subject.lower(): 1},
                "weekly_activity": {today_str: 1},
                "chapters_count": {}
            }
            if chapter_key:
                new_data["chapters_count"] = {f"{subject.lower()}_{chapter_id}": 1}

            doc_ref.set(new_data)
            logger.info(f"✅ Created new user stats for {uid} in Class {class_int}")

        else:
            # Document exists, update it
            current_data = doc.to_dict()
            
            # Calculate streak
            now = datetime.now(timezone.utc)

            # Default timezone fallback
            user_timezone = current_data.get("timezone", "Asia/Kolkata")

            # Get last active timestamp
            last_active = current_data.get("last_active")

            if last_active:
                if same_local_day(last_active, now, user_timezone):
                    # same day → streak unchanged
                    streak = current_data.get("streak", 1)
                else:
                    # different day → streak increments by 1
                    streak = current_data.get("streak", 0) + 1
            else:
                streak = 1  # first login
            
            # Prepare weekly activity key
            weekly_key = f"weekly_activity.{today_str}"
            
            # Prepare update data
            update_data = {
                "total_queries": firestore.Increment(1),
                "last_active": firestore.SERVER_TIMESTAMP,
                "streak": streak,
                subject_key: firestore.Increment(1),
                weekly_key: firestore.Increment(1),
            }
            if chapter_key:
                update_data[chapter_key] = firestore.Increment(1)

            doc_ref.update(update_data)
            logger.info(f"✅ User stats updated for {uid} (Class {class_int}): streak={streak}")

    except Exception as e:
        logger.error(f"❌ Failed to update user stats for {uid}: {e}", exc_info=True)
        raise


def update_chapter_stats(
    class_name: str,
    subject: str,
    chapter_id: int,
    chapter_name: str,
    uid: str
) -> None:
    """
    Updates chapter-level analytics with atomic operations.
    
    Args:
        class_name: Class (e.g., "8")
        subject: Subject name
        chapter_id: Chapter ID
        chapter_name: Chapter name
        uid: User ID (to track unique students)
    """
    try:
        # Parse class to integer
        try:
            class_int = int(class_name.replace("Class", "").replace("class", "").strip())
        except:
            class_int = 0
        
        # Document ID: {class}_{subject}_{chapter_id}
        doc_id = f"{class_int}_{subject.lower()}_{chapter_id}"
        doc_ref = db.collection("chapter_stats").document(doc_id)
        
        # Check if document exists
        doc = doc_ref.get()
        
        if not doc.exists:
            # Create new document with initial data
            doc_ref.set({
                "class": class_int,
                "subject": subject.lower(),
                "chapter_id": chapter_id,
                "chapter_name": chapter_name,
                "total_queries": 1,
                "unique_students": [uid],
                "avg_difficulty": 0.0,
                "last_asked": firestore.SERVER_TIMESTAMP
            })
            logger.info(f"✅ Created new chapter stats: {doc_id}")
        else:
            # Update existing document
            update_data = {
                "total_queries": firestore.Increment(1),
                "unique_students": firestore.ArrayUnion([uid]),
                "last_asked": firestore.SERVER_TIMESTAMP
            }
            doc_ref.update(update_data)
            logger.info(f"✅ Updated chapter stats: {doc_id}")
            
    except Exception as e:
        logger.error(f"❌ Failed to update chapter stats for {class_name}_{subject}_{chapter_id}: {e}")
        raise


def update_mistake_patterns(
    uid: str,
    patterns: Optional[List[str]] = None,
    confusion_topics: Optional[List[str]] = None,
    recommended_tasks: Optional[List[str]] = None
) -> None:
    """
    Updates student mistake patterns and learning recommendations.
    
    Args:
        uid: User ID
        patterns: List of identified patterns (optional)
        confusion_topics: List of topics causing confusion (optional)
        recommended_tasks: List of recommended practice tasks (optional)
    """
    try:
        doc_ref = db.collection("student_mistakes").document(uid)
        
        update_data = {}
        
        if patterns:
            update_data["patterns"] = firestore.ArrayUnion(patterns)
        
        if confusion_topics:
            update_data["confusion_topics"] = firestore.ArrayUnion(confusion_topics)
        
        if recommended_tasks:
            update_data["recommended_tasks"] = firestore.ArrayUnion(recommended_tasks)
        
        if update_data:
            doc_ref.set(update_data, merge=True)
            logger.info(f"✅ Mistake patterns updated for {uid}")
        
    except Exception as e:
        logger.error(f"❌ Failed to update mistake patterns for {uid}: {e}")
        raise


# ============================================
# NOTES MANAGEMENT (MY BAG FEATURE)
# ============================================

def add_note(uid: str, title: str, content: str) -> None:
    """
    Adds a note to user's saved notes.
    
    Args:
        uid: User ID
        title: Note title
        content: Note content
    """
    try:
        doc_ref = db.collection("saved_notes").document(uid)
        
        note = {
            "title": title,
            "content": content,
            "createdAt": firestore.SERVER_TIMESTAMP
        }
        
        doc_ref.set({
            "notes": firestore.ArrayUnion([note])
        }, merge=True)
        
        logger.info(f"✅ Note added for user {uid}: {title}")
        
    except Exception as e:
        logger.error(f"❌ Failed to add note for {uid}: {e}")
        raise


def get_notes(uid: str) -> List[Dict]:
    """
    Retrieves all notes for a user.
    
    Args:
        uid: User ID
    
    Returns:
        List of note dictionaries
    """
    try:
        doc_ref = db.collection("saved_notes").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            notes = data.get("notes", [])
            logger.info(f"✅ Retrieved {len(notes)} notes for user {uid}")
            return notes
        else:
            logger.info(f"No notes found for user {uid}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Failed to get notes for {uid}: {e}")
        raise


def delete_note(uid: str, note_index: int) -> None:
    """
    Deletes a note by index.
    
    Args:
        uid: User ID
        note_index: Index of note to delete (0-based)
    """
    try:
        doc_ref = db.collection("saved_notes").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            notes = data.get("notes", [])
            
            if 0 <= note_index < len(notes):
                notes.pop(note_index)
                doc_ref.update({"notes": notes})
                logger.info(f"✅ Note {note_index} deleted for user {uid}")
            else:
                logger.warning(f"Invalid note index {note_index} for user {uid}")
        else:
            logger.warning(f"No notes document found for user {uid}")
            
    except Exception as e:
        logger.error(f"❌ Failed to delete note for {uid}: {e}")
        raise


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_user_stats(uid: str) -> Optional[Dict]:
    """
    Retrieves user statistics.
    
    Args:
        uid: User ID
    
    Returns:
        User stats dictionary or None if not found
    """
    try:
        doc_ref = db.collection("user_stats").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to get user stats for {uid}: {e}")
        return None


def get_chapter_stats(class_name: str, subject: str, chapter_id: int) -> Optional[Dict]:
    """
    Retrieves chapter statistics.
    
    Args:
        class_name: Class
        subject: Subject name
        chapter_id: Chapter ID
    
    Returns:
        Chapter stats dictionary or None if not found
    """
    try:
        class_int = int(class_name.replace("Class", "").replace("class", "").strip())
        doc_id = f"{class_int}_{subject.lower()}_{chapter_id}"
        
        doc_ref = db.collection("chapter_stats").document(doc_id)
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to get chapter stats: {e}")
        return None


logger.info("✅ Analytics service loaded successfully")

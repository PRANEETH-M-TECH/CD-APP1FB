"""
Comprehensive Database Migration & Consolidation Script for CHADUVU-GURU
Migrates all legacy collections into the two core root pillars: 'classes' and 'users'.

Pillars & Sub-collections:
1. classes
   - summaries: classes/{class}/subjects/{subject}
   - stats (chapter stats): classes/{class}/subjects/{subject}/stats/{chapter_id}
   - query_cache: classes/{class}/subjects/{subject}/query_cache/{cache_id}
2. users
   - queries: users/{uid}/queries/{query_doc_id}
   - notebooks: users/{uid}/notebooks/{notebook_id}
   - bag items: users/{uid}/notebooks/{notebook_id}/items/{item_id}
   - stats (user stats): users/{uid}/stats/stats_doc
   - achievements: users/{uid}/achievements/achievements_doc
   - mistakes (mistakes, weak areas): users/{uid}/mistakes/mistakes_doc & users/{uid}/mistakes/weak_areas
   - topic_analytics: users/{uid}/topic_analytics/{topic_id}
   - user_analytics: users/{uid}/user_analytics/user_analytics_doc
   - frequent_questions: users/{uid}/stats/frequent_questions
"""

import os
import sys
import datetime
from google.cloud import firestore

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.app.core.firebase.firebase_init import db

def format_timestamp_id(timestamp, class_val):
    """Generates a sorted chronological doc ID."""
    if not timestamp:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    if hasattr(timestamp, 'to_dict'):
        timestamp = timestamp.datetime
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    try:
        class_clean = int(str(class_val).replace("Class", "").replace("class", "").strip())
    except Exception:
        class_clean = 0
    return f"{timestamp_str}_class{class_clean}"

def migrate_summaries():
    print("[INFO] Migrating book summaries from 'summaries'...")
    legacy_ref = db.collection("summaries")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} summaries to migrate.")
    count = 0
    for doc in docs:
        if doc.id == "dummy":
            continue
        data = doc.to_dict()
        class_name = data.get("class")
        subject = data.get("subject")
        if not class_name or not subject:
            parts = doc.id.split("_")
            if len(parts) >= 2:
                subject, class_name = parts[0], parts[1]
            else:
                continue
        clean_class = "".join(c for c in str(class_name) if c.isdigit()) or "unknown"
        new_ref = db.collection("classes").document(clean_class).collection("subjects").document(subject.strip().lower())
        new_ref.set(data)
        count += 1
    print(f"[SUCCESS] Migrated {count}/{len(docs)} summaries.\n")

def migrate_chapter_stats():
    print("[INFO] Migrating chapter stats from 'chapter_stats'...")
    legacy_ref = db.collection("chapter_stats")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} chapter stats to migrate.")
    count = 0
    for doc in docs:
        data = doc.to_dict()
        class_name = data.get("class", "0")
        subject = data.get("subject", "unknown").lower().strip()
        chapter_id = data.get("chapter_id", doc.id.split("_")[-1])
        
        clean_class = "".join(c for c in str(class_name) if c.isdigit()) or "0"
        new_ref = db.collection("classes").document(clean_class)\
                    .collection("subjects").document(subject)\
                    .collection("stats").document(str(chapter_id))
        new_ref.set(data)
        count += 1
    print(f"[SUCCESS] Migrated {count}/{len(docs)} chapter stats.\n")

def migrate_global_query_cache():
    print("[INFO] Migrating global query cache...")
    legacy_ref = db.collection("global_query_cache")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} cache docs to migrate.")
    count = 0
    for doc in docs:
        data = doc.to_dict()
        class_name = data.get("class", "unknown")
        subject = data.get("subject", "general").lower().strip()
        clean_class = "".join(c for c in str(class_name) if c.isdigit()) or "unknown"
        
        new_ref = db.collection("classes").document(clean_class)\
                    .collection("subjects").document(subject)\
                    .collection("query_cache").document(doc.id)
        new_ref.set(data)
        count += 1
    print(f"[SUCCESS] Migrated {count}/{len(docs)} cache docs.\n")

def migrate_queries():
    print("[INFO] Migrating queries from 'user_queries'...")
    legacy_ref = db.collection("user_queries")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} user queries to migrate.")
    count = 0
    for doc in docs:
        data = doc.to_dict()
        uid = data.get("uid")
        if not uid:
            continue
        timestamp = data.get("timestamp")
        class_val = data.get("class", 0)
        new_doc_id = format_timestamp_id(timestamp, class_val)
        new_ref = db.collection("users").document(uid).collection("queries").document(new_doc_id)
        new_ref.set(data)
        count += 1
    print(f"[SUCCESS] Migrated {count}/{len(docs)} queries.\n")

def migrate_notebooks_and_items():
    print("[INFO] Migrating notebooks and items...")
    notebooks_ref = db.collection("notebooks")
    nb_docs = list(notebooks_ref.stream())
    print(f"Found {len(nb_docs)} notebooks to migrate.")
    notebook_owners = {}
    count_nb = 0
    for doc in nb_docs:
        data = doc.to_dict()
        uid = data.get("uid")
        if not uid:
            continue
        notebook_owners[doc.id] = uid
        new_ref = db.collection("users").document(uid).collection("notebooks").document(doc.id)
        new_ref.set(data)
        count_nb += 1
        
    items_ref = db.collection("bag_items")
    item_docs = list(items_ref.stream())
    print(f"Found {len(item_docs)} items to migrate.")
    count_items = 0
    for doc in item_docs:
        data = doc.to_dict()
        notebook_id = data.get("notebook_id")
        uid = data.get("uid") or notebook_owners.get(notebook_id)
        if not uid or not notebook_id:
            continue
        new_ref = db.collection("users").document(uid).collection("notebooks").document(notebook_id).collection("items").document(doc.id)
        new_ref.set(data)
        count_items += 1
    print(f"[SUCCESS] Migrated {count_nb} notebooks and {count_items} items.\n")

def migrate_user_stats():
    print("[INFO] Migrating user stats...")
    stats_ref = db.collection("user_stats")
    docs = list(stats_ref.stream())
    print(f"Found {len(docs)} user stats docs.")
    count = 0
    for doc in docs:
        new_ref = db.collection("users").document(doc.id).collection("stats").document("stats_doc")
        new_ref.set(doc.to_dict())
        count += 1
    print(f"[SUCCESS] Migrated {count} user stats docs.\n")

def migrate_weak_areas():
    print("[INFO] Migrating weak areas...")
    legacy_ref = db.collection("weak_areas")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} weak area docs.")
    count = 0
    for doc in docs:
        new_ref = db.collection("users").document(doc.id).collection("mistakes").document("weak_areas")
        new_ref.set(doc.to_dict())
        count += 1
    print(f"[SUCCESS] Migrated {count} weak area docs.\n")

def migrate_topic_analytics():
    print("[INFO] Migrating topic analytics...")
    legacy_ref = db.collection("topic_analytics")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} topic analytics to migrate.")
    count = 0
    for doc in docs:
        data = doc.to_dict()
        uid = data.get("uid")
        if not uid:
            # Parse from ID: {uid}_{subject}_{chapter}_{slug}
            parts = doc.id.split("_")
            if len(parts) >= 4:
                uid = parts[0]
            else:
                continue
        # Extract topic slug and construct new sub-doc ID
        subject = data.get("subject", "general")
        chapter_id = data.get("chapter_id", "0")
        topic = data.get("topic", "unknown")
        topic_slug = topic.lower().replace(" ", "_")[:50]
        topic_doc_id = f"{subject}_{chapter_id}_{topic_slug}"
        
        new_ref = db.collection("users").document(uid).collection("topic_analytics").document(topic_doc_id)
        new_ref.set(data)
        count += 1
    print(f"[SUCCESS] Migrated {count}/{len(docs)} topic analytics docs.\n")

def migrate_user_analytics():
    print("[INFO] Migrating user analytics...")
    legacy_ref = db.collection("user_analytics")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} user analytics to migrate.")
    count = 0
    for doc in docs:
        new_ref = db.collection("users").document(doc.id).collection("user_analytics").document("user_analytics_doc")
        new_ref.set(doc.to_dict())
        count += 1
    print(f"[SUCCESS] Migrated {count} user analytics docs.\n")

def migrate_frequent_questions():
    print("[INFO] Migrating frequent questions...")
    legacy_ref = db.collection("frequent_questions")
    docs = list(legacy_ref.stream())
    print(f"Found {len(docs)} frequent questions docs.")
    count = 0
    for doc in docs:
        new_ref = db.collection("users").document(doc.id).collection("stats").document("frequent_questions")
        new_ref.set(doc.to_dict())
        count += 1
    print(f"[SUCCESS] Migrated {count} frequent questions docs.\n")

def delete_legacy_collections():
    print("[INFO] Starting cleanup of all legacy root collections...")
    collections_to_delete = [
        "user_queries",
        "user_stats",
        "notebooks",
        "bag_items",
        "student_mistakes",
        "summaries",
        "chapter_stats",
        "weak_areas",
        "topic_analytics",
        "user_analytics",
        "frequent_questions",
        "global_query_cache"
    ]
    for col_name in collections_to_delete:
        print(f"   Deleting documents in legacy collection: '{col_name}'...")
        col_ref = db.collection(col_name)
        docs = list(col_ref.stream())
        deleted_count = 0
        for doc in docs:
            doc.reference.delete()
            deleted_count += 1
        print(f"   [SUCCESS] Deleted {deleted_count} documents from '{col_name}'.")
    print("[SUCCESS] Consolidated database! Old collections are now empty.\n")

def run_migration():
    print("==================================================")
    print("STARTING FULL MIGRATION & CONSOLIDATION")
    print("==================================================")
    try:
        migrate_summaries()
        migrate_chapter_stats()
        migrate_global_query_cache()
        migrate_queries()
        migrate_notebooks_and_items()
        migrate_user_stats()
        migrate_weak_areas()
        migrate_topic_analytics()
        migrate_user_analytics()
        migrate_frequent_questions()
        
        # Clear out the old root collections
        delete_legacy_collections()
        
        print("==================================================")
        print("MIGRATION & CLEANUP SUCCESSFUL!")
        print("All data is now consolidated under 'classes' and 'users'.")
        print("==================================================")
    except Exception as e:
        print(f"\n[ERROR] CONSOLIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_migration()

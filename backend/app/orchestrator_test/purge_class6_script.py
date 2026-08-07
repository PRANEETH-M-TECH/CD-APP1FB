import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

import json
from backend.app.core.firebase.firebase_init import db
from backend.app.services.retrieval import qdrant_service
from qdrant_client import models

def delete_collection_recursive(coll_ref, batch_size=100):
    docs = list(coll_ref.limit(batch_size).stream())
    deleted = 0
    for doc in docs:
        for subcoll in doc.reference.collections():
            deleted += delete_collection_recursive(subcoll, batch_size)
        doc.reference.delete()
        deleted += 1
    if len(docs) >= batch_size:
        deleted += delete_collection_recursive(coll_ref, batch_size)
    return deleted

def delete_document_recursive(doc_ref):
    deleted = 0
    for subcoll in doc_ref.collections():
        deleted += delete_collection_recursive(subcoll)
    doc_ref.delete()
    deleted += 1
    return deleted

def run_purge():
    print("==================================================")
    print("      PURGING ALL CLASS-6 DATA FROM SYSTEM       ")
    print("==================================================")
    
    # ---------------------------------------------------------
    # 1. FIRESTORE PURGE
    # ---------------------------------------------------------
    print("\n--- 1. FIRESTORE PURGE ---")
    try:
        class_doc_ref = db.collection("classes").document("6")
        doc_snap = class_doc_ref.get()
        if doc_snap.exists or len(list(class_doc_ref.collections())) > 0:
            count = delete_document_recursive(class_doc_ref)
            print(f"   [Firestore] Deleted document 'classes/6' and {count} associated subcollection docs.")
        else:
            print("   [Firestore] Document 'classes/6' not found or already deleted.")
            
        # Top-level books collection cleanup
        books_ref = db.collection("books").stream()
        b_deleted = 0
        for b in books_ref:
            d = b.to_dict()
            if str(d.get("class_name") or d.get("class") or d.get("grade") or "") == "6":
                delete_document_recursive(b.reference)
                b_deleted += 1
                print(f"   [Firestore] Deleted book doc '{b.id}' ({d.get('title') or d.get('name')}) from 'books'.")
        if b_deleted == 0:
            print("   [Firestore] No Class 6 books found in top-level 'books' collection.")

        # Top-level chapters collection cleanup
        chaps_ref = db.collection("chapters").stream()
        c_deleted = 0
        for c in chaps_ref:
            d = c.to_dict()
            if str(d.get("class_name") or d.get("class") or d.get("grade") or "") == "6":
                c.reference.delete()
                c_deleted += 1
        if c_deleted > 0:
            print(f"   [Firestore] Deleted {c_deleted} Class 6 chapter docs from top-level 'chapters'.")
        else:
            print("   [Firestore] No Class 6 chapter docs found in top-level 'chapters'.")

    except Exception as e:
        print(f"   [Firestore ERROR]: {e}")

    # ---------------------------------------------------------
    # 2. QDRANT PURGE BY SCROLL & POINT ID DELETION
    # ---------------------------------------------------------
    print("\n--- 2. QDRANT VECTOR PURGE ---")
    try:
        qdrant_service.initialize()

        class6_uuids = [
            "02388bd2e1e88738b9ce21a5d1c7ce9b",  # Class 6 Science
            "d2918761-6481-5762-acdc-3d59bbeccd15"  # Class 6 Social
        ]

        offset = None
        all_point_ids_to_delete = []

        while True:
            scroll_res = qdrant_service.client.scroll(
                collection_name=qdrant_service.COLLECTION_NAME,
                limit=250,
                offset=offset,
                with_payload=True
            )
            points, next_offset = scroll_res
            if not points:
                break

            for p in points:
                payload = p.payload or {}
                b_uuid = str(payload.get("book_uuid") or "")
                c_name = str(payload.get("class_name") or payload.get("class") or payload.get("grade") or "")
                f_name = str(payload.get("filename") or payload.get("book_name") or "")

                if (b_uuid in class6_uuids) or (c_name == "6") or ("class6" in f_name.lower()) or ("science6" in f_name.lower()):
                    all_point_ids_to_delete.append(p.id)

            if not next_offset:
                break
            offset = next_offset

        print(f"   [Qdrant] Identified {len(all_point_ids_to_delete)} Class 6 point IDs to delete.")

        if all_point_ids_to_delete:
            # Batch delete point IDs
            batch_size = 100
            for i in range(0, len(all_point_ids_to_delete), batch_size):
                batch = all_point_ids_to_delete[i:i + batch_size]
                qdrant_service.client.delete(
                    collection_name=qdrant_service.COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=batch)
                )
            print(f"   [Qdrant] Successfully deleted {len(all_point_ids_to_delete)} points from collection '{qdrant_service.COLLECTION_NAME}'.")
        else:
            print("   [Qdrant] No Class 6 points found in Qdrant.")

        # Post-purge verification scroll
        scroll_res_verify, _ = qdrant_service.client.scroll(
            collection_name=qdrant_service.COLLECTION_NAME,
            limit=500,
            with_payload=True
        )
        remaining_class6 = 0
        for p in scroll_res_verify:
            payload = p.payload or {}
            b_uuid = str(payload.get("book_uuid") or "")
            c_name = str(payload.get("class_name") or payload.get("class") or payload.get("grade") or "")
            if b_uuid in class6_uuids or c_name == "6":
                remaining_class6 += 1

        print(f"   [Qdrant Verification] Remaining Class 6 vector points: {remaining_class6}")

    except Exception as e:
        print(f"   [Qdrant ERROR]: {e}")

    # ---------------------------------------------------------
    # 3. LOCAL FILE CACHE PURGE
    # ---------------------------------------------------------
    print("\n--- 3. LOCAL FILES & CACHE PURGE ---")

    # A. chapters_cache.json
    cache_path = os.path.join(PROJECT_ROOT, "chapterdata", "chapters_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            keys_to_delete = []
            for key, val in cache_data.items():
                if key.startswith("6_") or str(val.get("class_name") or val.get("class") or "") == "6":
                    keys_to_delete.append(key)
            
            for k in keys_to_delete:
                del cache_data[k]
                print(f"   [Cache File] Removed key '{k}' from chapters_cache.json.")

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            print(f"   [Cache File] Updated {cache_path}.")
        except Exception as e:
            print(f"   [Cache File ERROR]: {e}")

    # B. chpchunks/
    chpchunks_dir = os.path.join(PROJECT_ROOT, "chpchunks")
    if os.path.exists(chpchunks_dir):
        for fname in os.listdir(chpchunks_dir):
            fpath = os.path.join(chpchunks_dir, fname)
            if os.path.isfile(fpath):
                if "6" in fname or fname.startswith("science6") or fname.startswith("social6"):
                    os.remove(fpath)
                    print(f"   [Chunks] Removed chunk file: {fname}")

    # C. bm25_indices/
    bm25_dir = os.path.join(PROJECT_ROOT, "bm25_indices")
    if os.path.exists(bm25_dir):
        for fname in os.listdir(bm25_dir):
            if fname.endswith(".pkl"):
                for uuid_str in class6_uuids:
                    if uuid_str in fname:
                        fpath = os.path.join(bm25_dir, fname)
                        os.remove(fpath)
                        print(f"   [BM25 Index] Removed index file: {fname}")

    # D. uploads/
    uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
    if os.path.exists(uploads_dir):
        for fname in os.listdir(uploads_dir):
            if fname.startswith("class6") or fname == "science6.pdf" or fname == "social6.pdf":
                fpath = os.path.join(uploads_dir, fname)
                os.remove(fpath)
                print(f"   [Uploads] Removed PDF file: {fname}")

    print("\n==================================================")
    print("      PURGE COMPLETED SUCCESSFULLY FOR CLASS-6   ")
    print("==================================================")

if __name__ == "__main__":
    run_purge()

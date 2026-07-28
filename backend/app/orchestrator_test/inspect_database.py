import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.app.core.firebase.firebase_init import db
from backend.app.services.retrieval import qdrant_service

def inspect():
    lines = []
    lines.append("==================================================")
    lines.append("      CHADUVU GURU - DATABASE & VECTOR INVENTORY  ")
    lines.append("==================================================")
    
    # 1. Firestore Inventory
    lines.append("\n1. FIRESTORE DATABASE INVENTORY:")
    try:
        books_ref = db.collection("books").get()
        lines.append(f"   Total Books in Firestore: {len(books_ref)}")
        for b in books_ref:
            d = b.to_dict()
            lines.append(f"   Book: {d.get('title') or d.get('name')} | Class: {d.get('class_name') or d.get('class')} | Subject: {d.get('subject')}")
    except Exception as e:
        lines.append(f"   Firestore Books Error: {e}")

    try:
        chapters_ref = db.collection("chapters").get()
        lines.append(f"   Total Chapter Records in Firestore: {len(chapters_ref)}")
    except Exception as e:
        lines.append(f"   Firestore Chapters Error: {e}")

    # 2. Qdrant Vector Collection Inventory
    lines.append("\n2. QDRANT VECTOR COLLECTION INVENTORY:")
    qdrant_service.initialize()
    
    try:
        points, _ = qdrant_service.client.scroll(
            collection_name=qdrant_service.COLLECTION_NAME,
            limit=500,
            with_payload=True
        )
        lines.append(f"   Total Indexed Chunk Points Scrolled in Qdrant: {len(points)}")
        
        vector_inventory = {}
        for p in points:
            payload = p.payload or {}
            cls = str(payload.get("class_name") or payload.get("class") or payload.get("grade") or "Unknown")
            subj = str(payload.get("subject") or "Unknown")
            book = str(payload.get("book_name") or payload.get("book_uuid") or "Unknown")
            chap = str(payload.get("chapter_name") or "Unknown")
            
            if cls not in vector_inventory:
                vector_inventory[cls] = {}
            if subj not in vector_inventory[cls]:
                vector_inventory[cls][subj] = set()
            vector_inventory[cls][subj].add((book, chap))
            
        for cls, subjs in sorted(vector_inventory.items()):
            lines.append(f"\n   [CLASS / GRADE IN QDRANT]: {cls}")
            for subj, b_set in sorted(subjs.items()):
                lines.append(f"      - Subject: {subj}")
                for b_title, chap in sorted(b_set):
                    lines.append(f"        * Book/Source: {b_title} | Chapter: {chap}")
    except Exception as e:
        lines.append(f"   [ERROR] Error inspecting Qdrant: {e}")

    lines.append("\n==================================================")
    report_content = "\n".join(lines)
    
    out_file = os.path.join(os.path.dirname(__file__), "database_inventory_report.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(report_content)

if __name__ == "__main__":
    inspect()

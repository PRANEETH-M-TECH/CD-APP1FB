import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.app.services.retrieval import qdrant_service

def inspect_classes():
    qdrant_service.initialize()
    points, _ = qdrant_service.client.scroll(
        collection_name=qdrant_service.COLLECTION_NAME,
        limit=1000,
        with_payload=True
    )
    
    mapping = {}
    for p in points:
        payload = p.payload or {}
        book = payload.get("book_name") or payload.get("book_uuid") or "Unknown Book"
        cls = payload.get("class_name") or payload.get("class") or payload.get("grade") or "Not Specified"
        subj = payload.get("subject") or "Not Specified"
        chap = payload.get("chapter_name") or "Unknown Chapter"
        
        key = (cls, subj, book)
        if key not in mapping:
            mapping[key] = set()
        mapping[key].add(chap)
        
    print("==================================================")
    print("   EXACT CLASS & SUBJECT MAPPING IN QDRANT VECTOR DB ")
    print("==================================================")
    
    for (cls, subj, book), chapters in sorted(mapping.items()):
        print(f"\n🎓 CLASS / GRADE: {cls}")
        print(f"   • Subject : {subj}")
        print(f"   • Book    : {book}")
        print(f"   • Chapters Indexed ({len(chapters)}):")
        for chap in sorted(chapters)[:5]:  # print first 5 chapters per book
            print(f"     - {chap}")
        if len(chapters) > 5:
            print(f"     ... and {len(chapters) - 5} more chapters")
            
    print("==================================================")

if __name__ == "__main__":
    inspect_classes()

#!/usr/bin/env python3
"""
Script to check what's actually in Firestore for chapter summaries.
This will help us understand if page numbers are being overwritten.
"""
from google.cloud import firestore
import json
import os

# Initialize Firestore client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceAccountKey.json'
db = firestore.Client()


def check_firestore_summaries():
    """
    Read all summary documents from Firestore and display their structure.
    """
    print("="*80)
    print("CHECKING FIRESTORE SUMMARY DOCUMENTS")
    print("="*80)
    
    # Get all documents from summaries collection
    summaries_ref = db.collection('summaries')
    docs = summaries_ref.stream()
    
    found_any = False
    for doc in docs:
        found_any = True
        doc_id = doc.id
        data = doc.to_dict()
        
        if doc_id == 'dummy':
            continue
            
        print(f"\n📄 Document ID: {doc_id}")
        print(f"   Class: {data.get('class', 'N/A')}")
        print(f"   Subject: {data.get('subject', 'N/A')}")
        print(f"   Book UUID: {data.get('book_uuid', 'N/A')}")
        print(f"   Number of chapters: {len(data.get('chapters', []))}")
        
        chapters = data.get('chapters', [])
        if chapters:
            print(f"\n   📚 Chapter Details:")
            for idx, chapter in enumerate(chapters[:3], 1):  # Show first 3 chapters
                print(f"\n   Chapter {idx}:")
                print(f"      sno: {chapter.get('sno', '❌ MISSING')}")
                print(f"      chapter_name: {chapter.get('chapter_name', '❌ MISSING')}")
                print(f"      pdf_startpg: {chapter.get('pdf_startpg', '❌ MISSING')}")
                print(f"      pdf_endpg: {chapter.get('pdf_endpg', '❌ MISSING')}")
                print(f"      chpstpage: {chapter.get('chpstpage', '❌ MISSING')}")
                print(f"      chpendpage: {chapter.get('chpendpage', '❌ MISSING')}")
                print(f"      summary length: {len(chapter.get('summary', ''))} chars")
                
                # Check for any unexpected fields
                expected_fields = {'sno', 'chapter_name', 'pdf_startpg', 'pdf_endpg', 
                                 'chpstpage', 'chpendpage', 'summary'}
                actual_fields = set(chapter.keys())
                unexpected = actual_fields - expected_fields
                missing = expected_fields - actual_fields
                
                if unexpected:
                    print(f"      ⚠️  Unexpected fields: {unexpected}")
                if missing:
                    print(f"      ❌ Missing fields: {missing}")
            
            if len(chapters) > 3:
                print(f"\n   ... and {len(chapters) - 3} more chapters")
        
        print("\n" + "-"*80)
    
    if not found_any:
        print("\n❌ No summary documents found in Firestore!")
        print("   Please upload and process a book first.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        check_firestore_summaries()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

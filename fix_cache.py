#!/usr/bin/env python3
"""
Script to fix the corrupted chapters_cache.json file.
This converts old format (where pdf_startpg actually contained chapter pages)
to the new format (with separate pdf and chapter page fields).
"""
import json
import os

CACHE_PATH = "chapterdata/chapters_cache.json"
BACKUP_PATH = "chapterdata/chapters_cache_backup.json"

def fix_cache():
    """
    Fix the cache file by converting old format to new format.
    """
    print("="*80)
    print("FIXING CHAPTERS CACHE")
    print("="*80)
    
    # Load existing cache
    try:
        with open(CACHE_PATH, "r") as f:
            cache = json.load(f)
        print(f"\n✓ Loaded cache with {len(cache)} books")
    except FileNotFoundError:
        print("\n❌ Cache file not found!")
        return
    
    # Create backup
    with open(BACKUP_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"✓ Created backup at {BACKUP_PATH}")
    
    # Fix each book
    fixed_count = 0
    for book_key, book_data in cache.items():
        if book_key == "dummy":
            continue
            
        pdf_offset = book_data.get("pdf_offset", 0)
        chapters = book_data.get("chapters", [])
        
        needs_fix = False
        for chapter in chapters:
            # Check if chapter has the old format (missing chpstpage/chpendpage)
            if "chpstpage" not in chapter or "chpendpage" not in chapter:
                needs_fix = True
                break
        
        if not needs_fix:
            print(f"\n✓ {book_key}: Already in correct format")
            continue
        
        print(f"\n🔧 Fixing {book_key}...")
        print(f"   PDF offset: {pdf_offset}")
        print(f"   Chapters: {len(chapters)}")
        
        # Fix each chapter
        for chapter in chapters:
            # The current pdf_startpg/pdf_endpg actually contain CHAPTER pages
            # We need to convert them to PDF pages and calculate chapter pages
            
            if "chpstpage" in chapter and "chpendpage" in chapter:
                # Already has the new format
                continue
            
            # Get the CHAPTER pages (currently stored as pdf_startpg/endpg)
            chp_start = chapter.get("pdf_startpg")
            chp_end = chapter.get("pdf_endpg")
            
            if chp_start is None or chp_end is None:
                print(f"   ⚠️  Skipping {chapter.get('chapter_name')}: missing page numbers")
                continue
            
            # Calculate the CORRECT PDF pages
            pdf_start = chp_start + pdf_offset
            pdf_end = chp_end + pdf_offset
            
            # Update the chapter with correct values
            chapter["pdf_startpg"] = pdf_start
            chapter["pdf_endpg"] = pdf_end
            chapter["chpstpage"] = chp_start
            chapter["chpendpage"] = chp_end
            
            print(f"   ✓ {chapter['chapter_name']}")
            print(f"      OLD: pdf_startpg={chp_start} (was actually chapter page)")
            print(f"      NEW: pdf_startpg={pdf_start}, chpstpage={chp_start}")
        
        fixed_count += 1
    
    # Save the fixed cache
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ FIXED {fixed_count} books")
    print(f"✅ Updated cache saved to {CACHE_PATH}")
    print(f"✅ Backup saved to {BACKUP_PATH}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    fix_cache()

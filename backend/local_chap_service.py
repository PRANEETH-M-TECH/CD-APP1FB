import json
import os
from typing import List, Dict, Optional

CHAPTER_CACHE_PATH = "chapterdata/chapters_cache.json"

def get_chapters(class_name: str, subject: str) -> List[Dict]:
    """
    Retrieves a list of chapters for a given class and subject from the local JSON cache.
    """
    try:
        with open(CHAPTER_CACHE_PATH, "r") as f:
            cache = json.load(f)
        
        cache_key = f"{class_name}_{subject.lower()}"
        if cache_key in cache:
            return cache[cache_key].get("chapters", [])
        else:
            return []
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def get_book_uuid(class_name: str, subject: str) -> Optional[str]:
    """
    Retrieves the book_uuid for a given class and subject from the local JSON cache.
    """
    try:
        with open(CHAPTER_CACHE_PATH, "r") as f:
            cache = json.load(f)
        
        cache_key = f"{class_name}_{subject.lower()}"
        if cache_key in cache:
            return cache[cache_key].get("book_uuid")
        else:
            return None
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def get_books(class_name: Optional[str] = None, subject: Optional[str] = None) -> List[Dict]:
    """
    Retrieves a list of books from the local JSON cache, with optional filtering.
    """
    books = []
    try:
        with open(CHAPTER_CACHE_PATH, "r") as f:
            cache = json.load(f)
        
        for key, value in cache.items():
            # key is like "8_social"
            parts = key.split('_')
            if len(parts) != 2:
                continue
            
            cached_class_name = parts[0]
            cached_subject = parts[1]

            if class_name and class_name != cached_class_name:
                continue
            if subject and subject.lower() != cached_subject.lower():
                continue

            books.append({
                "id": value.get("book_uuid"),
                "class_name": cached_class_name,
                "subject": cached_subject,
                "filename": value.get("filename")
            })
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    return books

def get_summary(class_name: str, subject: str, chapter_name: str) -> Optional[str]:
    """
    Retrieves the summary for a specific chapter from the local summary file.
    """
    try:
        summary_dir = "summary"
        summary_filename = f"{class_name}_{subject.lower()}.json"
        summary_filepath = os.path.join(summary_dir, summary_filename)

        with open(summary_filepath, "r") as f:
            data = json.load(f)
        
        for chapter in data.get("chapters", []):
            if chapter.get("chapter_name") == chapter_name:
                return chapter.get("summary")
        
        return None
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_book_details(class_name: str, subject: str, book_uuid: str, filename: str, chapters: List[Dict]):
    """
    Saves or updates book details in the local JSON cache.
    """
    try:
        with open(CHAPTER_CACHE_PATH, "r") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    cache_key = f"{class_name}_{subject.lower()}"
    
    # Create or update the entry
    if cache_key not in cache:
        cache[cache_key] = {}
        
    cache[cache_key]["class_name"] = class_name
    cache[cache_key]["subject"] = subject
    cache[cache_key]["book_uuid"] = book_uuid
    cache[cache_key]["filename"] = filename
    cache[cache_key]["chapters"] = chapters

    with open(CHAPTER_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

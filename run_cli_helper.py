#!/usr/bin/env python3
import sys
import json
import urllib.request
import urllib.error

# Force UTF-8 stdout encoding to avoid Windows console crashes with special characters
sys.stdout.reconfigure(encoding='utf-8')

BACKEND_URL = "http://localhost:8000"

def get_books():
    print(f"Fetching books from {BACKEND_URL}/api/books...")
    try:
        req = urllib.request.Request(f"{BACKEND_URL}/api/books")
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Could not connect to the backend server at {BACKEND_URL}.")
        print("Please make sure your FastAPI backend application is running in another terminal (e.g. uvicorn backend.app.main:app --reload).\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch books: {e}")
        return []

def main():
    print("==================================================")
    print("      VISUAL LEARNING STORYBOARD CLI GENERATOR    ")
    print("==================================================")

    books = get_books()
    
    selected_book = None
    if books:
        print("\nAvailable Books in Database:")
        for idx, book in enumerate(books):
            book_id = book.get("book_uuid", book.get("id", "Unknown"))
            title = book.get("title", book.get("book_name", "Untitled"))
            class_name = book.get("class_name", "Unknown Class")
            subject = book.get("subject", "Unknown Subject")
            print(f"[{idx + 1}] Title: {title}")
            print(f"    Class: {class_name} | Subject: {subject}")
            print(f"    UUID: {book_id}")
            print("-" * 50)
            
        while True:
            choice = input(f"\nSelect a book number (1-{len(books)}): ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(books):
                    selected_book = books[choice_idx]
                    break
            except ValueError:
                pass
            print("Invalid selection. Please enter a valid number.")
    else:
        print("\n[WARNING] No books found in the database. Using fallback book parameters.")
        fallback_uuid = input("Enter standard Book UUID (or press Enter for default 'fallback_book_uuid'): ").strip()
        selected_book = {
            "book_uuid": fallback_uuid if fallback_uuid else "fallback_book_uuid",
            "class_name": "Class 9",
            "subject": "Science"
        }

    book_uuid = selected_book.get("book_uuid", selected_book.get("id"))
    class_name = selected_book.get("class_name")
    subject = selected_book.get("subject")

    print(f"\nConfigured parameters:")
    print(f"  - Book UUID: {book_uuid}")
    print(f"  - Class: {class_name}")
    print(f"  - Subject: {subject}")

    query = input("\nEnter your storyboard topic query (e.g. 'explain structure of neuron'): ").strip()
    while not query:
        query = input("Query cannot be empty. Enter topic query: ").strip()

    payload = {
        "query": query,
        "book_uuid": book_uuid,
        "class_name": class_name,
        "subject": subject
    }

    print(f"\nGenerating storyboard for: '{query}'...")
    
    req = urllib.request.Request(
        f"{BACKEND_URL}/api/visual_learning",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                if line_str.startswith('data:'):
                    data_content = line_str[5:].strip()
                    if data_content == '[DONE]':
                        print("\nFinished stream.")
                        break
                    
                    try:
                        event = json.loads(data_content)
                        event_type = event.get("type")
                        
                        if event_type == "progress":
                            step_msg = event.get("message", "")
                            status = event.get("status", "")
                            if status == "in_progress":
                                print(f" [*] {step_msg}")
                            elif status == "complete":
                                print(f" [OK] {step_msg}")
                        elif event_type == "lesson_ready":
                            lesson = event.get("lesson", {})
                            lesson_id = lesson.get("lesson_id", "")
                            title = lesson.get("lesson_title", "")
                            scenes_count = len(lesson.get("scenes", []))
                            print("\n==============================================")
                            print("         STORYBOARD GENERATION SUCCESS!       ")
                            print("==============================================")
                            print(f" Lesson ID: {lesson_id}")
                            print(f" Title: {title}")
                            print(f" Scenes: {scenes_count}")
                            print("==============================================")
                            print(f"\nSuccess! The storyboard has been generated and saved.")
                            print(f"Go to 'remotion_test_app' and run:")
                            print(f"  node run-storyboard.js")
                            print(f"to preview or render the video.")
                        elif event_type == "error":
                            print(f"\n[ERROR] Generation failed: {event.get('message')}")
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as e:
        print(f"\n[HTTP ERROR] Backend responded with status {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

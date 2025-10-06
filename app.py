import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from qrdant import (
    get_client,
    get_chapters,
    get_chunks_by_chapter,
    semantic_search,
    EMBEDDING_MODEL,
    COLLECTION_NAME
)
from sentence_transformers import SentenceTransformer

# -------------------------------
# INITIALIZATION
# -------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL_NAME = "models/gemini-pro-latest"

QDRANT_CLIENT = get_client()
EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)

print("✅ CHADUVU-GURU Terminal Version Initialized!\n")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def list_chapters_with_pages(textbook_id):
    chapters = get_chapters(QDRANT_CLIENT, textbook_id)
    chapter_info = []
    for chapter in chapters:
        chunks = get_chunks_by_chapter(QDRANT_CLIENT, textbook_id, chapter)
        if chunks:
            start_page = min(chunk['start_page'] for chunk in chunks if 'start_page' in chunk)
            end_page = max(chunk['end_page'] for chunk in chunks if 'end_page' in chunk)
        else:
            start_page, end_page = None, None
        chapter_info.append({"chapter_name": chapter, "start_page": start_page, "end_page": end_page})
    return chapter_info

def summarize_chapter(textbook_id, chapter_name):
    chunks = get_chunks_by_chapter(QDRANT_CLIENT, textbook_id, chapter_name)
    if not chunks:
        return "No content found for this chapter."
    full_text = "\n\n".join([chunk['text'] for chunk in chunks])
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    prompt = f"Summarize the following chapter concisely:\n\n{full_text[:10000]}"
    response = model.generate_content(prompt)
    return response.text

def answer_query(query, textbook_id):
    top_k = 5
    results = semantic_search(QDRANT_CLIENT, EMBEDDING_MODEL_INSTANCE, query, textbook_id, top_k=top_k)
    if not results:
        return "No relevant information found in the textbook."
    context = "\n\n".join(results)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    prompt = f"Based on the following textbook context, answer the question:\n\nQuestion: {query}\n\nContext:\n{context}"
    response = model.generate_content(prompt)
    return response.text

def search_keyword(keyword, textbook_id):
    return answer_query(keyword, textbook_id)

def load_books():
    response, _ = QDRANT_CLIENT.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=["textbook_uuid"]
    )
    return {p.payload['textbook_uuid']: f"Book-{p.payload['textbook_uuid'][:8]}" for p in response}

def select_textbook(books):
    print("\nAvailable textbooks:")
    for idx, (tb_id, name) in enumerate(books.items(), start=1):
        print(f"{idx}. {name} ({tb_id})")
    while True:
        try:
            selection = int(input("\nSelect a textbook by number: "))
            if 1 <= selection <= len(books):
                textbook_id = list(books.keys())[selection - 1]
                print(f"\nYou selected: {books[textbook_id]}\n")
                return textbook_id
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def load_chapters_for_textbook(textbook_id):
    chapters_info = list_chapters_with_pages(textbook_id)
    chapter_dict = {str(idx + 1): ch['chapter_name'] for idx, ch in enumerate(chapters_info)}
    return chapters_info, chapter_dict

# -------------------------------
# MAIN PROGRAM
# -------------------------------
books = load_books()
textbook_id = select_textbook(books)
chapters_info, chapter_dict = load_chapters_for_textbook(textbook_id)
query_history = []

def show_menu():
    print("\n======= CHADUVU-GURU MENU =======")
    print("1. List chapters")
    print("2. Summarize chapter")
    print("3. Keyword search")
    print("4. Ask conceptual question")
    print("5. Switch textbook")
    print("6. Show query history")
    print("7. Exit")
    print("==================================")

while True:
    show_menu()
    choice = input("Select an option (1-7): ").strip()

    if choice == "1":
        chapters_info, chapter_dict = load_chapters_for_textbook(textbook_id)
        print("\nChapters:")
        for idx, ch in enumerate(chapters_info, 1):
            sp = ch['start_page'] if ch['start_page'] is not None else "?"
            ep = ch['end_page'] if ch['end_page'] is not None else "?"
            print(f"{idx}. {ch['chapter_name']} (Pages {sp} - {ep})")

    elif choice == "2":
        chapters_info, chapter_dict = load_chapters_for_textbook(textbook_id)
        print("\nSelect chapter to summarize:")
        for idx, ch in enumerate(chapters_info, 1):
            print(f"{idx}. {ch['chapter_name']}")
        ch_input = input("Enter chapter number or name: ").strip()
        if ch_input.isdigit() and ch_input in chapter_dict:
            chapter_name = chapter_dict[ch_input]
        else:
            chapter_name = ch_input
        summary = summarize_chapter(textbook_id, chapter_name)
        query_history.append(f"summarize chapter {chapter_name}")
        print(f"\nSummary of '{chapter_name}':\n{summary}")

    elif choice == "3":
        keyword = input("Enter keyword to search: ").strip()
        result = search_keyword(keyword, textbook_id)
        query_history.append(f"keyword search {keyword}")
        print(f"\nResult:\n{result}")

    elif choice == "4":
        query = input("Enter your question: ").strip()
        result = answer_query(query, textbook_id)
        query_history.append(query)
        print(f"\nAnswer:\n{result}")

    elif choice == "5":
        textbook_id = select_textbook(books)
        query_history.clear()
        chapters_info, chapter_dict = load_chapters_for_textbook(textbook_id)
        print("Switched textbook. Query history cleared.")

    elif choice == "6":
        print(f"\n--- Query history ({len(query_history)} queries) ---")
        for idx, q in enumerate(query_history, 1):
            print(f"{idx}. {q}")
        print("-----------------------------------------------")

    elif choice == "7":
        print("Exiting CHADUVU-GURU. Bye!")
        break

    else:
        print("Invalid option. Please select 1-7.")

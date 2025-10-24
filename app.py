import os
import shutil
import json
import re
import datetime
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from pypdf import PdfReader

# Load environment variables
load_dotenv()
from qdrant import (
    initialize, # Updated import
    process_and_embed_book,
    get_books,
    get_book_metadata,
    get_chapters_for_book,
    hybrid_search,
    reformulate_and_classify_query,
    generate_answer,
    generate_chapters_from_text,
)

# --- Lifespan Management ---
async def lifespan(app: FastAPI):
    # On startup, initialize all models and database connections
    initialize()
    yield
    # On shutdown (not used here, but good practice)

# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# --- DIRECTORY SETUP ---
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# --- API MODELS ---
class QueryRequest(BaseModel):
    query: str
    book_uuid: str
    # Optional filter for chapter
    chapter: Optional[str] = None

# --- API ENDPOINTS ---
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles PDF file uploads. The file is stored temporarily and its name is returned.
    The frontend will then use this filename in the subsequent call to /api/books.
    """
    # Sanitize filename to prevent directory traversal issues
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOADS_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": safe_filename}

@app.post("/api/books")
async def create_book(
    background_tasks: BackgroundTasks,
    class_name: str = Form(...),
    subject: str = Form(...),
    filename: str = Form(...)
):
    """
    Processes and stores book metadata and content based on the uploaded file.
    """
    pdf_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"Uploaded file not found: {filename}")

    cache_path = "chapters_cache.json"
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
        cached_chapters_data = cache.get(os.path.basename(filename))
        if not cached_chapters_data or not cached_chapters_data.get("chapters"):
            raise HTTPException(status_code=404, detail="Chapter data not found in cache. Please extract chapters first.")
        chapters_list = cached_chapters_data.get("chapters")
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Chapter cache not found or invalid. Please extract chapters first.")

    # Run the long-running task in the background
    background_tasks.add_task(process_and_embed_book, pdf_path, class_name, subject, chapters_list)
    
    # Immediately return a response to the user
    return {"message": "Book processing started in the background. This may take several minutes.", "status": "processing"}

@app.get("/api/books")
async def list_books(class_name: Optional[str] = None, subject: Optional[str] = None):
    """
    Returns a list of available books, optionally filtered by class and subject.
    """
    return get_books(class_name=class_name, subject=subject)

@app.post("/api/query")
async def query_book(request: QueryRequest):
    """
    Performs RAG pipeline to generate an answer.
    """
    # --- Start of New RAG Workflow ---
    # 1. Fetch context for the LLM
    metadata = get_book_metadata(request.book_uuid)
    class_name = metadata.get("class_name")
    subject = metadata.get("subject")
    
    chapters_data = get_chapters_for_book(request.book_uuid)
    chapter_list = [chapter['name'] for chapter in chapters_data]

    # 2. Reformulate and Classify Query using the full context
    processed_query_data = reformulate_and_classify_query(
        query=request.query,
        class_name=class_name,
        subject=subject,
        chapter_list=chapter_list
    )
    
    reformulated_query = processed_query_data.get("reformulated_query", request.query)
    classification = processed_query_data.get("classification", "conceptual")
    keywords = processed_query_data.get("keywords", [])
    conceptual_score = processed_query_data.get("conceptual_score", 0.5)

    print(f"Original Query: '{request.query}'")
    print(f"Processed Query Data: {processed_query_data}")

    # 3. Perform Hybrid Search
    filters = {}
    if request.chapter:
        filters['chapter'] = request.chapter

    search_results, semantic_results, normalized_bm25_results = hybrid_search(
        book_uuid=request.book_uuid,
        query=reformulated_query,
        keywords=keywords,
        conceptual_score=conceptual_score,
        metadata_filters=filters
    )
    
    # 4. Log all steps to ans.txt, clearing previous content
    ans_log_filename = "ans.txt"
    with open(ans_log_filename, "w", encoding="utf-8") as f:
        f.write(f"--- Query Log ---\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Original Query: {request.query}\n")
        f.write(f"Book UUID: {request.book_uuid}\n")
        f.write(f"Class: {class_name}, Subject: {subject}\n")
        f.write(f"\n--- LLM Query Processing ---\n")
        f.write(f"Reformulated Query: {reformulated_query}\n")
        f.write(f"Classification: {classification}\n")
        f.write(f"Conceptual Score: {conceptual_score:.2f}\n")
        f.write(f"Keywords: {', '.join([item['keyword'] for item in keywords])}\n")

        f.write(f"\n--- Semantic Search Results ({len(semantic_results)} Chunks) ---\n\n")
        if not semantic_results:
            f.write("No chunks retrieved from Semantic Search.\n")
        else:
            for i, res in enumerate(semantic_results):
                f.write(f"  {i+1}. Score: {res.score:.4f} | Text: {res.payload['text'].strip()}\n\n")

        f.write(f"\n--- BM25 Keyword Search Results ({len(normalized_bm25_results)} Chunks) ---\n\n")
        if not normalized_bm25_results:
            f.write("No chunks retrieved from BM25 Keyword Search.\n")
        else:
            for i, (score, doc) in enumerate(normalized_bm25_results):
                f.write(f"  {i+1}. Normalized Score: {score:.4f} | Text: {doc['text'].strip()}\n\n")

        f.write(f"\n--- Hybrid Search Results ({len(search_results)} Chunks) ---\n\n")
        if not search_results:
            f.write("No chunks retrieved from Hybrid Search.\n")
        else:
            for i, (score, payload) in enumerate(search_results):
                f.write(f"  {i+1}. Hybrid Score: {score:.4f}\n")
                f.write(f"     Chapter: {payload.get('chapter', 'N/A')}\n")
                f.write(f"     Text: {payload.get('text', '').strip()}\n\n")

        # 5. Generate Final Answer
        if not search_results:
            print("No chunks retrieved from Hybrid Search.")
            answer = "I couldn't find any relevant information to answer your question."
            sources = []
        else:
            context = "\n\n---\n\n".join([payload['text'] for score, payload in search_results])
            book_details = {"class_name": class_name, "subject": subject}
            answer = generate_answer(request.query, book_details, context)
            sources = [payload for score, payload in search_results]
        
        f.write(f"\n--- Generated Answer ---\n")
        f.write(answer)
        f.write(f"\n--- End Log ---\n\n")

    return {"answer": answer, "sources": sources}

@app.get("/api/list-chapters")
async def list_chapters(class_name: str, subject: str):
    """
    Returns a sorted list of chapters for a given book, using a cache.
    """
    cache_path = "chapters_cache.json"
    cache_key = f"{class_name}_{subject}"

    # 1. Check cache first
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
        if cache_key in cache:
            print(f"Cache hit for {cache_key}. Returning cached chapters.")
            return {"chapters": cache[cache_key]}
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    # 2. If not in cache, get from database
    print(f"Cache miss for {cache_key}. Fetching chapters from database.")
    books = get_books(class_name=class_name, subject=subject)
    if not books:
        raise HTTPException(status_code=404, detail="Book not found.")

    book_uuid = books[0]['id']
    chapters = get_chapters_for_book(book_uuid)

    # 3. Save to cache
    cache[cache_key] = chapters
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    return {"chapters": chapters}

def extract_chapters_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extracts chapters from a PDF using an LLM-only approach.
    """
    print("Extracting chapters using LLM-only approach.")
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        pdf_pages_data = []
        for i in range(num_pages):
            text = reader.pages[i].extract_text() or ""
            pdf_pages_data.append({"pdf_page": i + 1, "text": text})

        with open("chap_extraction.json", "w", encoding="utf-8") as f:
            json.dump(pdf_pages_data, f, indent=2)
        
        print("Extracted text saved to chap_extraction.json")

        try:
            llm_response_str = generate_chapters_from_text("chap_extraction.json")
        except Exception as e:
            print(f"Error calling generate_chapters_from_text: {e}")
            raise HTTPException(status_code=500, detail=f"AI model failed to generate chapters: {e}")

        chapters_data = json.loads(llm_response_str)
        return chapters_data

    except json.JSONDecodeError:
        print(f"Failed to parse JSON from LLM response: {llm_response_str}")
        raise HTTPException(status_code=500, detail="Failed to parse chapter data from the AI model.")
    except Exception as e:
        print(f"An error occurred during PDF processing for LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/extract-chapters")
async def extract_chapters(book_id: str = Query(...)):
    """
    Extracts chapter information from the specified PDF file, using a cache to avoid re-processing.
    """
    try:
        if not book_id:
            raise HTTPException(status_code=400, detail="book_id is required.")

        safe_filename = os.path.basename(book_id)
        pdf_path = os.path.join(UPLOADS_DIR, safe_filename)
        cache_path = "chapters_cache.json"

        # 1. Check cache first
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            if safe_filename in cache:
                print(f"Cache hit for {safe_filename}. Returning cached chapters.")
                return JSONResponse(content=cache[safe_filename])
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}

        # 2. If not in cache, process the PDF
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {safe_filename}")

        print(f"Cache miss for {safe_filename}. Extracting chapters from PDF.")
        chapters = extract_chapters_from_pdf(pdf_path)
        
        # 3. Save to cache
        cache[safe_filename] = chapters
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
            
        return JSONResponse(content=chapters)
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions directly
    except Exception as e:
        print(f"Error in extract_chapters endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract chapters: {e}")


# --- STATIC FILE SERVING ---
# Mount the 'public' directory to serve HTML, CSS, JS
app.mount("/static", StaticFiles(directory="public"), name="static")
# Mount the 'uploads' directory to serve uploaded PDF files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def read_root():
    return FileResponse('public/index.html')

@app.get("/admin")
async def admin_page():
    return FileResponse('public/admin.html')

@app.get("/user")
async def user_page():
    return FileResponse('public/user.html')

@app.get("/chapters")
async def chapters_page():
    return FileResponse('public/chapters.html')
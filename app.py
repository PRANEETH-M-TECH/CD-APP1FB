import os
import shutil
import json
import re
import datetime
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
from qrdant import (
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
    background_tasks: BackgroundTasks, # Add this
    class_name: str = Form(...),
    subject: str = Form(...),
    chapters: str = Form(...),      # JSON string of chapter metadata
    filename: str = Form(...)       # Filename from the /api/upload response
):
    """
    Processes and stores book metadata and content based on the uploaded file.
    """
    try:
        chapters_list = json.loads(chapters)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for chapters.")

    pdf_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"Uploaded file not found: {filename}")

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

    print(f"Original Query: '{request.query}'")
    print(f"Processed Query Data: {processed_query_data}")

    # 3. Perform Hybrid Search
    filters = {}
    if request.chapter:
        filters['chapter'] = request.chapter

    search_results = hybrid_search(
        book_uuid=request.book_uuid, 
        query=reformulated_query, 
        classification=classification,
        metadata_filters=filters
    )
    
    # 4. Log simplified results to file
    with open("result.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Query Log ---\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Original Query: {request.query}\n")
        f.write(f"Reformulated Query: {reformulated_query}\n")
        f.write(f"Classification: {classification}\n")
        f.write(f"Retrieved {len(search_results)} Chunks (Hybrid Search):\n\n")
        
        for i, (score, payload) in enumerate(search_results):
            f.write(f"  {i+1}. Hybrid Score: {score:.4f}\n")
            f.write(f"     Chapter: {payload.get('chapter', 'N/A')}\n")
            f.write(f"     Text: {payload.get('text', '').strip()}\n\n")
        f.write(f"--- End Log ---\n\n")

    
    if not search_results:
        print("No chunks retrieved from Hybrid Search.")
        return {"answer": "I couldn't find any relevant information to answer your question.", "sources": []}

    # 5. Generate Final Answer
    context = " ".join([payload['text'] for score, payload in search_results])
    answer = generate_answer(reformulated_query, context, class_name)
    
    sources = [payload for score, payload in search_results]
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
        
        # Create a text sample: first 25 pages + last 5 pages
        text_sample = ""
        for i in range(min(25, num_pages)):
            text_sample += reader.pages[i].extract_text() or ""
        
        if num_pages > 30:
            text_sample += "\n\n... (content from middle of the book omitted) ...\n\n"
            for i in range(num_pages - 5, num_pages):
                text_sample += reader.pages[i].extract_text() or ""

        text_sample += f"\n\n--- End of Sample ---\nTotal pages in book: {num_pages}"

        if not text_sample.strip():
            print("PDF text sample is empty. Cannot extract chapters.")
            return []

        llm_response_str = generate_chapters_from_text(text_sample)
        response_data = json.loads(llm_response_str)
        chapters = response_data.get("chapters", [])
        
        if not isinstance(chapters, list):
            print(f"LLM returned an unexpected data type: {type(chapters)}")
            return []

        # Rename 'chapter_name' to 'title' to match frontend expectations
        for chapter in chapters:
            if 'chapter_name' in chapter:
                chapter['title'] = chapter.pop('chapter_name')

        return chapters

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
    try:
        chapters = extract_chapters_from_pdf(pdf_path)
        
        # 3. Save to cache
        cache[safe_filename] = chapters
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
            
        return JSONResponse(content=chapters)
    except Exception as e:
        print(f"Error extracting chapters: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract chapters from the PDF.")


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
import os
import shutil
import json
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
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
    semantic_search,
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

    # Process the book using the new logic in qrdant.py
    result = process_and_embed_book(pdf_path, class_name, subject, chapters_list)
    
    return result

@app.get("/api/books")
async def list_books(class_name: Optional[str] = None, subject: Optional[str] = None):
    """
    Returns a list of available books, optionally filtered by class and subject.
    """
    return get_books(class_name=class_name, subject=subject)

@app.post("/api/query")
async def query_book(request: QueryRequest):
    """
    Performs semantic search and generates an answer.
    Optionally filters by chapter.
    """
    # Prepare metadata filters
    filters = {}
    if request.chapter:
        filters['chapter'] = request.chapter

    # 1. Perform semantic search to get relevant context
    search_results = semantic_search(request.book_uuid, request.query, metadata_filters=filters)
    
    print(f"Retrieved {len(search_results)} chunks from Qdrant.")
    
    if not search_results:
        print("No chunks retrieved, returning empty answer.")
        return {"answer": "I couldn't find any relevant information in the selected book to answer your question.", "sources": []}

    # All retrieved chunks are used as context for the LLM in this current implementation
    print(f"Using {len(search_results)} chunks as context for the LLM.")

    # 2. Combine context and generate a final answer
    context = " ".join([result['text'] for result in search_results])
    answer = generate_answer(request.query, context)
    
    return {"answer": answer, "sources": search_results}

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
    Extracts chapter information from the specified PDF file.
    """
    if not book_id:
        raise HTTPException(status_code=400, detail="book_id is required.")

    # Sanitize filename
    safe_filename = os.path.basename(book_id)
    pdf_path = os.path.join(UPLOADS_DIR, safe_filename)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {safe_filename}")

    try:
        chapters = extract_chapters_from_pdf(pdf_path)
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
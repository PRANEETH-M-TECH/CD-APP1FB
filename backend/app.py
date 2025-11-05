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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio

# Load environment variables
load_dotenv()
from .qdrant import (
    initialize, # Updated import
    log_query_details,
    process_and_embed_book,
    get_books,
    get_book_metadata,
    get_chapters_for_book,
    hybrid_search,
    reformulate_and_classify_query,

    generate_answer,
    generate_chapters_from_text,
    generate_teacher_explanation,
)

# --- Lifespan Management ---
@asynccontextmanager
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

    cache_path = "chapterdata/chapters_cache.json"
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
        cache_key = f"{class_name}_{subject}"
        cached_chapters_data = cache.get(cache_key)
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

@app.get("/api/query")
async def query_book(query: str, book_uuid: str, chapter: Optional[str] = None):
    """
    Performs RAG pipeline to generate an answer and streams the response.
    """
    print(f"--- Raw Query: {query} ---")
    async def stream_generator():
        # --- Start of New RAG Workflow ---
        # 1. Fetch context for the LLM
        metadata = get_book_metadata(book_uuid)
        class_name = metadata.get("class_name")
        subject = metadata.get("subject")
        
        chapters_data = get_chapters_for_book(book_uuid)
        chapter_list = [chapter['chapter_name'] for chapter in chapters_data]

        # 2. Reformulate and Classify Query using the full context
        processed_query_data = reformulate_and_classify_query(
            query=query,
            class_name=class_name,
            subject=subject,
            chapter_list=chapter_list
        )
        
        reformulated_query = processed_query_data.get("reformulated_query", query)
        print(f"--- Reformulated Query: {reformulated_query} ---")
        classification = processed_query_data.get("classification", "conceptual")
        keywords = processed_query_data.get("keywords", [])
        conceptual_score = processed_query_data.get("conceptual_score", 0.5)

        # 3. Perform Hybrid Search
        filters = {}
        if chapter:
            filters['chapter'] = chapter

        search_results, semantic_results, normalized_bm25_results = hybrid_search(
            book_uuid=book_uuid,
            query=reformulated_query,
            keywords=keywords,
            conceptual_score=conceptual_score,
            metadata_filters=filters
        )
        
        # 4. Generate Final Answer
        if not search_results:
            answer = {"display_text": "I couldn't find any relevant information to answer your question.", "read_text": "I couldn't find any relevant information to answer your question."}
            yield f"data: {json.dumps(answer)}\n\n"
        else:
            context = "\n\n---\n\n".join([payload['text'] for score, payload in search_results])
            book_details = {"class_name": class_name, "subject": subject}
            display_text_sent = False
            read_text_buffer = ""
            in_read_text = False

            for chunk in generate_answer(query, book_details, context):
                if "[READ_TEXT_START]" in chunk:
                    parts = chunk.split("[READ_TEXT_START]")
                    display_chunk = parts[0]
                    if display_chunk:
                        yield f"data: {json.dumps({'display_text': display_chunk, 'read_text': ''})}\n\n"
                    
                    in_read_text = True
                    read_text_buffer += parts[1]
                elif in_read_text:
                    read_text_buffer += chunk
                else:
                    yield f"data: {json.dumps({'display_text': chunk, 'read_text': ''})}\n\n"

            if read_text_buffer:
                yield f"data: {json.dumps({'display_text': '', 'read_text': read_text_buffer})}\n\n"

            print(f"--- Generated Answer: {read_text_buffer} ---")
            log_query_details(query, {"id": book_uuid, "class_name": class_name, "subject": subject}, processed_query_data, search_results, read_text_buffer)
        yield f"data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")



@app.get("/api/list-chapters")
async def list_chapters(class_name: str, subject: str):
    """
    Returns a sorted list of chapters for a given book, using a cache.
    """
    cache_path = "chapterdata/chapters_cache.json"
    cache_key = f"{class_name}_{subject.lower()}"

    print(f"--- API: /api/list-chapters called for Class: {class_name}, Subject: {subject} (Cache Key: {cache_key}) ---")

    # 1. Check cache first
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
        if cache_key in cache:
            print(f"--- Cache HIT for {cache_key}. Returning {len(cache[cache_key]['chapters'])} chapters from cache. ---")
            return {"chapters": cache[cache_key]["chapters"]}
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"--- Cache MISS or invalid cache file for {cache_key}. Attempting to retrieve from database. ---")
        cache = {} # Ensure cache is empty if file not found or invalid

    # 2. If not in cache, get from database
    books = get_books(class_name=class_name, subject=subject)
    if not books:
        raise HTTPException(status_code=404, detail="Book not found.")

    book_uuid = books[0]['id']
    chapters = get_chapters_for_book(book_uuid)
    print(f"--- Retrieved {len(chapters)} chapters from database for book UUID: {book_uuid}. ---")

    # This endpoint should not write to the cache. The cache is managed by /extract-chapters.
    # If the data is not in cache, it means /extract-chapters hasn't been run for this book.
    # The frontend should handle this by prompting the user to extract chapters first.

    return {"chapters": chapters}


class SummaryRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str

@app.post("/api/summarize")
async def get_summary(request: SummaryRequest):
    """
    Generates a teacher-like explanation for a specific chapter of a book.
    """
    class_name = request.class_name
    subject = request.subject
    chapter_name = request.chapter_name

    summary_filename = f"{subject.lower()}{class_name.replace(' ', '')}.json"
    summary_filepath = os.path.join("..", "summary", summary_filename)

    if not os.path.exists(summary_filepath):
        raise HTTPException(status_code=404, detail="Summary file not found for this book.")

    try:
        with open(summary_filepath, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Summary data not found or invalid.")

    chapter_summary = None
    for chapter in summary_data.get("chapters", []):
        if chapter.get("chapter_name") == chapter_name:
            chapter_summary = chapter.get("summary")
            break
    
    if chapter_summary is None:
        raise HTTPException(status_code=404, detail="Summary not found for this chapter.")

    # Generate the detailed explanation using the new function
    explanation = generate_teacher_explanation(
        class_name=class_name,
        subject=subject,
        chapter_name=chapter_name,
        summary_text=chapter_summary
    )
    
    return {"summary": explanation}

def extract_chapters_from_pdf(pdf_path: str) -> Dict:
    """
    Extracts chapters from a PDF using an LLM-only approach, calculates chapter-specific page numbers,
    and includes pdf_offset.
    """
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        pages_to_extract_indices = set()

        # Add first 20 pages
        for i in range(min(20, num_pages)):
            pages_to_extract_indices.add(i)

        # Add last 5 pages
        for i in range(max(0, num_pages - 5), num_pages):
            pages_to_extract_indices.add(i)
        
        sorted_page_indices = sorted(list(pages_to_extract_indices))

        pdf_pages_data = []
        for i in sorted_page_indices:
            text = reader.pages[i].extract_text() or ""
            pdf_pages_data.append({"pdf_page": i + 1, "text": text})

        with open("chapterdata/chap_extraction.json", "w", encoding="utf-8") as f:
            json.dump(pdf_pages_data, f, indent=2)
        
        try:
            llm_response_str = generate_chapters_from_text("chapterdata/chap_extraction.json")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI model failed to generate chapters: {e}")

        chapters_data_from_llm = json.loads(llm_response_str)
        
        if not isinstance(chapters_data_from_llm, dict) or "chapters" not in chapters_data_from_llm:
            raise HTTPException(status_code=500, detail="LLM response is not in the expected format (missing 'chapters' key).")

        llm_chapters_list = chapters_data_from_llm.get("chapters")
        llm_pdf_offset = chapters_data_from_llm.get("pdf_offset")

        if not llm_chapters_list:
            raise HTTPException(status_code=500, detail="AI model returned empty chapter list.")
        if llm_pdf_offset is None:
            raise HTTPException(status_code=500, detail="LLM response missing 'pdf_offset'.")

        # The pdf_offset is now directly from the LLM response
        pdf_offset = llm_pdf_offset

        processed_chapters = []
        for chapter in llm_chapters_list: # Iterate over the actual list of chapters
            pdf_startpg = chapter.get("pdf_startpg")
            pdf_endpg = chapter.get("pdf_endpg")

            # If pdf_startpg or pdf_endpg are missing, try to use start_page and end_page
            if pdf_startpg is None:
                pdf_startpg = chapter.get("start_page")
            if pdf_endpg is None:
                pdf_endpg = chapter.get("end_page")

            if pdf_startpg is None or pdf_endpg is None:
                print(f"Warning: Chapter '{chapter.get('chapter_name', 'Unknown')}' is missing start or end page numbers. Appending as is.")
                processed_chapters.append({
                    "chapter_name": chapter.get("chapter_name"),
                    "pdf_startpg": pdf_startpg,
                    "pdf_endpg": pdf_endpg,
                    "chpstpage": None, # Assign None for consistency
                    "chpendpage": None   # Assign None for consistency
                })
                continue

            chpstpage = pdf_startpg - pdf_offset
            chpendpage = pdf_endpg - pdf_offset
            
            processed_chapters.append({
                "chapter_name": chapter.get("chapter_name"),
                "pdf_startpg": pdf_startpg,
                "pdf_endpg": pdf_endpg,
                "chpstpage": chpstpage,
                "chpendpage": chpendpage
            })
        
        return {"pdf_offset": pdf_offset, "chapters": processed_chapters}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse chapter data from the AI model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

@app.post("/extract-chapters")
async def extract_chapters(
    book_id: str = Query(...),
    class_name: str = Query(...),
    subject: str = Query(...)
):
    """
    Extracts chapter information from the specified PDF file, using a cache to avoid re-processing.
    """
    try:
        if not book_id:
            raise HTTPException(status_code=400, detail="book_id is required.")

        safe_filename = os.path.basename(book_id)
        pdf_path = os.path.join(UPLOADS_DIR, safe_filename)
        cache_path = "chapterdata/chapters_cache.json"
        cache_key = f"{class_name}_{subject.lower()}" # Define cache_key before the try block

        # 1. Check cache first
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            if cache_key in cache: # Check with new cache key
                return JSONResponse(content=cache[cache_key])
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}

        # 2. If not in cache, process the PDF
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {safe_filename}")

        # extract_chapters_from_pdf now returns a dict with "pdf_offset" and "chapters"
        extracted_data = extract_chapters_from_pdf(pdf_path) 
        
        # 3. Save to cache
        cache[cache_key] = extracted_data # Save with new cache key
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
            
        return JSONResponse(content=extracted_data) # Return the entire dict
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions directly
    except Exception as e:
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
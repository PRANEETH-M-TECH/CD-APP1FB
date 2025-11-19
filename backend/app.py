import os
import shutil
import json
import re
import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from .conversation import conversation_manager
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
from pypdf import PdfReader

# Load environment variables
load_dotenv()

import logging
logger = logging.getLogger(__name__)

# --- Configure Gemini ---
import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "❌ No Gemini API key found in environment (.env). "
        "Please check GOOGLE_API_KEY or GEMINI_API_KEY."
    )

genai.configure(api_key=api_key)
print("✅ Google Gemini configured successfully.")
from . import qdrant
from . import local_chap_service
from . import firestore_service

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, initialize all models and database connections
    qdrant.initialize()
    yield
    # On shutdown (not used here, but good practice)

# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

from .firebase.firebase_init import db, bucket

# --- DIRECTORY SETUP ---
# Make the path absolute to avoid CWD issues
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# --- API MODELS ---
class QueryRequest(BaseModel):
    query: str
    class_name: str
    subject: str
    book_uuid: str

class BookCreateRequest(BaseModel):
    class_name: str
    subject: str
    filename: str
    chapters: List[Dict]

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
async def create_book_and_process(
    background_tasks: BackgroundTasks,
    book_data: BookCreateRequest
):
    """
    Starts the background processing task for a book.
    """
    logger.info(f"Received request to process and save book with data: {book_data.dict()}")
    try:
        class_name = book_data.class_name
        subject = book_data.subject
        filename = book_data.filename
        chapters = book_data.chapters

        logger.info(f"Received request to process and save book: {filename}")
        pdf_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"Uploaded file not found: {filename}")

        book_uuid = qdrant.get_book_uuid(pdf_path)
        
        # Save the book details to the cache
        local_chap_service.save_book_details(class_name, subject, book_uuid, filename, chapters)

        # Start the background processing task
        logger.info(f"Starting background processing for book {book_uuid}")
        background_tasks.add_task(process_book_in_background, book_uuid, pdf_path, class_name, subject, chapters)
        
        return {"message": "Book processing started in the background.", "status": "processing", "book_id": book_uuid}
    except Exception as e:
        logger.error(f"Error processing book creation request: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Error processing book creation request: {e}")

from qdrant_client import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

async def process_book_in_background(book_uuid: str, pdf_path: str, class_name: str, subject: str, chapters: List[Dict]):
    """
    Processes the book in the background, creates summaries, and saves to databases.
    """
    print("\n[ADMIN] Starting book processing pipeline...\n")
    logger.info(f"BACKGROUND TASK STARTED for book {book_uuid}")

    try:
        # Initialize services
        qdrant.initialize()
        reader = PdfReader(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chapters_to_process = chapters
        if not chapters_to_process:
            raise ValueError("No confirmed chapters found to process.")

        print(f"[ADMIN] Total chapters to process: {len(chapters_to_process)}")

        all_chapters_with_summaries = []

        # Steps 1 & 3: Generate Summaries and Upload Chunks to Qdrant
        for i, chapter_data in enumerate(chapters_to_process):
            chapter_name = chapter_data['chapter_name']
            print(f"[ADMIN] Processing Chapter {i+1}/{len(chapters_from_cache)}: {chapter_name}")

            start_page = chapter_data.get("pdf_startpg")
            end_page = chapter_data.get("pdf_endpg")

            if start_page is None or end_page is None:
                print(f"[WARN] Skipping chapter '{chapter_name}' due to missing page numbers.")
                continue

            # Extract text
            chapter_text = ""
            for page_num in range(start_page - 1, end_page):
                if 0 <= page_num < len(reader.pages):
                    chapter_text += reader.pages[page_num].extract_text() or ""

            # Create and upload chunks to Qdrant
            text_chunks = text_splitter.split_text(chapter_text)
            print(f"  - Split into {len(text_chunks)} chunks.")
            
            points_to_upload = []
            for j, chunk_text in enumerate(text_chunks):
                chunk_id = str(uuid.uuid4())
                qdrant_id = str(uuid.uuid4())
                embedding = qdrant.local_embedder.encode(chunk_text).tolist()
                points_to_upload.append(
                    models.PointStruct(
                        id=qdrant_id,
                        vector=embedding,
                        payload={
                            "book_uuid": book_uuid,
                            "chapter_id": str(i + 1),
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "chapter_name": chapter_name,
                        },
                    )
                )

            if points_to_upload:
                qdrant.qdrant_client.upsert(collection_name="data", points=points_to_upload, wait=True)
                print(f"  - Saved {len(points_to_upload)} chunks to Qdrant.")

            # Generate summary
            summary_text = qdrant.generate_chapter_summary(class_name, subject, chapter_name, text_chunks)
            print(f"  - Generated summary for chapter.")
            
            chapter_summary_data = {
                "chapter_name": chapter_name,
                "summary": summary_text
            }
            all_chapters_with_summaries.append(chapter_summary_data)

        # Step 4: Save single summary document for LLM context
        print("[ADMIN] Saving summary document for LLM.")
        firestore_service.save_summary_document(
            class_name=class_name,
            subject=subject,
            book_uuid=book_uuid,
            chapters=all_chapters_with_summaries
        )

        print("\n[ADMIN] Book processing finished successfully.\n")

    except Exception as e:
        logger.error(f"BACKGROUND TASK FAILED for book {book_uuid}: {e}", exc_info=True)

    logger.info(f"Finished background processing for book {book_uuid}")

@app.get("/api/books")
async def list_books(class_name: Optional[str] = None, subject: Optional[str] = None):
    """
    Returns a list of available books from the local cache, optionally filtered by class and subject.
    """
    return local_chap_service.get_books(class_name=class_name, subject=subject)

import time
from google.cloud import firestore

#############################################################
# FINAL BACKEND PIPELINE (FAST, CACHED, FIRESTORE SUMMARIES)
#############################################################

# 1. SUMMARY CACHE (IN MEMORY)
SUMMARY_CACHE = {}


def load_summary_from_firestore(class_name: str, subject: str):
    """
    Loads summaries/{subject}_{class} from Firestore.
    Caches in memory for FAST access (0ms after first load).
    """
    key = f"{subject.lower()}_{class_name.replace(' ', '')}"

    # Check cached
    if key in SUMMARY_CACHE:
        return SUMMARY_CACHE[key]

    # Fetch from Firestore
    db = firestore.Client()
    doc_ref = db.collection("summaries").document(key)
    doc = doc_ref.get()

    if not doc.exists:
        raise Exception(f"Summary document not found: summaries/{key}")

    data = doc.to_dict()
    SUMMARY_CACHE[key] = data  # cache it

    print(f"[CACHE] Loaded summary → summaries/{key}")

    return data


# 2. JSON Extractor (LLM Output Cleaner)
def extract_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != -1 and end > start:
        return text[start:end]
    return None


# 3. REFORMULATION + CHAPTER RANKING LLM
def reformulate_with_llm(raw_query: str, class_name: str, subject: str, chapters):
    # Convert chapters to JSON string
    chapters_json = json.dumps(chapters, ensure_ascii=False, indent=2)

    # --------------- FINAL PROMPT ---------------
    prompt = f"""
You are an expert ASR-correction and curriculum-aware query processor.

Your tasks:

1) Correct ASR mistakes in the raw query. Only correct errors clearly wrong based on the summaries.

2) Reformulate the corrected query into a descriptive, retrieval-ready form (8–30 words).

3) Extract important keywords (importance >= 0.3).

4) Return conceptual_score (0-1) and classification:
   - "conceptual" if >0.5
   - "factual" otherwise

5) Rank the most relevant chapters using ONLY the provided summaries.
   Return:
   [
     {{
       "chapter_id": str,
       "chapter_name": str,
       "start_page": int,
       "end_page": int,
       "score": float
     }}
   ]

6) STRICT JSON OUTPUT ONLY:
{{
  "reformulated_query": str,
  "normalized_query": str,
  "keywords": [...],
  "conceptual_score": float,
  "classification": str,
  "chapter_ranking": [...]
}}

--------------------------------------

CLASS = "{class_name}"
SUBJECT = "{subject}"
RAW_QUERY = "{raw_query}"

# CHAPTER SUMMARIES:
{chapters_json}

--------------------------------------
Return ONLY the JSON response.
"""

    # LLM Call
    try:
        response = qdrant.generation_model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        print("[LLM ERROR]", e)
        raw = "{}"

    json_block = extract_json_block(raw)

    if not json_block:
        return {
            "reformulated_query": raw_query,
            "normalized_query": raw_query.lower(),
            "keywords": [],
            "classification": "conceptual",
            "conceptual_score": 0.5,
            "chapter_ranking": []
        }

    try:
        parsed = json.loads(json_block)
    except:
        parsed = {
            "reformulated_query": raw_query,
            "normalized_query": raw_query.lower(),
            "keywords": [],
            "classification": "conceptual",
            "conceptual_score": 0.5,
            "chapter_ranking": []
        }

    # classification rule
    cs = parsed.get("conceptual_score", 0.5)
    parsed["classification"] = "conceptual" if cs > 0.5 else "factual"

    return parsed


# 4. QDRANT RETRIEVAL (FILTER BY CHAPTERS)
def retrieve_from_qdrant(query: str, book_uuid: str, chapter_ranking: List[Dict]):
    embedding = qdrant.embed_query(query)

    chapter_ids = [
        ch.get("chapter_id") or ch["chapter_name"].replace(" ", "_")
        for ch in chapter_ranking
    ]

    q_filter = {
        "must": [
            {"key": "book_uuid", "match": {"value": book_uuid}}
        ]
    }

    if chapter_ids:
        q_filter["must"].append({
            "key": "chapter_id",
            "match": {"any": chapter_ids}
        })

    results = qdrant.qdrant_client.search(
        collection_name="data",
        query_vector=embedding,
        limit=5,
        filter=q_filter
    )

    return results


# 5. MAIN ENDPOINT — COMPLETE PIPELINE
@app.get("/api/query", tags=["LLM"])
def query_engine(
    book_uuid: str = Query(...),
    query: str = Query(...),
    class_name: str = Query(...),
    subject: str = Query(...)
):
    start = time.time()

    # Load cached summaries (no Firestore after first load)
    summary_doc = load_summary_from_firestore(class_name, subject)
    chapters = summary_doc["chapters"]

    # Reformulate query + chapter ranking
    reform = reformulate_with_llm(
        raw_query=query,
        class_name=class_name,
        subject=subject,
        chapters=chapters
    )

    # Retrieve context from Qdrant
    qdrant_hits = retrieve_from_qdrant(
        reform["reformulated_query"],
        book_uuid,
        reform["chapter_ranking"]
    )

    context = "\n".join([
        hit.payload.get("text", "") for hit in qdrant_hits
    ])

    # Final answer
    final_prompt = f"""
You are a helpful teacher. Use the context to answer the question clearly:

QUESTION:
{reform['reformulated_query']}

CONTEXT:
{context}

Return only the answer.
"""

    try:
        final_resp = qdrant.generation_model.generate_content(final_prompt)
        answer = final_resp.text.strip()
    except:
        answer = "Sorry, I couldn't generate the answer."

    return {
        "raw_query": query,
        "reformulated_query": reform["reformulated_query"],
        "classification": reform["classification"],
        "chapter_ranking": reform["chapter_ranking"],
        "answer": answer,
        "latency": round(time.time() - start, 2)
    }



@app.get("/api/list-chapters")
async def list_chapters(class_name: str, subject: str):
    """
    Returns a sorted list of chapters for a given book from the local cache.
    """
    chapters = local_chap_service.get_chapters(class_name=class_name, subject=subject)
    if not chapters:
        raise HTTPException(status_code=404, detail="Chapters not found for this book.")
    
    # The chapters from the cache are already sorted.
    
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

    chapter_summary = local_chap_service.get_summary(class_name, subject, chapter_name)
    
    if chapter_summary is None or chapter_summary == "":
        raise HTTPException(status_code=404, detail="Summary not found for this chapter or is being generated.")

    # Generate the detailed explanation using the new function
    explanation = qdrant.generate_teacher_explanation(
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

        # Add first 30 pages
        for i in range(min(30, num_pages)):
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
            llm_response_str = qdrant.generate_chapters_from_text("chapterdata/chap_extraction.json")
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
        book_uuid = qdrant.get_book_uuid(pdf_path)
        extracted_data['book_uuid'] = book_uuid
        extracted_data['filename'] = safe_filename
        
        # 3. Save to cache
        cache[cache_key] = extracted_data # Save with new cache key
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
            
        return JSONResponse(content=extracted_data) # Return the entire dict
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions directly
    except Exception as e:
        logger.error(f"Failed to extract chapters due to an unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract chapters: {e}")


@app.get("/api/clear-qdrant")
async def clear_qdrant_data():
    """
    Clears all data from the Qdrant collection.
    """
    try:
        qdrant.clear_qdrant_collection()
        return {"message": "Qdrant collection cleared and re-initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear Qdrant collection: {e}")


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

@app.websocket("/ws/conversation/{conversation_id}")
async def websocket_conversation(websocket: WebSocket, conversation_id: str, book_uuid: str):
    await conversation_manager.connect(websocket, conversation_id, book_uuid)
    print(f"[App] WebSocket handler started for conversation_id={conversation_id}, book_uuid={book_uuid}")
    try:
        while True:
            data = await websocket.receive_json()
            print(f"[App] Received WS message for {conversation_id}: {str(data)[:200]}")
            
            if data.get("type") == "query":
                # Log that query processing is starting
                print(f"[App] Dispatching 'query' to ConversationManager for {conversation_id}")
                await conversation_manager.process_query(conversation_id, data.get("query", ""))
            elif data.get("type") == "interrupt":
                print(f"[App] Received 'interrupt' for {conversation_id}")
                await conversation_manager.interrupt(conversation_id)
    
    except WebSocketDisconnect:
        print(f"[App] WebSocket disconnected for conversation_id={conversation_id}")
    except Exception as e:
        print(f"[App] WebSocket error for {conversation_id}: {e}")
    finally:
        conversation_manager.disconnect(conversation_id)
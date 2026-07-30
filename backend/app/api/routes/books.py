import os
import shutil
import json
import uuid
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import APIRouter, File, UploadFile, Query, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from qdrant_client import models
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.services.retrieval import local_chap_service
from backend.app.core import firestore_service
from backend.app.core.firebase.firebase_init import db, bucket
from backend.app.services.chat.answer_service import generate_chapter_summary

logger = logging.getLogger(__name__)

router = APIRouter()

# --- DIRECTORY SETUP ---
# backend/app/api/routes/books.py is 4 levels deep:
# 1 level: routes
# 2 levels: api
# 3 levels: app
# 4 levels: backend (CG-DEV/CD-APP1FB/backend)
# 5 levels: CG-DEV/CD-APP1FB (project root)
ROUTES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROUTES_DIR, "..", "..", "..", ".."))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

class BookCreateRequest(BaseModel):
    class_name: str
    subject: str
    filename: str
    chapters: List[Dict]


async def process_book_in_background(book_uuid: str, pdf_path: str, class_name: str, subject: str, chapters: List[Dict]):
    """
    Processes the book in the background, creates summaries, and saves to databases.
    """
    print(f"\n{'='*100}")
    print(f"[PROCESS] ========== BOOK PROCESSING START ==========")
    print(f"[PROCESS] Book: Class {class_name} - {subject.capitalize()}")
    print(f"[PROCESS] UUID: {book_uuid}")
    print(f"[PROCESS] PDF: {os.path.basename(pdf_path)}")
    print(f"[PROCESS] Total Chapters: {len(chapters)}")
    print(f"{'='*100}\n")
    
    logger.info(f"BACKGROUND TASK STARTED for book {book_uuid}")

    try:
        # Initialize services
        print(f"[PROCESS] Initializing services...")
        qdrant.initialize()
        reader = PdfReader(pdf_path)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", " ", ""]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        print(f"[PROCESS] ✓ Services initialized\n")

        chapters_to_process = chapters
        if not chapters_to_process:
            raise ValueError("No confirmed chapters found to process.")

        all_chapters_with_summaries = []

        # Steps 1 & 3: Generate Summaries and Upload Chunks to Qdrant
        for i, chapter_data in enumerate(chapters_to_process):
            chapter_name = chapter_data['chapter_name']
            
            print(f"[PROCESS] ┌─ [{i+1}/{len(chapters_to_process)}] {chapter_name}")

            start_page = chapter_data.get("pdf_startpg")
            end_page = chapter_data.get("pdf_endpg")
            chp_start = chapter_data.get("chpstpage")
            chp_end = chapter_data.get("chpendpage")

            if start_page is None or end_page is None:
                print(f"[PROCESS] │  ✗ Skipping - missing page numbers\n")
                continue
            
            print(f"[PROCESS] │  Pages: PDF {start_page}-{end_page}, Chapter {chp_start}-{chp_end}")
            print(f"[PROCESS] │  Extracting and chunking text from PDF...")

            points_to_upload = []
            chapter_parent_chunks = []

            # Process page by page
            for page_num in range(start_page - 1, end_page):
                if page_num < 0 or page_num >= len(reader.pages):
                    continue
                
                page_text = reader.pages[page_num].extract_text() or ""
                if not page_text.strip():
                    continue

                # Split page text into parent chunks
                parent_chunks = parent_splitter.split_text(page_text)
                for parent_text in parent_chunks:
                    parent_text = parent_text.strip()
                    if not parent_text:
                        continue
                    
                    chapter_parent_chunks.append(parent_text)
                    
                    # Split parent chunk into child chunks
                    child_chunks = child_splitter.split_text(parent_text)
                    for chunk_text in child_chunks:
                        chunk_text = chunk_text.strip()
                        if not chunk_text:
                            continue
                        
                        chunk_id = str(uuid.uuid4())
                        qdrant_id = str(uuid.uuid4())
                        
                        # Generate embedding for child text
                        embedding = qdrant.local_embedder.encode(chunk_text).tolist()
                        
                        # Compute actual printed page
                        current_printed_page = chp_start + (page_num - (start_page - 1)) if chp_start is not None else 1
                        
                        points_to_upload.append(
                            models.PointStruct(
                                id=qdrant_id,
                                vector=embedding,
                                payload={
                                    "book_uuid": book_uuid,
                                    "chapter_id": str(i + 1),
                                    "chunk_id": chunk_id,
                                    "text": chunk_text,
                                    "parent_text": parent_text,
                                    "chapter_name": chapter_name,
                                    "pdf_page": page_num + 1,
                                    "pdf_startpg": start_page,
                                    "pdf_endpg": end_page,
                                    "chpstpage": current_printed_page,
                                    "chpendpage": current_printed_page,
                                },
                            )
                        )

            if points_to_upload:
                print(f"[PROCESS] │  ✓ Saved {len(points_to_upload)} chunks to Qdrant")
                
                # Upload in batches to prevent timeout
                BATCH_SIZE = 50  # Upload 50 points at a time
                total_points = len(points_to_upload)
                
                for batch_start in range(0, total_points, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, total_points)
                    batch = points_to_upload[batch_start:batch_end]
                    
                    print(f"[PROCESS] │  Uploading batch {batch_start+1}-{batch_end} of {total_points}...")
                    
                    # Retry logic for network issues
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            qdrant.client.upsert(
                                collection_name=qdrant.COLLECTION_NAME,
                                points=batch,
                                wait=True
                            )
                            print(f"[PROCESS] │  ✓ Batch uploaded successfully")
                            break  # Success, exit retry loop
                        except Exception as e:
                            if attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                                print(f"[PROCESS] │  ⚠️ Upload failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s...")
                                import time
                                time.sleep(wait_time)
                            else:
                                print(f"[PROCESS] │  ✗ Upload failed after {max_retries} attempts: {e}")
                                raise  # Re-raise after all retries exhausted

            # Generate summary
            print(f"[PROCESS] │  Generating summary with LLM...")
            summary_text = generate_chapter_summary(class_name, subject, chapter_name, chapter_parent_chunks)
            print(f"[PROCESS] │  ✓ Summary generated ({len(summary_text)} chars)")
            
            chapter_summary_data = {
                "sno": i + 1,  # Serial number starting from 1
                "chapter_name": chapter_name,
                "summary": summary_text,
                "pdf_startpg": chapter_data.get("pdf_startpg"),
                "pdf_endpg": chapter_data.get("pdf_endpg"),
                "chpstpage": chapter_data.get("chpstpage"),
                "chpendpage": chapter_data.get("chpendpage"),
            }
            
            # Log what we're saving to Firestore for debugging
            print(f"[PROCESS] │  ✓ Firestore data for chapter {i + 1}:")
            print(f"[PROCESS] │    - sno: {i + 1}")
            print(f"[PROCESS] │    - chapter_name: {chapter_name}")
            print(f"[PROCESS] │    - pdf_startpg: {chapter_data.get('pdf_startpg')}")
            print(f"[PROCESS] │    - pdf_endpg: {chapter_data.get('pdf_endpg')}")
            print(f"[PROCESS] │    - chpstpage: {chapter_data.get('chpstpage')}")
            print(f"[PROCESS] │    - chpendpage: {chapter_data.get('chpendpage')}")
            print(f"[PROCESS] │    - summary_length: {len(summary_text)} chars")
            
            all_chapters_with_summaries.append(chapter_summary_data)
            print(f"[PROCESS] └─ ✓ Chapter complete\n")

        # Step 4: Save single summary document for LLM context
        print(f"[PROCESS] Saving {len(all_chapters_with_summaries)} summaries to Firestore...")
        firestore_service.save_summary_document(
            class_name=class_name,
            subject=subject,
            book_uuid=book_uuid,
            chapters=all_chapters_with_summaries
        )
        print(f"[PROCESS] ✓ Summaries saved to Firestore\n")

        print(f"{'='*100}")
        print(f"[PROCESS] ========== BOOK PROCESSING COMPLETE ==========")
        print(f"[PROCESS] ✓ {len(chapters_to_process)} chapters processed")
        print(f"[PROCESS] ✓ All data saved to Qdrant and Firestore")
        print(f"{'='*100}\n")

    except Exception as e:
        print(f"\n[PROCESS] ✗ ERROR: {e}\n")
        logger.error(f"BACKGROUND TASK FAILED for book {book_uuid}: {e}", exc_info=True)

    logger.info(f"Finished background processing for book {book_uuid}")


@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles PDF file uploads. The file is stored temporarily and its name is returned.
    The frontend will then use this filename in the subsequent call to /api/books.
    """
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOADS_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": safe_filename}


@router.post("/api/books")
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

        # Compute book_uuid based on MD5 checksum
        import hashlib
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as afile:
            buf = afile.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(65536)
        book_uuid = hasher.hexdigest()
        
        # Get PDF offset from cache to calculate PDF pages
        try:
            with open("chapterdata/chapters_cache.json", "r") as f:
                book_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            book_cache = {}
            
        book_key = f"{class_name}_{subject.lower()}"
        pdf_offset = book_cache.get(book_key, {}).get("pdf_offset", 0)
        
        logger.info(f"📖 Book key: {book_key}, PDF offset: {pdf_offset}")
        
        # Calculate PDF pages from chapter pages if needed
        for chapter in chapters:
            if 'chpstpage' in chapter and 'chpendpage' in chapter:
                chapter['pdf_startpg'] = chapter['chpstpage'] + pdf_offset
                chapter['pdf_endpg'] = chapter['chpendpage'] + pdf_offset
                logger.info(f"Calculated PDF pages for {chapter.get('chapter_name')}: "
                           f"chp {chapter['chpstpage']}-{chapter['chpendpage']} -> "
                           f"pdf {chapter['pdf_startpg']}-{chapter['pdf_endpg']}")
        
        # Start the background processing task
        logger.info(f"Starting background processing for book {book_uuid}")
        background_tasks.add_task(process_book_in_background, book_uuid, pdf_path, class_name, subject, chapters)
        
        return {"message": "Book processing started in the background.", "status": "processing", "book_id": book_uuid}
    except Exception as e:
        logger.error(f"Error processing book creation request: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Error processing book creation request: {e}")


@router.get("/api/books")
async def get_books_endpoint(
    class_name: Optional[str] = Query(None),
    subject: Optional[str] = Query(None)
):
    """
    Returns a list of books matching optional class and subject filters from local cache.
    """
    books = local_chap_service.get_books(class_name=class_name, subject=subject)
    return books  # Return the list directly to match script.js expectation


@router.get("/api/subjects")
async def get_subjects_endpoint(
    class_name: Optional[str] = Query(None)
):
    """
    Returns all unique subjects available for a given class formatted as objects with icons and display names.
    """
    from backend.app.core import subject_config
    
    # Parse class number (e.g. "Class 8" or "8" -> 8)
    class_num = 8
    if class_name:
        try:
            clean = "".join(c for c in str(class_name) if c.isdigit())
            class_num = int(clean) if clean else 8
        except:
            class_num = 8
            
    # Get subjects configured for this class
    configured_subjects = subject_config.get_subjects_for_class(class_num)
    
    # Get unique subjects from uploaded books for this class to only show active subjects
    books = local_chap_service.get_books(class_name=class_name)
    uploaded_subjects = set(b["subject"].lower() for b in books)
    
    subjects_list = []
    for sub in configured_subjects:
        # Check if subject is either uploaded or if no books uploaded yet we show all configured ones as default fallback
        if not uploaded_subjects or sub.lower() in uploaded_subjects:
            icon = subject_config.get_subject_icon(sub, class_num)
            display_name = sub.capitalize()
            if sub.lower() == "maths":
                display_name = "Maths"
            subjects_list.append({
                "name": sub,
                "icon": icon,
                "display_name": display_name
            })
            
    return {"subjects": subjects_list}


@router.get("/api/list-chapters")
async def list_chapters(class_name: str, subject: str):
    """
    Returns a sorted list of chapters for a given book from the local cache.
    """
    chapters = local_chap_service.get_chapters(class_name=class_name, subject=subject)
    if not chapters:
        raise HTTPException(status_code=404, detail="Chapters not found for this book.")
    return {"chapters": chapters}


def extract_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != -1 and end > start:
        return text[start:end]
    return None


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

        # Ensure chapterdata folder exists
        os.makedirs("chapterdata", exist_ok=True)
        with open("chapterdata/chap_extraction.json", "w", encoding="utf-8") as f:
            json.dump(pdf_pages_data, f, indent=2)
        
        try:
            from backend.app.services.chat.answer_service import generate_chapters_from_text
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

        pdf_offset = llm_pdf_offset

        processed_chapters = []
        for chapter in llm_chapters_list:
            pdf_startpg = chapter.get("pdf_startpg")
            pdf_endpg = chapter.get("pdf_endpg")

            if pdf_startpg is None:
                pdf_startpg = chapter.get("start_page")
            if pdf_endpg is None:
                pdf_endpg = chapter.get("end_page")

            if pdf_startpg is None or pdf_endpg is None:
                print(f"[WARN] Chapter '{chapter.get('chapter_name', 'Unknown')}' missing page numbers")
                processed_chapters.append({
                    "chapter_name": chapter.get("chapter_name"),
                    "pdf_startpg": pdf_startpg,
                    "pdf_endpg": pdf_endpg,
                    "chpstpage": None,
                    "chpendpage": None
                })
                continue

            if pdf_startpg < pdf_offset:
                # Convert chapter pages to PDF pages
                pdf_startpg = pdf_startpg + pdf_offset
                pdf_endpg = pdf_endpg + pdf_offset

            chpstpage = pdf_startpg - pdf_offset
            chpendpage = pdf_endpg - pdf_offset
            
            if chpstpage < 1:
                error_msg = f"Invalid calculation: chpstpage={chpstpage} (pdf_startpg={pdf_startpg} - pdf_offset={pdf_offset})"
                raise HTTPException(status_code=500, detail=error_msg)
            
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


@router.post("/extract-chapters")
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
        cache_key = f"{class_name}_{subject.lower()}"

        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            print(f"[CACHE] Loaded existing cache with {len(cache)} books")
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}
            print(f"[CACHE] No existing cache found, starting fresh")
        
        if cache_key in cache:
            print(f"[CACHE] Found cached data for {cache_key}")
            return JSONResponse(content=cache[cache_key])

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {safe_filename}")

        print(f"[EXTRACT] Processing PDF for {cache_key}...")
        
        extracted_data = extract_chapters_from_pdf(pdf_path)
        
        # Compute book_uuid based on MD5 checksum
        import hashlib
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as afile:
            buf = afile.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(65536)
        book_uuid = hasher.hexdigest()
        
        extracted_data['book_uuid'] = book_uuid
        extracted_data['filename'] = safe_filename
        extracted_data['class_name'] = class_name
        extracted_data['subject'] = subject
        
        cache[cache_key] = extracted_data
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        
        print(f"[CACHE] Saved {cache_key} to cache (now {len(cache)} books total)")
            
        return JSONResponse(content=extracted_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to extract chapters: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract chapters: {e}")


@router.get("/api/clear-qdrant")
async def clear_qdrant_data():
    """
    Clears all data from the Qdrant collection.
    """
    try:
        qdrant.clear_qdrant_collection()
        return {"message": "Qdrant collection cleared and re-initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear Qdrant collection: {e}")

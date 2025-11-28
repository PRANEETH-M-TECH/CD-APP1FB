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
from .session_service import session_manager
from .intent_classifier import determine_next_action


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
        # Frontend now sends chpstpage/chpendpage, we need to calculate pdf_startpg/pdf_endpg
        for chapter in chapters:
            if 'chpstpage' in chapter and 'chpendpage' in chapter:
                chapter['pdf_startpg'] = chapter['chpstpage'] + pdf_offset
                chapter['pdf_endpg'] = chapter['chpendpage'] + pdf_offset
                logger.info(f"Calculated PDF pages for {chapter.get('chapter_name')}: "
                           f"chp {chapter['chpstpage']}-{chapter['chpendpage']} → "
                           f"pdf {chapter['pdf_startpg']}-{chapter['pdf_endpg']}")
        
        # NOTE: Do NOT save to cache here! The cache was already saved during chapter extraction.
        # The frontend sends incomplete chapter data (missing chpstpage/chpendpage in some cases),
        # and saving it here would corrupt the cache. The cache is the source of truth.
        # REMOVED: local_chap_service.save_book_details(class_name, subject, book_uuid, filename, chapters)

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
            print(f"[PROCESS] │  Extracting text from PDF...")

            # Extract text
            chapter_text = ""
            for page_num in range(start_page - 1, end_page):
                if 0 <= page_num < len(reader.pages):
                    chapter_text += reader.pages[page_num].extract_text() or ""

            # Create and upload chunks to Qdrant
            text_chunks = text_splitter.split_text(chapter_text)
            print(f"[PROCESS] │  ✓ Extracted and split into {len(text_chunks)} chunks")
            print(f"[PROCESS] │  Uploading to Qdrant...")
            
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
                            "pdf_startpg": chapter_data.get("pdf_startpg"),
                            "pdf_endpg": chapter_data.get("pdf_endpg"),
                            "chpstpage": chapter_data.get("chpstpage"),
                            "chpendpage": chapter_data.get("chpendpage"),
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
                                collection_name="data",
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
            summary_text = qdrant.generate_chapter_summary(class_name, subject, chapter_name, text_chunks)
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

    results = qdrant.client.search(
        collection_name="data",
        query_vector=embedding,
        limit=5,
        query_filter=q_filter
    )

    return results


# === SMART CONVERSATIONAL CONTEXT HELPERS ===

def context_aware_reformulate(query: str, conversation_window: List[dict]) -> dict:
    """
    Reformulate query using previous conversation context.
    Expands vague references like "that", "it", "more" using previous Q&A.
    
    Args:
        query: Current user query (may be vague)
        conversation_window: List of previous turns
    
    Returns:
        {
            "reformulated_query": str,
            "keywords": List[str]
        }
    """
    if not conversation_window:
        # No context available, return as-is
        return {
            "reformulated_query": query,
            "keywords": []
        }
    
    # Get last 2 turns for context
    recent_turns = conversation_window[-2:] if len(conversation_window) >= 2 else conversation_window
    context_summary = ""
    
    for turn in recent_turns:
        # Truncate long answers
        answer_preview = turn.get('answer', '')[:200]
        if len(turn.get('answer', '')) > 200:
            answer_preview += "..."
        context_summary += f"Q: {turn['query']}\nA: {answer_preview}\n\n"
    
    prompt = f"""You are reformulating a follow-up query that references previous conversation.

PREVIOUS CONVERSATION:
{context_summary}

CURRENT USER QUERY (may be vague): "{query}"

Your tasks:
1. Expand vague references ("that", "it", "more", "this") using previous context
2. Make the query self-contained and specific
3. Extract keywords relevant to the EXPANDED query
4. Keep the query focused on the user's intent

Return ONLY JSON (no markdown, no code blocks):
{{
  "reformulated_query": "expanded, self-contained query",
  "keywords": ["keyword1", "keyword2", ...]
}}

Example:
Previous: Q: "What is photosynthesis?" A: "Photosynthesis is a process..."
Current: "explain more about that"
Result: {{"reformulated_query": "Provide more detailed explanation of the photosynthesis process, including light-dependent and light-independent reactions", "keywords": ["photosynthesis", "light reactions", "calvin cycle"]}}

Return only the JSON object:
"""
    
    try:
        response = qdrant.generation_model.generate_content(prompt)
        raw = response.text.strip()
        
        # Extract JSON from response
        json_text = extract_json_block(raw)
        if not json_text:
            # If extraction fails, try to parse directly
            json_text = raw
        
        result = json.loads(json_text)
        
        # Validate structure
        if "reformulated_query" not in result:
            raise ValueError("Missing reformulated_query in response")
        
        print(f"[REFORM] Context-aware reformulation successful")
        print(f"[REFORM] Original: {query}")
        print(f"[REFORM] Reformulated: {result['reformulated_query']}")
        
        return result
    
    except Exception as e:
        print(f"[REFORM] ⚠️ Context-aware reformulation failed: {e}")
        # Fallback: return original query
        return {
            "reformulated_query": query,
            "keywords": []
        }


def generate_smart_followups(query: str, answer: str, top_chunks: List) -> List[str]:
    """
    Generate answer-specific follow-up questions tailored for Indian students.
    Questions are age-appropriate, in simple English, and contextually relevant.
    
    Args:
        query: Original query
        answer: Generated answer
        top_chunks: Top retrieved chunks (for chapter/class/subject context)
    
    Returns:
        List of 3 follow-up question strings
    """
    try:
        # Extract metadata from top chunks
        chapter_names = []
        class_level = None
        subject = None
        
        for item in top_chunks[:3]:
            if isinstance(item, tuple) and len(item) >= 2:
                # Format: (score, payload)
                payload = item[1]
                chapter_name = payload.get("chapter_name", "Unknown")
                if chapter_name not in chapter_names and chapter_name != "Unknown":
                    chapter_names.append(chapter_name)
                
                # Extract class and subject
                if not class_level:
                    class_level = payload.get("class_name", None)
                if not subject:
                    subject = payload.get("subject", None)
        
        # Determine age-appropriate language level
        if class_level:
            try:
                class_num = int(str(class_level).replace("class", "").replace("Class", "").strip())
            except:
                class_num = 8  # Default to middle school
        else:
            class_num = 8

        # Define language complexity based on class
        if class_num <= 5:
            language_level = "very simple words, short sentences (like talking to a 10-year-old)"
            complexity = "basic concepts only, use everyday examples"
        elif class_num <= 8:
            language_level = "simple, clear English that a 13-year-old understands easily"
            complexity = "moderate depth, relatable examples from daily life"
        else:
            language_level = "clear, straightforward English (not complicated academic words)"
            complexity = "detailed but still clear, real-world applications"
        
        # Truncate long answers
        answer_preview = answer[:500]
        if len(answer) > 500:
            answer_preview += "..."
        
        prompt = f"""You are generating follow-up questions for an Indian student in Class {class_level or 'middle school'} studying {subject or 'the topic'}.

ORIGINAL QUESTION: {query}

ANSWER GIVEN:
{answer_preview}

RELEVANT CHAPTERS: {chapter_names if chapter_names else ['General']}

CRITICAL REQUIREMENTS:
1. **Student Age**: Class {class_level or '8'} Indian student ({class_num}-{class_num+2} years old)
2. **Language Level**: Use {language_level}
3. **English Style**: 
   - How Indian students actually speak/write English
   - Simple, clear words (avoid: "elaborate", "elucidate", "comprehend", "utilize")
   - Use common words (like: "explain more", "understand", "use", "what about")
4. **Question Style**: How an Indian kid would naturally ask
   - NOT: "Could you elaborate on the mechanism of..."
   - YES: "How does this work?" or "What happens when..."
5. **Context Boundary**: Questions MUST be:
   - About topics in the answer or mentioned chapters
   - {complexity}
   - Never introduce completely new advanced topics
6. **Variety**: Mix of question types:
   - "What happens if..." (consequence)
   - "How is X different from Y?" (comparison)
   - "Can you give an example of..." (application)
   - "Why does..." (reason)

BAD Examples (TOO COMPLEX for Class {class_level}):
- "Could you elaborate on the intricacies of the biochemical pathway?"
- "What are the ramifications of this phenomenon?"
- "How does this mechanism correlate with contemporary scenarios?"

GOOD Examples (RIGHT for Class {class_level}):
- "What happens inside a plant when it makes food?"
- "Why do plants need sunlight to grow?"
- "How is this different from what animals do?"

Return ONLY JSON (no markdown, no code blocks):
{{
  "followups": [
    "question 1 in simple Indian student English",
    "question 2 in simple Indian student English", 
    "question 3 in simple Indian student English"
  ]
}}

Generate 3 follow-up questions NOW:
"""
        
        response = qdrant.generation_model.generate_content(prompt)
        
        if not response.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            print(f"[FOLLOWUPS] ⚠️ LLM returned an empty response. Finish Reason: {finish_reason}.")
            raise ValueError(f"Empty response from LLM (finish reason: {finish_reason})")

        raw = response.text.strip()
        
        # Extract JSON
        json_text = extract_json_block(raw)
        if not json_text:
            json_text = raw
        
        result = json.loads(json_text)
        
        followups = result.get("followups", [])
        
        # Validate we have 3 followups
        if not followups or not isinstance(followups, list):
            raise ValueError("Invalid followups format")
        
        # Ensure we have exactly 3 (or at least some)
        followups = followups[:3]  # Take first 3
        
        print(f"[FOLLOWUPS] ✅ Generated {len(followups)} age-appropriate follow-ups for Class {class_level}")
        for i, f in enumerate(followups, 1):
            print(f"[FOLLOWUPS]   {i}. {f}")
        
        return followups
    
    except Exception as e:
        print(f"[FOLLOWUPS] ⚠️ Generation failed: {e}")
        # Fallback: return simple, age-appropriate generic follow-ups
        return [
            f"Can you explain more about this?",
            f"What is an example of this?",
            f"Why is this important?"
        ]


# 5. MAIN ENDPOINT — SSE STREAMING FOR REAL-TIME DISPLAY
@app.get("/api/query", tags=["LLM"])
async def query_engine(
    book_uuid: str = Query(...),
    query: str = Query(...),
    class_name: str = Query(...),
    subject: str = Query(...)
):
    """
    Streams the answer in real-time using Server-Sent Events (SSE).
    Frontend uses EventSource to receive chunks incrementally.
    """
    
    async def event_generator():
        start = time.time()
        
        # ========== START QUERY LOGGING ==========
        print(f"\n{'='*80}")
        print(f"[QUERY] New query received at {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"[QUERY] User question: {query}")
        print(f"[QUERY] Book: Class {class_name} - {subject.capitalize()}")
        print(f"[QUERY] Book UUID: {book_uuid[:16]}...")
        print(f"{'='*80}\n")
        
        # Load cached summaries (no Firestore after first load)
        print(f"[FIRESTORE] Loading summaries from cache/Firestore...")
        summary_doc = load_summary_from_firestore(class_name, subject)
        chapters = summary_doc["chapters"]
        print(f"[FIRESTORE] ✓ Loaded {len(chapters)} chapters\n")
        
        # Reformulate query + chapter ranking
        print(f"[REFORMULATE] Processing query with LLM...")
        try:
            reform = reformulate_with_llm(
                raw_query=query,
                class_name=class_name,
                subject=subject,
                chapters=chapters
            )
            
            # Validate response structure
            if not isinstance(reform, dict):
                print(f"[REFORMULATE] ⚠️ Invalid response type: {type(reform)}")
                raise ValueError("LLM returned non-dict response")
            
            if "reformulated_query" not in reform:
                print(f"[REFORMULATE] ⚠️ Missing 'reformulated_query' in response")
                print(f"[REFORMULATE] Response keys: {list(reform.keys())}")
                raise ValueError("Missing reformulated_query in LLM response")
            
            reformulated_query = reform["reformulated_query"]
            classification = reform.get("classification", "general")
            chapter_ranking = reform.get("chapter_ranking", [])
            
            print(f"[REFORMULATE] ✓ Original query: {query}")
            print(f"[REFORMULATE] ✓ Reformulated: {reformulated_query}")
            print(f"[REFORMULATE] ✓ Classification: {classification}")
            print(f"[REFORMULATE] ✓ Top chapters identified: {len(chapter_ranking)}\n")
            
        except Exception as e:
            print(f"[REFORMULATE] ✗ Error: {e}")
            print(f"[REFORMULATE] Using fallback: original query without reformulation\n")
            # Fallback to using original query
            reformulated_query = query
            classification = "general"
            chapter_ranking = chapters[:5]  # Use first 5 chapters as fallback
        
        # ========== CALCULATE SEMANTIC SIMILARITY SCORES ==========
        # This adds actual relevance scores instead of relying on LLM
        print(f"[SIMILARITY] Calculating semantic similarity scores for chapters...")
        try:
            from sentence_transformers import util
            
            # Embed the reformulated query
            query_embedding = qdrant.local_embedder.encode(reformulated_query, convert_to_tensor=True)
            
            # Calculate similarity for each chapter
            scored_chapters = []
            for chapter in chapters:
                summary = chapter.get("summary", "")
                if summary:
                    summary_embedding = qdrant.local_embedder.encode(summary, convert_to_tensor=True)
                    similarity = util.cos_sim(query_embedding, summary_embedding)[0][0].item()
                    
                    chapter_with_score = chapter.copy()
                    chapter_with_score['relevance_score'] = round(similarity, 3)
                    scored_chapters.append(chapter_with_score)
                else:
                    chapter_copy = chapter.copy()
                    chapter_copy['relevance_score'] = 0.0
                    scored_chapters.append(chapter_copy)
            
            # Sort by relevance score
            scored_chapters.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Use top 5 most relevant chapters
            chapter_ranking = scored_chapters[:5]
            
            print(f"[SIMILARITY] ✓ Calculated similarity scores for {len(scored_chapters)} chapters")
            print(f"[SIMILARITY] Top 3 most relevant:")
            for idx, ch in enumerate(chapter_ranking[:3], 1):
                print(f"  {idx}. {ch.get('chapter_name', 'Unknown')} (score: {ch.get('relevance_score', 0):.3f})")
            print()
            
        except Exception as e:
            print(f"[SIMILARITY] ✗ Error calculating similarity: {e}")
            # Fallback: map LLM scores if available
            if chapter_ranking:
                for ch in chapter_ranking:
                    if 'score' in ch and 'relevance_score' not in ch:
                        ch['relevance_score'] = ch['score']
                    elif 'relevance_score' not in ch:
                        ch['relevance_score'] = 0.0
            print()
        
        print(f"[RANKING] Chapter ranking (top 5):")
        for idx, ch in enumerate(chapter_ranking[:5], 1):
            print(f"  {idx}. {ch.get('chapter_name', 'Unknown')} (relevance: {ch.get('relevance_score', 'N/A')})")
        print()
        
        
        # Retrieve context from Qdrant with hybrid search
        print(f"[RETRIEVAL] Performing hybrid search...")
        
        # Get metadata from qdrant for hybrid search
        metadata = qdrant.get_book_metadata(book_uuid)
        
        # Perform reformulation with keywords
        processed_data = qdrant.reformulate_and_classify_query(
            query=reformulated_query,
            class_name=metadata.get("class_name"),
            subject=metadata.get("subject"),
            chapter_list=[ch["chapter_name"] for ch in chapter_ranking]
        )
        
        keywords = processed_data.get("keywords", [])
        conceptual_score = processed_data.get("conceptual_score", 0.5)
        
        
        # Perform hybrid search with chapter filtering
        # Only search in top 5 most relevant chapters
        top_chapter_names = [ch["chapter_name"] for ch in chapter_ranking[:5]]
        
        print(f"[RETRIEVAL] 🎯 Restricting search to top {len(top_chapter_names)} chapters")
        
        hybrid_results, semantic_results, bm25_results = qdrant.hybrid_search(
            book_uuid=book_uuid,
            query=reformulated_query,
            keywords=keywords,
            conceptual_score=conceptual_score,
            metadata_filters={"chapter_names": top_chapter_names}
        )
        
        print(f"[RETRIEVAL] ✓ Semantic search returned {len(semantic_results)} results")
        print(f"[RETRIEVAL] ✓ BM25 keyword search returned {len(bm25_results)} results")
        print(f"[RETRIEVAL] ✓ Hybrid ranking produced {len(hybrid_results)} final chunks\n")
        
        # Build context for LLM
        context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
        
        # ========== WRITE TO ans.txt (in background) ==========
        # We'll write after streaming completes
        ans_txt_data = {
            "query": query,
            "reformulated_query": reformulated_query,
            "classification": classification,
            "conceptual_score": conceptual_score,
            "chapter_ranking": chapter_ranking,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "hybrid_results": hybrid_results,
            "start_time": start
        }
        
        # ========== STREAM ANSWER ==========
        print(f"[LLM] Streaming answer with top {min(10, len(hybrid_results))} chunks as context...")
        
        final_prompt = f"""
You are a helpful teacher. Use the context to answer the question clearly:

QUESTION:
{reformulated_query}

CONTEXT:
{context}

Return only the answer.
"""
        
        full_answer = ""
        try:
            response = qdrant.generation_model.generate_content(final_prompt, stream=True)
            
            for chunk in response:
                try:
                    if chunk.text:
                        full_answer += chunk.text
                        # Send SSE event with both display and read text
                        event_data = json.dumps({
                            "display_text": chunk.text,
                            "read_text": chunk.text 
                        })
                        yield f"data: {event_data}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for smooth streaming
                except ValueError:
                    # This can happen if the chunk has no 'parts' but a finish_reason.
                    # We can safely ignore it and continue to the next chunk.
                    pass
            
            print(f"[LLM] ✓ Answer streamed ({len(full_answer)} characters)\n")
            
        except Exception as e:
            print(f"[LLM] ✗ Error generating answer: {e}\n")
            error_msg = "Sorry, I couldn't generate the answer."
            full_answer = error_msg
            event_data = json.dumps({"display_text": error_msg, "read_text": error_msg})
            yield f"data: {event_data}\n\n"
        
        # Send completion signal
        yield "data: [DONE]\n\n"
        
        # Write to ans.txt after streaming
        print(f"[LOG] Writing detailed log to ans.txt...")
        try:
            with open("ans.txt", "w", encoding="utf-8") as f:
                f.write(f"{'='*80}\n")
                f.write(f"QUERY LOG - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"1. ORIGINAL QUERY:\n")
                f.write(f"   {ans_txt_data['query']}\n\n")
                
                f.write(f"2. REFORMULATED QUERY:\n")
                f.write(f"   {ans_txt_data['reformulated_query']}\n")
                f.write(f"   Classification: {ans_txt_data['classification']}\n")
                f.write(f"   Conceptual Score: {ans_txt_data['conceptual_score']:.2f}\n\n")
                
                f.write(f"3. CHAPTER RANKING ({len(ans_txt_data['chapter_ranking'])} chapters):\n")
                for idx, ch in enumerate(ans_txt_data['chapter_ranking'], 1):
                    relevance = ch.get('relevance_score', 'N/A')
                    f.write(f"   {idx}. {ch['chapter_name']} (relevance: {relevance})\n")
                f.write(f"\n")
                
                f.write(f"4. TOP 10 SEMANTIC SEARCH RESULTS:\n")
                for idx, result in enumerate(ans_txt_data['semantic_results'][:10], 1):
                    chapter = result.payload.get("chapter_name", "N/A")
                    score = result.score
                    text = result.payload.get("text", "")
                    f.write(f"\n   --- Semantic Result #{idx} (score: {score:.4f}) ---\n")
                    f.write(f"   Chapter: {chapter}\n")
                    f.write(f"   Text: {text[:300]}...\n")
                f.write(f"\n")
                
                f.write(f"5. TOP 10 BM25 KEYWORD SEARCH RESULTS:\n")
                for idx, (score, doc) in enumerate(ans_txt_data['bm25_results'][:10], 1):
                    chapter = doc.get("chapter_name", "N/A")
                    text = doc.get("text", "")
                    f.write(f"\n   --- BM25 Result #{idx} (score: {score:.4f}) ---\n")
                    f.write(f"   Chapter: {chapter}\n")
                    f.write(f"   Text: {text[:300]}...\n")
                f.write(f"\n")
                
                f.write(f"6. FINAL HYBRID CHUNKS (Context sent to LLM):\n")
                for idx, (score, doc) in enumerate(ans_txt_data['hybrid_results'][:10], 1):
                    chapter = doc.get("chapter_name", "N/A")
                    text = doc.get("text", "")
                    f.write(f"\n   --- Hybrid Chunk #{idx} (score: {score:.4f}) ---\n")
                    f.write(f"   Chapter: {chapter}\n")
                    f.write(f"   Text: {text}\n")
                f.write(f"\n")
                
                f.write(f"7. GENERATED ANSWER:\n")
                f.write(f"{full_answer}\n\n")
                
                f.write(f"{'='*80}\n")
                f.write(f"Query processed in {time.time() - ans_txt_data['start_time']:.2f} seconds\n")
                f.write(f"{'='*80}\n")
            
            print(f"[LOG] ✓ Detailed log written to ans.txt\n")
        except Exception as e:
            print(f"[LOG] ✗ Error writing to ans.txt: {e}\n")
        
        print(f"{'='*80}")
        print(f"[COMPLETE] Query processed in {time.time() - start:.2f} seconds")
        print(f"{'='*80}\n")
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# === SMART QUERY ENDPOINT WITH CONVERSATIONAL CONTEXT ===

@app.get("/api/smart_query", tags=["LLM"])
async def smart_query_engine(
    book_uuid: str = Query(...),
    query: str = Query(...),
    class_name: str = Query(...),
    subject: str = Query(...),
    session_id: str = Query(None),
    is_clicked_followup: bool = Query(False)
):
    """
    Smart query endpoint with conversational context, using an action-based routing model.
    """
    import time
    
    async def event_generator():
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[SMART QUERY] New query at {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"  Query: {query} | Book: {class_name} - {subject} | Session: {session_id}")
        print(f"{'='*80}\n")

        try:
            # 1. Get or create session
            session = session_manager.get_or_create_session(book_uuid, session_id)
            active_context_window = session["active_context_window"]
            
            # Extract last action for context awareness (NEW)
            last_action = None
            if active_context_window:
                last_action = active_context_window[-1].get("intent_type")
            
            print(f"[CONTEXT] Last action: {last_action}")
            print(f"[CONTEXT] Is clicked follow-up: {is_clicked_followup}\n")
            
            # 2. Determine the next action using 5-tier routing (UPDATED)
            action_details = determine_next_action(
                current_query=query,
                conversation_window=active_context_window,
                generation_model=qdrant.generation_model,
                embedder=qdrant.local_embedder,  # Pass embedder for similarity analysis
                is_clicked_followup=is_clicked_followup,  # NEW: Flag for clicked follow-ups
                last_action=last_action  # NEW: Previous action for context
            )
            action = action_details.get("action")
            reason = action_details.get("reason", "No reason provided.")
            similarity_score = action_details.get("similarity_score", 0.0)
            tier = action_details.get("tier", "UNKNOWN")
            
            print(f"[ACTION] Determined Action: {action}")
            print(f"[ACTION] Tier: {tier}")
            print(f"[ACTION] Reason: {reason}")
            print(f"[ACTION] Similarity Score: {similarity_score:.3f}\n")
            
            # Send action info to the frontend for debugging/display
            yield f"data: {json.dumps({'type': 'intent', 'intent': action})}\n\n"

            # Initialize variables
            context = ""
            hybrid_results = []
            reformulated_query = query
            keywords = []
            full_answer = ""

            # 3. Execute the determined action
            if action == "RETRIEVE_NEW_CONTEXT":
                print("[PATH] 🔍 New topic detected. Starting full retrieval pipeline...\n")
                # If it's a new topic, start a new topic in the session manager
                new_topic_name = action_details.get("new_topic_name", "New Topic")
                session_manager.start_new_topic(session['session_id'], new_topic_name)

                # Load summaries for chapter ranking
                summary_doc = load_summary_from_firestore(class_name, subject)
                chapters = summary_doc.get("chapters", [])
                
                # Reformulate query for retrieval
                reform = reformulate_with_llm(query, class_name, subject, chapters)
                reformulated_query = reform.get("reformulated_query", query)
                keywords = reform.get("keywords", [])
                chapter_ranking = reform.get("chapter_ranking", [])
                conceptual_score = reform.get("conceptual_score", 0.5) # Extract conceptual_score
                print(f"[REFORM] Reformulated for retrieval: {reformulated_query}\n")

                # Perform hybrid search
                top_chapter_names = [ch["chapter_name"] for ch in chapter_ranking[:5]] if chapter_ranking else []
                hybrid_results, _, _ = qdrant.hybrid_search(
                    book_uuid=book_uuid,
                    query=reformulated_query,
                    keywords=[{"keyword": kw, "importance": 0.5} for kw in keywords],
                    conceptual_score=conceptual_score, # Pass conceptual_score
                    metadata_filters={"chapter_names": top_chapter_names} if top_chapter_names else {}
                )
                print(f"[RETRIEVAL] Retrieved {len(hybrid_results)} new chunks for topic '{new_topic_name}'.\n")

                # Build context and cache it
                context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
                session_manager.update_topic_chunks(session['session_id'], hybrid_results)

            elif action == "USE_CACHED_CONTEXT":
                print("[PATH] ⚡ Follow-up detected. Using cached context...\n")
                cached_chunks = session_manager.get_current_topic_chunks(session['session_id'])
                if cached_chunks:
                    hybrid_results = cached_chunks
                    context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
                    print(f"[REUSE] Successfully reused {len(hybrid_results)} cached chunks.\n")
                else:
                    # Fallback if cache is somehow empty
                    print("[WARN] 'USE_CACHED_CONTEXT' chosen, but cache was empty. Falling back to full retrieval.\n")
                    action = "RETRIEVE_NEW_CONTEXT" # Force retrieval
                    # This will re-run the logic in the next step, might need a refactor later
                    # For now, we will just re-do the retrieval logic here.
                    summary_doc = load_summary_from_firestore(class_name, subject)
                    chapters = summary_doc.get("chapters", [])
                    reform = reformulate_with_llm(query, class_name, subject, chapters)
                    reformulated_query = reform.get("reformulated_query", query)
                    keywords = reform.get("keywords", [])
                    chapter_ranking = reform.get("chapter_ranking", [])
                    top_chapter_names = [ch["chapter_name"] for ch in chapter_ranking[:5]] if chapter_ranking else []
                    hybrid_results, _, _ = qdrant.hybrid_search(
                        book_uuid=book_uuid, query=reformulated_query, keywords=[{"keyword": kw, "importance": 0.5} for kw in keywords],
                        metadata_filters={"chapter_names": top_chapter_names} if top_chapter_names else {}
                    )
                    context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
                    session_manager.update_topic_chunks(session['session_id'], hybrid_results)

                # For follow-ups, we still want to reformulate the query for clarity in the prompt
                reform = context_aware_reformulate(query, active_context_window)
                reformulated_query = reform.get("reformulated_query", query)
                print(f"[REFORM] Context-aware reformulation for follow-up: {reformulated_query}\n")

            # 4. Generate Answer
            if action in ["RETRIEVE_NEW_CONTEXT", "USE_CACHED_CONTEXT"]:
                conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
                for turn in session.get("full_history", [])[-3:]:
                    conversation_context += f"Q: {turn['query']}\nA: {turn.get('answer', 'N/A')[:200]}...\n\n"

                final_prompt = f"""You are a helpful AI tutor. Maintain conversational continuity.

{conversation_context}
CURRENT QUESTION: {reformulated_query}

Use the following retrieved information to answer the current question:
RETRIEVED INFORMATION:
{context}

Answer the current question clearly and educationally.
"""
            else: # ANSWER_FROM_HISTORY
                print("[PATH] 🗣️ Answering directly from history...\n")
                context_summary = ""
                for turn in session.get("full_history", []):
                    answer_preview = turn.get('answer', '')[:200]
                    if len(turn.get('answer', '')) > 200:
                        answer_preview += "..."
                    context_summary += f"Q: {turn['query']}\nA: {answer_preview}\n\n"
                
                final_prompt = f"""You are an AI assistant. Answer the following question based ONLY on the provided conversation history.

CONVERSATION HISTORY:
{context_summary}

QUESTION:
"{query}"

Answer the question based only on the history.
"""

            print(f"[LLM] Streaming answer for action: {action}...\n")
            response_stream = qdrant.generation_model.generate_content(final_prompt, stream=True)
            for chunk in response_stream:
                try:
                    if chunk.text:
                        full_answer += chunk.text
                        yield f"data: {json.dumps({'display_text': chunk.text})}\n\n"
                        await asyncio.sleep(0.01)
                except ValueError:
                    # This can happen if the chunk has no 'parts' but a finish_reason.
                    # We can safely ignore it and continue to the next chunk.
                    pass
            print(f"[LLM] ✓ Answer generated ({len(full_answer)} chars)\n")

            # 5. Generate and send follow-ups (if not answering from history)
            follow_ups = []
            if action != "ANSWER_FROM_HISTORY":
                print("[FOLLOWUPS] Generating answer-specific follow-ups...\n")
                follow_ups = generate_smart_followups(reformulated_query, full_answer, hybrid_results[:5])
            
            yield f"data: {json.dumps({'type': 'followups', 'followups': follow_ups})}\n\n"

            # 6. Save the complete turn to the session
            turn_data = {
                "query": query, "reformulated": reformulated_query, "answer": full_answer,
                "intent_type": action, "is_clicked_followup": is_clicked_followup,
                "tier": tier,  # NEW: Track which tier determined the action
                "similarity_score": similarity_score,  # NEW: Track similarity for analytics
                "follow_ups": follow_ups, "timestamp": datetime.datetime.now().isoformat()
            }
            if action == "RETRIEVE_NEW_CONTEXT":
                 turn_data["context_cache"] = {"retrieved_chunks": hybrid_results, "context": context, "keywords": keywords}

            session_manager.add_turn(session["session_id"], turn_data)

            # 7. Send final metadata with cache info and chunks
            updated_history = session_manager.get_full_history(session["session_id"])
            current_turn_number = len(updated_history)
            
            # Prepare retrieved chunks summary for user visibility
            chunks_summary = []
            if hybrid_results:
                for score, doc in hybrid_results[:5]:  # Top 5 chunks
                    chunks_summary.append({
                        "chapter_name": doc.get("chapter_name", "Unknown"),
                        "relevance_score": round(score, 3),
                        "text_preview": doc.get("text", "")[:150] + "..." if len(doc.get("text", "")) > 150 else doc.get("text", ""),
                        "pdf_pages": f"{doc.get('pdf_startpg', '?')}-{doc.get('pdf_endpg', '?')}",
                        "chapter_pages": f"{doc.get('chpstpage', '?')}-{doc.get('chpendpage', '?')}"
                    })
            
            # Calculate cache performance metrics
            cache_hit = (action == "USE_CACHED_CONTEXT")
            retrieval_time_saved = 0
            if cache_hit:
                retrieval_time_saved = 1500  # Approximate ms saved by not doing retrieval
            
            metadata = {
                "type": "metadata",
                "turn": current_turn_number,
                "session_id": session["session_id"],
                "intent_type": action,
                "topic_change": action == "RETRIEVE_NEW_CONTEXT",
                "cache_info": {
                    "cache_hit": cache_hit,
                    "similarity_score": round(similarity_score, 3),
                    "chunks_reused": len(hybrid_results) if cache_hit else 0,
                    "retrieval_time_saved_ms": retrieval_time_saved
                },
                "retrieved_chunks": chunks_summary
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            elapsed = time.time() - start_time
            print(f"{'='*80}")
            print(f"[COMPLETE] Smart query processed in {elapsed:.2f}s")
            print(f"[COMPLETE] Action: {action} | Turn: {current_turn_number}")
            print(f"{'='*80}\n")
            
            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"[ERROR] Smart query failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# === SESSION VISIBILITY ENDPOINTS ===

@app.get("/api/session/history", tags=["Session"])
async def get_session_history(session_id: str = Query(...)):
    """
    Returns complete chat history with metadata for a given session.
    Provides full transparency into conversation state and topics.
    
    Args:
        session_id: The session ID to retrieve history for
    
    Returns:
        Complete session data including topics, history, and cache status
    """
    from .redis_service import redis_service
    
    session = redis_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    # Calculate statistics
    total_turns = len(session.get("full_history", []))
    cache_hits = sum(1 for turn in session.get("full_history", []) 
                     if turn.get("intent_type") == "USE_CACHED_CONTEXT")
    cache_hit_rate = (cache_hits / total_turns * 100) if total_turns > 0 else 0
    
    return {
        "session_id": session_id,
        "book_uuid": session.get("book_uuid"),
        "created_at": session.get("created_at"),
        "last_updated": session.get("last_updated"),
        "statistics": {
            "total_turns": total_turns,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 1),
            "total_topics": len(session.get("topics", []))
        },
        "topics": session.get("topics", []),
        "current_topic_id": session.get("current_topic_id"),
        "full_history": session.get("full_history", []),
        "active_context_window": session.get("active_context_window", [])
    }


@app.get("/api/session/chunks", tags=["Session"])
async def get_current_chunks(session_id: str = Query(...)):
    """
    Returns currently cached chunks for the active topic.
    Shows what context is being reused for follow-up queries.
    
    Args:
        session_id: The session ID to retrieve chunks for
    
    Returns:
        List of cached chunks with relevance scores and metadata
    """
    chunks = session_manager.get_current_topic_chunks(session_id)
    
    if not chunks:
        return {
            "chunks": [],
            "total_count": 0,
            "message": "No cached chunks for current topic"
        }
    
    formatted_chunks = []
    for score, doc in chunks:
        formatted_chunks.append({
            "relevance_score": round(score, 4),
            "chapter_name": doc.get("chapter_name", "Unknown"),
            "text": doc.get("text", ""),
            "text_length": len(doc.get("text", "")),
            "pdf_pages": f"{doc.get('pdf_startpg', '?')}-{doc.get('pdf_endpg', '?')}",
            "chapter_pages": f"{doc.get('chpstpage', '?')}-{doc.get('chpendpage', '?')}"
        })
    
    return {
        "chunks": formatted_chunks,
        "total_count": len(formatted_chunks),
        "message": f"Currently using {len(formatted_chunks)} cached chunks"
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

    # Load summaries from Firestore (or cache)
    summary_doc = firestore_service.load_summary_from_firestore(class_name, subject)
    
    chapter_summary = None
    if summary_doc and "chapters" in summary_doc:
        for chap in summary_doc["chapters"]:
            if chap.get("chapter_name") == chapter_name:
                chapter_summary = chap.get("summary")
                break
    
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
        for chapter in llm_chapters_list:
            pdf_startpg = chapter.get("pdf_startpg")
            pdf_endpg = chapter.get("pdf_endpg")

            # Fallback to alternative field names
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

            # ========== CRITICAL: Detect if LLM returned chapter pages instead of PDF pages ==========
            # If pdf_startpg < pdf_offset, LLM probably gave us chapter page numbers
            if pdf_startpg < pdf_offset:
                print(f"\n{'='*70}")
                print(f"[AUTO-FIX] LLM returned chapter page instead of PDF page!")
                print(f"[AUTO-FIX] Chapter: {chapter.get('chapter_name')}")
                print(f"[AUTO-FIX] LLM gave: pdf_startpg={pdf_startpg}, pdf_endpg={pdf_endpg}")
                print(f"[AUTO-FIX] pdf_offset={pdf_offset}")
                print(f"[AUTO-FIX] Correcting: Adding offset to convert to PDF pages")
                
                # Convert chapter pages to PDF pages
                original_start = pdf_startpg
                original_end = pdf_endpg
                pdf_startpg = pdf_startpg + pdf_offset
                pdf_endpg = pdf_endpg + pdf_offset
                
                print(f"[AUTO-FIX] Corrected: pdf_startpg={pdf_startpg}, pdf_endpg={pdf_endpg}")
                print(f"{'='*70}\n")

            # Calculate chapter pages
            chpstpage = pdf_startpg - pdf_offset
            chpendpage = pdf_endpg - pdf_offset
            
            # Validate the calculation
            if chpstpage < 1:
                error_msg = f"Invalid calculation: chpstpage={chpstpage} (pdf_startpg={pdf_startpg} - pdf_offset={pdf_offset})"
                print(f"\n[ERROR] {error_msg}")
                print(f"[ERROR] This indicates the LLM data is incorrect!")
                print(f"[ERROR] Chapter: {chapter.get('chapter_name')}\n")
                raise HTTPException(status_code=500, detail=error_msg)
            
            processed_chapters.append({
                "chapter_name": chapter.get("chapter_name"),
                "pdf_startpg": pdf_startpg,
                "pdf_endpg": pdf_endpg,
                "chpstpage": chpstpage,
                "chpendpage": chpendpage
            })
            
            # Log successful processing
            print(f"[CHAPTER] {chapter.get('chapter_name')}: PDF pages {pdf_startpg}-{pdf_endpg}, Chapter pages {chpstpage}-{chpendpage}")
        
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
        cache_key = f"{class_name}_{subject.lower()}"

        # 1. Load existing cache - PRESERVE all books
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            print(f"[CACHE] Loaded existing cache with {len(cache)} books")
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}
            print(f"[CACHE] No existing cache found, starting fresh")
        
        # 2. Check if this book is already cached
        if cache_key in cache:
            print(f"[CACHE] Found cached data for {cache_key}")
            return JSONResponse(content=cache[cache_key])

        # 3. If not in cache, process the PDF
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {safe_filename}")

        print(f"[EXTRACT] Processing PDF for {cache_key}...")
        
        # extract_chapters_from_pdf returns dict with pdf_offset and chapters
        extracted_data = extract_chapters_from_pdf(pdf_path)
        book_uuid = qdrant.get_book_uuid(pdf_path)
        
        # Add metadata
        extracted_data['book_uuid'] = book_uuid
        extracted_data['filename'] = safe_filename
        extracted_data['class_name'] = class_name
        extracted_data['subject'] = subject
        
        # 4. Update cache for THIS book only (preserves other books)
        cache[cache_key] = extracted_data
        
        # 5. Save back to file - KEEPS all other books intact
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        
        print(f"[CACHE] Saved {cache_key} to cache (now {len(cache)} books total)")
            
        return JSONResponse(content=extracted_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to extract chapters: {e}", exc_info=True)
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

@app.get("/admin-login.html")
async def admin_login():
    return FileResponse('public/admin-login.html')

@app.get("/mode-selection")
async def mode_selection():
    return FileResponse('public/mode-selection.html')

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
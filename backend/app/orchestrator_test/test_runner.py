import os
import sys
import os
import json
import time
import datetime
import re
from typing import Dict, Any, Optional

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

# Imports from backend app
from backend.app.services.llm.openai_client import OPENAI_MODEL, create_client
from backend.app.core.firebase.firebase_init import db
from backend.app.services.retrieval import qdrant_service

# Path to master prompt file
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "master_orchestrator_prompt.txt")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Initialize Qdrant and Gemini API Client
try:
    qdrant_service.initialize()
except Exception as _qdrant_init_err:
    print(f"[WARN] Qdrant initialization failed at startup (will retry on first request): {_qdrant_init_err}")
# openai_client will be dynamically fetched inside functions to avoid NoneType binding issue



def load_master_prompt_template() -> str:
    """Reads the locked Master System Prompt from file."""
    if os.path.exists(PROMPT_FILE_PATH):
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Master prompt file not found at: {PROMPT_FILE_PATH}")


from google.cloud.firestore_v1.base_query import FieldFilter

def authenticate_student_by_email(email: str) -> Dict[str, Any]:
    """
    Authenticates/fetches student profile from Firestore by email.
    Strips trailing spaces from Firestore string fields to ensure robust matching.
    """
    email_clean = email.strip().lower()
    firestore_db = db

    if firestore_db:
        try:
            users_ref = firestore_db.collection("users").get()
            for doc in users_ref:
                data = doc.to_dict()
                doc_email = str(data.get("email", "")).strip().lower()
                if doc_email == email_clean:
                    user_name = str(data.get("name", "Student")).strip()
                    user_class = int(data.get("class", 7))
                    user_board = str(data.get("board", "CBSE")).strip()
                    user_role = str(data.get("role", "student")).strip()
                    print(f"[AUTH SUCCESS] Logged in as: {user_name} (Class {user_class}, {user_board})")
                    return {
                        "uid": doc.id,
                        "email": email_clean,
                        "name": user_name,
                        "class": user_class,
                        "board": user_board,
                        "role": user_role
                    }
        except Exception as e:
            print(f"[AUTH WARN] Error querying Firestore users: {e}")

    # Default fallback profile for testing (e.g. Praneeth Class 7)
    print(f"[AUTH NOTICE] Email '{email_clean}' not found in Firestore. Using fallback profile.")
    return {
        "uid": "test_user_007",
        "email": email_clean,
        "name": "Praneeth",
        "class": 7,
        "board": "CBSE",
        "role": "student"
    }


def get_cached_curriculum_metadata(grade: int) -> str:
    """
    Fetches available subject & chapter summaries from Firestore or local service for the student's grade.
    """
    firestore_db = db
    chapter_summaries = []

    if firestore_db:
        try:
            chapters_ref = firestore_db.collection("chapters").where(filter=FieldFilter("class_level", "==", grade)).get()
            for doc in chapters_ref:
                data = doc.to_dict()
                subject = data.get("subject", "Science")
                title = data.get("title") or data.get("chapter_name", "Chapter")
                summary = data.get("summary") or data.get("topics", "")
                chapter_summaries.append(f"â€¢ {subject} | {title} -> Key topics: {summary}")
        except Exception as e:
            print(f"[CURRICULUM CACHE WARN] Firestore query exception: {e}")

    # Load local JSON chapter cache from chapterdata/chapters_cache.json
    cache_path = os.path.join(PROJECT_ROOT, "chapterdata", "chapters_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                local_cache = json.load(f)
            for key, val in local_cache.items():
                # Key format: '8_social', '6_science'
                parts = key.split("_")
                key_grade = int(parts[0]) if parts[0].isdigit() else None
                subj_name = parts[1].capitalize() if len(parts) > 1 else "Curriculum"
                
                # Match grade (or load for all available grades)
                if key_grade is None or key_grade == grade:
                    chaps = val.get("chapters", [])
                    for ch in chaps:
                        ch_name = ch.get("chapter_name", "")
                        chapter_summaries.append(f"â€¢ Class {key_grade or grade} {subj_name} | Chapter: {ch_name}")
        except Exception as e:
            print(f"[CURRICULUM CACHE WARN] chapters_cache.json load exception: {e}")

    if not chapter_summaries:
        # Fallback chapter list for Class 7 / Class 8 testing
        if grade == 7:
            chapter_summaries = [
                "â€¢ Science | Chapter 1: Nutrition in Plants -> Key topics: Photosynthesis, chlorophyll, stomata, autotrophs, solar energy.",
                "â€¢ Science | Chapter 2: Nutrition in Animals -> Key topics: Human digestive system, alimentary canal, stomach, small intestine, ruminants.",
                "â€¢ Science | Chapter 3: Heat -> Key topics: Temperature measurement, conduction, convection, radiation, insulators.",
                "â€¢ Mathematics | Chapter 1: Integers -> Key topics: Positive/negative numbers, addition/subtraction rules, number line.",
                "â€¢ Social Science | Chapter 1: Environment -> Key topics: Ecosystem, biotic/abiotic components, atmosphere, hydrosphere."
            ]
        else:
            chapter_summaries = [
                "â€¢ Social Studies | Chapter: The Kakatiyas - Emergence of a Regional Kingdom -> Key topics: Rani Rudramadevi, Prataparudra, Warangal Fort, Kakatiya administration.",
                "â€¢ Social Studies | Chapter: Making of Laws in the State Assembly -> Key topics: Legislative Assembly, MLA, Bill to Law, Governor approval.",
                "â€¢ Social Studies | Chapter: The Indian Constitution -> Key topics: Preamble, Fundamental Rights, Secularism, Democracy.",
                "â€¢ Science | Chapter: Crop Production -> Key topics: Agricultural practices, sowing, irrigation, harvesting."
            ]

    return "\n".join(chapter_summaries)


def get_valid_subjects_for_grade(grade: int) -> set:
    """
    Real subject names actually available for this grade, derived from the
    same sources as get_cached_curriculum_metadata(). Used to validate the
    orchestrator LLM's claimed matched_subject before trusting a CURRICULUM
    classification - the LLM can otherwise hallucinate a plausible-sounding
    subject (e.g. "Class 8 Science" for a student whose grade only actually
    has Social Studies loaded), silently mis-marking a general-knowledge
    question as curriculum and skipping real grounding.
    """
    subjects = set()
    firestore_db = db

    if firestore_db:
        try:
            chapters_ref = firestore_db.collection("chapters").where(filter=FieldFilter("class_level", "==", grade)).get()
            for doc in chapters_ref:
                data = doc.to_dict()
                subj = data.get("subject")
                if subj:
                    subjects.add(str(subj).strip().lower())
        except Exception as e:
            print(f"[CURRICULUM VALIDATION WARN] Firestore query exception: {e}")

    cache_path = os.path.join(PROJECT_ROOT, "chapterdata", "chapters_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                local_cache = json.load(f)
            for key in local_cache.keys():
                parts = key.split("_")
                if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == grade:
                    subjects.add(parts[1].strip().lower())
        except Exception as e:
            print(f"[CURRICULUM VALIDATION WARN] chapters_cache.json load exception: {e}")

    return subjects


def normalize_subject_name(raw: str) -> str:
    """Strips 'Class N'/'Grade N' framing the LLM sometimes adds, e.g.
    'Class 8 Science' -> 'science', so it can be compared against real subject names."""
    if not raw:
        return ""
    s = re.sub(r"\b(?:class|grade)\s*\d+\b", "", raw, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -_:")
    return s.lower()


def run_orchestrator_pipeline(raw_query: str, student_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the single-pass Orchestrator LLM, runs RAG search if CURRICULUM,
    and returns a complete execution report without Sarvam TTS audio or video rendering.
    """
    # Debug: log the incoming student_profile to verify propagation from request
    try:
        print(f"[ORCH DEBUG] run_orchestrator_pipeline received student_profile: {student_profile}")
    except Exception:
        print("[ORCH DEBUG] run_orchestrator_pipeline received student_profile: <unprintable>")

    openai_client = qdrant_service.openai_client
    if openai_client is None:
        try:
            qdrant_service.initialize()
        except Exception as e:
            print(f"[WARN] Failed to fully initialize qdrant_service dynamically (possibly due to Qdrant DB connection): {e}")
        # Always fetch it after trying, since Gemini client initialization happens first
        openai_client = qdrant_service.openai_client

    if openai_client is None:
        try:
            from backend.app.utils.llm_tracker import instrument_client
            openai_client = create_client()
            openai_client = instrument_client(openai_client)
            print("[ORCHESTRATOR] Standalone fallback OpenAI client initialized successfully.")
        except Exception as fallback_err:
            print(f"[ERROR] Standalone fallback OpenAI client initialization failed: {fallback_err}")

    start_time = time.time()
    system_prompt = load_master_prompt_template()

    # Step 1: Fetch Curriculum Chapter Cache
    grade = student_profile.get("class", 7)
    curriculum_cache_text = get_cached_curriculum_metadata(grade)

    # Step 2: Format System Prompt
    current_date_time = datetime.datetime.now().strftime("%A, %B %d, %Y (%H:%M:%S)")
    formatted_prompt = system_prompt.replace("{student_name}", student_profile.get("name", "Student"))
    formatted_prompt = formatted_prompt.replace("{student_grade}", str(grade))
    formatted_prompt = formatted_prompt.replace("{student_board}", student_profile.get("board", "CBSE"))
    formatted_prompt = formatted_prompt.replace("{current_date_time}", current_date_time)
    formatted_prompt = formatted_prompt.replace("{cached_subjects_and_chapter_summaries}", curriculum_cache_text)
    formatted_prompt = formatted_prompt.replace("{retrieved_top10_chunks}", "[RAG Chunks will be provided if CURRICULUM]")

    # Step 3: Run Orchestrator LLM (Single Pass)
    print(f"\n[ORCHESTRATOR LLM] Executing single-pass evaluation for Class {grade} query...")
    user_prompt = f"USER RAW QUERY: \"{raw_query}\""

    MODEL = os.environ.get("OPENAI_MODEL", OPENAI_MODEL)

    # Step 1/3 â€” Query classification (keyword-based, instant, free)
    # If query is GK/current events, we perform live Google Search grounding to answer.
    # Otherwise, for curriculum/school questions, we skip search to save 15-20 seconds.
    _GK_KEYWORDS = {
        "yesterday", "today", "latest", "recent", "breaking", "live", "ongoing",
        "won", "win", "lost", "score", "result", "match", "election", "elected",
        "protest", "strike", "rally", "arrested", "verdict", "announced", "launched",
        "world cup", "ipl", "fifa", "olympics", "championship",
        "party", "government", "minister", "president", "prime minister",
        "news", "happened", "incident", "2026", "2025",
    }
    query_lower = raw_query.lower()
    # Match keywords as complete words. A substring check would classify
    # "winter" as GK because the GK list contains the word "win".
    is_gk_query = any(
        re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", query_lower)
        for kw in _GK_KEYWORDS
    )
    query_type = "GK_KNOWLEDGE" if is_gk_query else "CURRICULUM"
    print(f"[ORCHESTRATOR] Step 1/3 â€” Query type: {query_type} (keyword match, 0ms)")

    # Enable Google Search grounding dynamically for GK/live queries
    config = {"temperature": 0.2}

    # Step 2/3 â€” Main Orchestrator LLM call â€” single model: gemini-2.5-flash
    search_note = "with Google Search Grounding (may take 15-25s)" if is_gk_query else "without Search Grounding (fast)"
    response = None
    last_error = None
    for attempt in range(1, 4):  # up to 3 retries on 503
        try:
            _llm_start = time.time()
            print(f"[ORCHESTRATOR] Step 2/3 â€” [{MODEL}] {search_note} (Attempt {attempt}/3)...")
            response = openai_client.models.generate_content(
                model=MODEL,
                contents=[formatted_prompt, user_prompt],
                config=config
            )
            _llm_dur = time.time() - _llm_start
            if response and response.text:
                print(f"[ORCHESTRATOR LLM SUCCESS] [{MODEL}] responded in {_llm_dur:.1f}s")
                break
        except Exception as err:
            last_error = err
            _err_str = str(err)
            _elapsed = time.time() - _llm_start
            if "503" in _err_str or "UNAVAILABLE" in _err_str:
                _backoff = 2.0 * attempt
                print(f"[WARN] [{MODEL}] 503 (attempt {attempt}/3, {_elapsed:.1f}s), retrying in {_backoff:.0f}s...")
                time.sleep(_backoff)
            else:
                print(f"[ERROR] [{MODEL}] failed after {_elapsed:.1f}s: {_err_str[:120]}")
                break

    if not response or not response.text:
        print(f"[ERROR] [{MODEL}] failed. Last error: {last_error}")
        return {
            "error": str(last_error),
            "raw_user_query": raw_query,
            "status": "FAILED"
        }





    raw_json_text = response.text.strip()
    
    # Robust JSON extractor
    def extract_and_parse_json(text: str) -> dict:
        text_clean = text.strip()
        first_brace = text_clean.find("{")
        last_brace = text_clean.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = text_clean[first_brace:last_brace + 1]
            return json.loads(json_candidate)
        return json.loads(text_clean)

    try:
        # Try cleaning markdown code fences first
        cleaned_text = raw_json_text
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        orchestrator_output = extract_and_parse_json(cleaned_text.strip())
    except Exception as parse_err:
        print(f"[WARN] Failed to parse LLM JSON: {parse_err}. Creating fallback query container.")
        # Create a clean fallback response using the raw model output directly as text_narration
        orchestrator_output = {
            "is_authorized": True,
            "refusal_reason": None,
            "classification": "GENERAL_KNOWLEDGE",
            "reformulated_query": raw_query,
            "matched_subject": "General Knowledge",
            "matched_chapter": None,
            "complexity_level": 1,
            "format_decision": "QUICK_ANSWER",
            "text_narration": raw_json_text,
            "video_storyboard": None
        }


    # Extract Orchestrator Decisions
    is_authorized = orchestrator_output.get("is_authorized", True)
    classification = orchestrator_output.get("classification", "CURRICULUM")
    reformulated_query = orchestrator_output.get("reformulated_query") or raw_query
    matched_subject = orchestrator_output.get("matched_subject")
    matched_chapter = orchestrator_output.get("matched_chapter")
    format_decision = orchestrator_output.get("format_decision", "QUICK_ANSWER")

    # Validate the LLM's claimed curriculum match against real subject data
    # before trusting it - see get_valid_subjects_for_grade() docstring.
    if is_authorized and classification == "CURRICULUM":
        valid_subjects = get_valid_subjects_for_grade(grade)
        normalized_match = normalize_subject_name(matched_subject or "")
        subject_is_real = bool(normalized_match) and any(
            normalized_match == v or normalized_match in v or v in normalized_match
            for v in valid_subjects
        )
        if not subject_is_real:
            print(
                f"[CURRICULUM VALIDATION] Rejecting hallucinated match '{matched_subject}' - "
                f"Class {grade} only actually has: {sorted(valid_subjects) or 'no cached subjects'}. "
                f"Downgrading classification to GENERAL_KNOWLEDGE."
            )
            classification = "GENERAL_KNOWLEDGE"
            matched_subject = None
            matched_chapter = None
            orchestrator_output["classification"] = classification
            orchestrator_output["matched_subject"] = None
            orchestrator_output["matched_chapter"] = None

    rag_chunks = []
    rag_executed = False
    resolved_book_uuid = ""

    # Step 4: Handle RAG Retrieval if Authorized + CURRICULUM
    if is_authorized and classification == "CURRICULUM":
        print(f"[ORCHESTRATOR] Step 3/3 â€” RAG vector search for: '{reformulated_query[:60]}...'")
        print(f"[RAG SEARCH] Running hybrid vector search for: '{reformulated_query}'...")
        rag_executed = True
        try:
            # Resolve book_uuid from Firestore summaries using matched_subject and grade
            if matched_subject:
                try:
                    from backend.app.core.firebase.firebase_init import db
                    # Gemini may return "Science" or "Class 6 Science".
                    # Summary documents are keyed as summaries/{subject}_{class}.
                    normalized_subject = re.sub(
                        r"\b(?:class|grade)\s*\d+\b", "", str(matched_subject), flags=re.IGNORECASE
                    )
                    normalized_subject = re.sub(r"\s+", " ", normalized_subject).strip(" -_:")
                    sub_key = normalized_subject.lower().strip()
                    summary_key = f"{sub_key}_{grade}"
                    # Summaries are stored under summaries/{subject}_{class}
                    sum_doc = db.collection("summaries").document(summary_key).get()
                    if sum_doc.exists:
                        resolved_book_uuid = sum_doc.to_dict().get("book_uuid", "")
                        print(f"[RAG SEARCH] Resolved summaries/{summary_key} book_uuid (Class {grade}): {resolved_book_uuid}")
                    else:
                        print(f"[RAG SEARCH WARNING] Summary document not found: summaries/{summary_key}")
                except Exception as e:
                    print(f"[RAG RESOLVE WARNING] Failed to resolve book_uuid from Firestore: {e}")

            if not resolved_book_uuid:
                raise RuntimeError(
                    f"Could not resolve a book UUID for subject '{matched_subject}' and Class {grade}."
                )

            # Query Qdrant vector database using qdrant_service.hybrid_search
            if hasattr(qdrant_service, 'hybrid_search'):
                filters = {}
                if matched_chapter:
                    filters["chapter_names"] = [matched_chapter]
                raw_chunks, _, _ = qdrant_service.hybrid_search(
                    book_uuid=resolved_book_uuid,
                    query=reformulated_query,
                    keywords=[],
                    conceptual_score=0.7,
                    metadata_filters=filters
                )
            elif hasattr(qdrant_service, 'search_books_hybrid'):
                raw_chunks = qdrant_service.search_books_hybrid(
                    query=reformulated_query,
                    limit=10,
                    subject=matched_subject,
                    class_level=grade
                )
            else:
                raw_chunks = []

            for idx, c in enumerate(raw_chunks[:10], start=1):
                # hybrid_search returns (hybrid_score, payload_dict), while
                # some fallback implementations return a result object/dict.
                # Normalize both shapes before writing the audit report.
                if isinstance(c, tuple) and len(c) >= 2:
                    result_score, payload = c[0], c[1]
                else:
                    result_score, payload = getattr(c, "score", 0.0), c

                if hasattr(payload, "payload"):
                    payload = payload.payload
                if not isinstance(payload, dict):
                    payload = {}

                rag_chunks.append({
                    "chunk_index": idx,
                    "score": round(float(result_score or 0.0), 4),
                    "book_name": payload.get("book_name", payload.get("book", "")),
                    "chapter_name": payload.get("chapter_name", payload.get("chapter", "")),
                    "content_snippet": (payload.get("text", payload.get("content", "")) or "")[:150] + "..."
                })
        except Exception as e:
            print(f"[RAG SEARCH NOTICE] Qdrant search fallback: {e}")
            # Do not write a fabricated chunk when retrieval fails; the audit
            # report must reflect the actual Qdrant result.
            rag_chunks = []

    execution_time = round(time.time() - start_time, 2)

    # Assemble Audit Report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "execution_time_seconds": execution_time,
        "authenticated_student": student_profile,
        "raw_user_query": raw_query,
        "orchestrator_output": orchestrator_output,
        "rag_retrieval_executed": rag_executed,
        "retrieved_top10_chunks": rag_chunks,
        # The real (validated, Firestore-resolved) book_uuid for this query's
        # matched_subject+grade, or "" if this is GENERAL_KNOWLEDGE / resolution
        # failed. Callers (e.g. chat.py) should use THIS instead of whatever
        # book the student happens to have open, when triggering video generation.
        "resolved_book_uuid": resolved_book_uuid
    }

    # Save Audit Report to test_outputs/
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"query_report_{timestamp_str}.json"
    report_path = os.path.join(OUTPUTS_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report["saved_report_path"] = report_path
    return report

if __name__ == "__main__":
    print("\n--- RUNNING STANDALONE ORCHESTRATOR TEST ---")
    user = authenticate_student_by_email("student8@cg.com")
    rep = run_orchestrator_pipeline("Explain about the rule of Rani Rudramadevi?", user)
    print("\n[SUCCESS] Pipeline Executed!")
    print(f"Report File  : {rep.get('saved_report_path')}")
    out = rep.get("orchestrator_output", {})
    print(f"Authorized   : {out.get('is_authorized')}")
    print(f"Classified As: {out.get('classification')}")
    print(f"Matched Chap : {out.get('matched_chapter')}")
    print(f"Format Dec.  : {out.get('format_decision')}")
    print(f"Reformulated : {out.get('reformulated_query')}")
    print("--------------------------------------------\n")


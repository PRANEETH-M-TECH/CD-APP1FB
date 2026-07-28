import json
import time
import datetime
import logging
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.services.chat.answer_service import (
    load_summary_from_firestore,
    reformulate_with_llm,
    context_aware_reformulate,
    generate_smart_followups,
    generate_teacher_explanation
)
from backend.app.services.chat.session_service import session_manager
from backend.app.services.chat.intent_classifier import determine_next_action
from backend.app.services.chat.conversation import conversation_manager
from backend.app.services.analytics import analytics_service
from backend.app.services.analytics import enhanced_analytics
from backend.app.prompts import styler as prompt_styler
from backend.app.core.auth_middleware import get_user_id_or_default
from backend.app.core.firebase.firebase_init import db
from backend.app.core import firestore_service
from backend.app.core.firestore_service import check_global_query_cache, save_to_global_query_cache
from backend.app.orchestrator_test.test_runner import run_orchestrator_pipeline
from backend.app.services.deployment_logger import save_chat_log_background

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    class_name: str
    subject: str
    book_uuid: str

class SummaryRequest(BaseModel):
    class_name: str
    subject: str
    chapter_name: str


def track_cumulative_analytics(uid: str, query: str, subject: str, chapter_name: str = "Unknown"):
    """
    Track cumulative analytics for persistent dashboard stats.
    """
    from datetime import datetime
    try:
        logger.info(f"[CUMULATIVE ANALYTICS] Starting tracking for uid: {uid}, subject: {subject}, chapter: {chapter_name}")
        doc_ref = db.collection('user_analytics').document(uid)
        doc = doc_ref.get()
        
        today = datetime.now().strftime('%Y-%m-%d')
        week = datetime.now().strftime('%Y-W%W')
        
        if doc.exists:
            data = doc.to_dict()
        else:
            data = {
                'total_queries_all_time': 0,
                'total_subjects_explored': 0,
                'current_streak': 0,
                'longest_streak': 0,
                'last_activity_date': None,
                'daily_stats': {},
                'weekly_stats': {},
                'subjects_set': [],
                'created_at': datetime.now().isoformat()
            }
        
        data['total_queries_all_time'] = data.get('total_queries_all_time', 0) + 1
        
        if subject and subject.lower() not in [s.lower() for s in data.get('subjects_set', [])]:
            if 'subjects_set' not in data:
                data['subjects_set'] = []
            data['subjects_set'].append(subject.lower())
            data['total_subjects_explored'] = len(data['subjects_set'])
        
        if 'daily_stats' not in data:
            data['daily_stats'] = {}
        
        if today not in data['daily_stats']:
            data['daily_stats'][today] = {
                'queries_count': 0,
                'subjects': [],
                'chapters': []
            }
        
        data['daily_stats'][today]['queries_count'] += 1
        
        if subject and subject.lower() not in [s.lower() for s in data['daily_stats'][today].get('subjects', [])]:
            if 'subjects' not in data['daily_stats'][today]:
                data['daily_stats'][today]['subjects'] = []
            data['daily_stats'][today]['subjects'].append(subject.lower())
        
        if chapter_name and chapter_name not in data['daily_stats'][today].get('chapters', []):
            if 'chapters' not in data['daily_stats'][today]:
                data['daily_stats'][today]['chapters'] = []
            data['daily_stats'][today]['chapters'].append(chapter_name)
        
        if 'weekly_stats' not in data:
            data['weekly_stats'] = {}
            
        if week not in data['weekly_stats']:
            data['weekly_stats'][week] = {
                'queries_count': 0,
                'subjects': [],
                'active_days': []
            }
        
        data['weekly_stats'][week]['queries_count'] += 1
        
        if subject and subject.lower() not in [s.lower() for s in data['weekly_stats'][week].get('subjects', [])]:
            if 'subjects' not in data['weekly_stats'][week]:
                data['weekly_stats'][week]['subjects'] = []
            data['weekly_stats'][week]['subjects'].append(subject.lower())
        
        if today not in data['weekly_stats'][week].get('active_days', []):
            if 'active_days' not in data['weekly_stats'][week]:
                data['weekly_stats'][week]['active_days'] = []
            data['weekly_stats'][week]['active_days'].append(today)
        
        last_date = data.get('last_activity_date')
        if last_date:
            try:
                last = datetime.strptime(last_date, '%Y-%m-%d')
                today_dt = datetime.strptime(today, '%Y-%m-%d')
                diff = (today_dt - last).days
                
                if diff == 1:
                    data['current_streak'] = data.get('current_streak', 0) + 1
                elif diff == 0:
                    pass
                else:
                    data['current_streak'] = 1
            except Exception as e:
                logger.warning(f"[ANALYTICS] Error calculating streak: {e}")
                data['current_streak'] = 1
        else:
            data['current_streak'] = 1
        
        if data.get('current_streak', 0) > data.get('longest_streak', 0):
            data['longest_streak'] = data['current_streak']
        
        data['last_activity_date'] = today
        data['last_updated'] = datetime.now().isoformat()
        
        doc_ref.set(data)
        print(f"[CUMULATIVE ANALYTICS] Tracked for {uid}: Total={data['total_queries_all_time']}, Streak={data['current_streak']}, Subjects={data['total_subjects_explored']}")
        
    except Exception as e:
        logger.error(f"[CUMULATIVE ANALYTICS] Failed to track analytics for {uid}: {e}", exc_info=True)


@router.get("/api/query", tags=["LLM"])
async def query_engine(
    book_uuid: str = Query(...),
    query: str = Query(...),
    class_name: str = Query(...),
    subject: str = Query(...)
):
    """
    Streams the answer in real-time using Server-Sent Events (SSE).
    """
    async def event_generator():
        from backend.app.utils.gemini_tracker import request_stats
        request_stats.set({"calls": [], "start_time": time.time(), "query": query})
        start = time.time()
        print(f"\n{'='*80}")
        print(f"[QUERY] New query received at {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"[QUERY] User question: {query}")
        print(f"[QUERY] Book: Class {class_name} - {subject.capitalize()}")
        print(f"[QUERY] Book UUID: {book_uuid[:16]}...")
        print(f"{'='*80}\n")
        
        print(f"[FIRESTORE] Loading summaries from cache/Firestore...")
        summary_doc = load_summary_from_firestore(class_name, subject)
        chapters = summary_doc["chapters"]
        print(f"[FIRESTORE] Loaded {len(chapters)} chapters\n")
        
        try:
            reform = reformulate_with_llm(
                raw_query=query,
                class_name=class_name,
                subject=subject,
                chapters=chapters
            )
            
            reformulated_query = reform.get("reformulated_query", query)
            classification = reform.get("classification", "general")
            chapter_ranking = reform.get("chapter_ranking", [])
            
            print("\n" + "="*40)
            print("RAW USER QUESTION:")
            print(f"   \"{query}\"")
            print("="*40 + "\n")
            
            print("="*40)
            print("REFORMULATED QUERY:")
            print(f"   \"{reformulated_query}\"")
            print(f"CLASSIFICATION          : {classification}")
            print(f"TOP CHAPTERS IDENTIFIED : {len(chapter_ranking)}")
            print("="*40 + "\n")
            
        except Exception as e:
            print(f"[REFORMULATE] Error: {e}")
            reformulated_query = query
            classification = "general"
            chapter_ranking = chapters[:5]
        
        print(f"[SIMILARITY] Calculating semantic similarity scores for chapters...")
        try:
            from sentence_transformers import util
            query_embedding = qdrant.local_embedder.encode(reformulated_query, convert_to_tensor=True)
            
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
            
            scored_chapters.sort(key=lambda x: x['relevance_score'], reverse=True)
            chapter_ranking = scored_chapters[:5]
            
            print(f"[SIMILARITY] Calculated similarity scores for {len(scored_chapters)} chapters")
        except Exception as e:
            print(f"[SIMILARITY] Error calculating similarity: {e}")
            if chapter_ranking:
                for ch in chapter_ranking:
                    if 'score' in ch and 'relevance_score' not in ch:
                        ch['relevance_score'] = ch['score']
                    elif 'relevance_score' not in ch:
                        ch['relevance_score'] = 0.0
        
        print(f"[RETRIEVAL] Performing hybrid search...")
        metadata = qdrant.get_book_metadata(book_uuid)
        
        processed_data = qdrant.reformulate_and_classify_query(
            query=reformulated_query,
            class_name=metadata.get("class_name"),
            subject=metadata.get("subject"),
            chapter_list=[ch["chapter_name"] for ch in chapter_ranking]
        )
        
        keywords = processed_data.get("keywords", [])
        conceptual_score = processed_data.get("conceptual_score", 0.5)
        
        top_chapter_names = [ch["chapter_name"] for ch in chapter_ranking[:5]]
        
        cleaned_keywords = []
        for kw in keywords:
            if isinstance(kw, dict):
                cleaned_keywords.append({"keyword": kw.get("keyword", ""), "importance": kw.get("importance", 0.5)})
            else:
                cleaned_keywords.append({"keyword": str(kw), "importance": 0.5})

        hybrid_results, semantic_results, bm25_results = qdrant.hybrid_search(
            book_uuid=book_uuid,
            query=reformulated_query,
            keywords=cleaned_keywords,
            conceptual_score=conceptual_score,
            metadata_filters={"chapter_names": top_chapter_names}
        )
        
        context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
        
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
        
        print(f"[LLM] Streaming answer...")
        final_prompt = prompt_styler.get_answer_prompt(
            class_name=class_name,
            subject=subject,
            query=reformulated_query,
            context=context
        )
        
        full_answer = ""
        try:
            response_stream = qdrant.gemini_client.models.generate_content_stream(
                model=qdrant.generation_model_name,
                contents=final_prompt
            )
            
            for chunk in response_stream:
                try:
                    if chunk.text:
                        full_answer += chunk.text
                        event_data = json.dumps({
                            "display_text": chunk.text,
                            "read_text": chunk.text 
                        })
                        yield f"data: {event_data}\n\n"
                        await asyncio.sleep(0.01)
                except ValueError:
                    pass
            
            print(f"[LLM] Answer streamed ({len(full_answer)} characters)\n")
        except Exception as e:
            print(f"[LLM] Error generating answer: {e}\n")
            error_msg = "Sorry, I couldn't generate the answer."
            full_answer = error_msg
            event_data = json.dumps({"display_text": error_msg, "read_text": error_msg})
            yield f"data: {event_data}\n\n"
        
        yield "data: [DONE]\n\n"
        
        # Write to ans.txt
        try:
            with open("ans.txt", "w", encoding="utf-8") as f:
                f.write(f"{'='*80}\n")
                f.write(f"QUERY LOG - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"1. ORIGINAL QUERY:\n   {ans_txt_data['query']}\n\n")
                f.write(f"2. REFORMULATED QUERY:\n   {ans_txt_data['reformulated_query']}\n")
                f.write(f"   Classification: {ans_txt_data['classification']}\n")
                f.write(f"   Conceptual Score: {ans_txt_data['conceptual_score']:.2f}\n\n")
                f.write(f"3. CHAPTER RANKING:\n")
                for idx, ch in enumerate(ans_txt_data['chapter_ranking'], 1):
                    f.write(f"   {idx}. {ch['chapter_name']} (relevance: {ch.get('relevance_score', 'N/A')})\n")
                f.write(f"\n4. GENERATED ANSWER:\n{full_answer}\n\n")
        except Exception as e:
            print(f"[LOG] ✗ Error writing to ans.txt: {e}\n")
            
        try:
            rag_chunks = []
            if "hybrid_results" in ans_txt_data and ans_txt_data["hybrid_results"]:
                for score, doc in ans_txt_data["hybrid_results"][:5]:
                    rag_chunks.append({
                        "chunk_id": doc.get("chunk_id", "chunk_unknown"),
                        "text": doc.get("text", "")[:200],
                        "score": round(score, 3)
                    })
            save_chat_log_background(
                user_query=query,
                subject=subject,
                mode="text_to_text",
                session_id=None,
                retrieved_rag_chunks=rag_chunks,
                llm_response=full_answer,
                execution_time_ms=int((time.time() - start) * 1000)
            )
        except Exception as log_err:
            logger.error(f"[DeploymentLogger] Failed to log query_engine: {log_err}")

        from backend.app.utils.gemini_tracker import print_query_performance_report
        print_query_performance_report()
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/smart_query", tags=["LLM"])
async def smart_query_engine(
    request: Request,
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
    # Resolve FastAPI Query default objects if called programmatically
    if hasattr(session_id, "__class__") and session_id.__class__.__name__ == "Query":
        session_id = None
    if hasattr(is_clicked_followup, "__class__") and is_clicked_followup.__class__.__name__ == "Query":
        is_clicked_followup = False
    if hasattr(book_uuid, "__class__") and book_uuid.__class__.__name__ == "Query":
        book_uuid = ""
    if hasattr(query, "__class__") and query.__class__.__name__ == "Query":
        query = ""
    if hasattr(class_name, "__class__") and class_name.__class__.__name__ == "Query":
        class_name = ""
    if hasattr(subject, "__class__") and subject.__class__.__name__ == "Query":
        subject = ""

    async def event_generator():
        from backend.app.utils.gemini_tracker import request_stats
        request_stats.set({"calls": [], "start_time": time.time(), "query": query})
        uid = get_user_id_or_default(request)
        start_time = time.time()
        print(f"\n============================================================")
        print(f"USER QUESTION (ORCHESTRATOR PATH): '{query}'")
        print(f"============================================================")

        try:
            # 1. Load summary list for grade mapping
            session = session_manager.get_or_create_session(book_uuid, session_id)
            
            # Get authenticated student profile from Firestore
            student_profile = {
                "uid": uid,
                "email": "anonymous@cg.com",
                "name": "Sonu",
                "class": 8,
                "board": "CBSE",
                "role": "student"
            }
            if uid and uid != "anonymous":
                user_doc = db.collection("users").document(uid).get()
                if user_doc.exists:
                    udata = user_doc.to_dict()
                    # Strip all string fields fetched during auth
                    student_profile = {
                        "uid": uid,
                        "email": str(udata.get("email", "")).strip(),
                        "name": str(udata.get("name", "Sonu")).strip(),
                        "class": int(udata.get("class", 8)) if udata.get("class") is not None else 8,
                        "board": str(udata.get("board", "CBSE")).strip(),
                        "role": str(udata.get("role", "student")).strip()
                    }

            # 2. Check global cache hit
            cached = check_global_query_cache(query, student_profile["class"], subject)
            if cached:
                out = cached["orchestrator_output"]
                interactive_url = cached.get("interactive_url")
                
                classification = out.get("classification", "CURRICULUM")
                matched_subject = out.get("matched_subject")
                matched_chapter = out.get("matched_chapter")
                format_decision = out.get("format_decision", "QUICK_ANSWER")

                print(f"[CACHE HIT] Reusing cached query payload. Format: {format_decision}")
                yield f"data: {json.dumps({'type': 'intent', 'intent': classification, 'subject': matched_subject, 'chapter': matched_chapter, 'format': format_decision})}\n\n"
                
                # Stream pre-cached text_narration sentence-by-sentence for synchronized TTS pipeline
                text_script = out.get("text_narration") or ""
                import re
                cached_sentences = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', text_script) if s.strip()]
                if not cached_sentences:
                    cached_sentences = [text_script]

                for s in cached_sentences:
                    yield f"data: {json.dumps({'display_text': s + ' '})}\n\n"
                    await asyncio.sleep(0.05)
                
                # If a video lesson is ready, yield metadata details
                if format_decision == "VIDEO_REQUIRED" and interactive_url:
                    yield f"data: {json.dumps({'type': 'progress', 'step': 'launching_lesson', 'status': 'complete', 'message': 'Pre-rendered lesson loaded from cache!'})}\n\n"
                    await asyncio.sleep(0.5)
                    
                    ready_payload = {
                        "type": "lesson_ready",
                        "lesson_id": interactive_url.split("/")[-2] if "/" in interactive_url else "cached",
                        "lesson_title": out.get("video_storyboard", {}).get("lesson_title", "Cached Lesson"),
                        "interactive_url": interactive_url,
                        "html_url": interactive_url,
                        "video_url": None,
                        "scene_count": len(out.get("video_storyboard", {}).get("scenes", [])) if isinstance(out.get("video_storyboard"), dict) else 0,
                        "lesson": out.get("video_storyboard"),
                        "lesson_package": out.get("video_storyboard")
                    }
                    yield f"data: {json.dumps(ready_payload)}\n\n"
                
                yield "data: [DONE]\n\n"
                return

            # 3. Cache Miss: Run Orchestrator Pipeline
            # Run in thread executor so the async event loop is NOT blocked during LLM calls
            # Propagate ContextVars (tracking context) using copy_context().run to fix 0-stats issue
            print(f"[CACHE MISS] Calling single-pass Orchestrator Agent...")
            import contextvars
            ctx = contextvars.copy_context()
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(
                None,  # uses the default ThreadPoolExecutor
                ctx.run,
                run_orchestrator_pipeline,
                query,
                student_profile
            )
            out = report.get("orchestrator_output", {})


            classification = out.get("classification", "CURRICULUM")
            matched_subject = out.get("matched_subject")
            matched_chapter = out.get("matched_chapter")
            format_decision = out.get("format_decision", "QUICK_ANSWER")
            text_script = out.get("text_narration") or ""

            # Check Child Safety Refusal
            if not out.get("is_authorized", True):
                refusal = out.get("refusal_reason") or "I cannot answer this query."
                yield f"data: {json.dumps({'type': 'intent', 'intent': 'UNAUTHORIZED', 'format': 'QUICK_ANSWER'})}\n\n"
                words = refusal.split(" ")
                for w in words:
                    yield f"data: {json.dumps({'display_text': w + ' '})}\n\n"
                    await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
                return

            # Yield classification intent (this tells frontend about subject/chapter metadata for backgrounds)
            yield f"data: {json.dumps({'type': 'intent', 'intent': classification, 'subject': matched_subject, 'chapter': matched_chapter, 'format': format_decision})}\n\n"

            # Stream text narration sentence-by-sentence so TTS pipeline receives clean scene/sentence chunks
            import re
            sentences = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', text_script) if s.strip()]
            if not sentences:
                sentences = [text_script]

            for s in sentences:
                yield f"data: {json.dumps({'display_text': s + ' '})}\n\n"
                await asyncio.sleep(0.05)

            # If format decision is video required, compile the video lesson asynchronously in the background
            interactive_url = None
            if format_decision == "VIDEO_REQUIRED":
                print("[ORCHESTRATOR] Starting background Hyperframes video generation...")
                from backend.app.services.visual_learning.visual_learning_service import generate_visual_lesson_stream
                
                # Fetch pre-compiled storyboard from orchestrator agent result
                storyboard_payload = out.get("video_storyboard")
                
                # We feed the pre-computed storyboard directly to visual_learning_stream
                visual_stream = generate_visual_lesson_stream(
                    query=query,
                    book_uuid=book_uuid,
                    class_name=str(student_profile["class"]),
                    subject=subject,
                    precomputed_storyboard=storyboard_payload
                )

                async for sse_chunk in visual_stream:
                    # Strip 'data: ' prefix if present and parse
                    raw_data = sse_chunk.strip()
                    if raw_data.startswith("data: "):
                        raw_data = raw_data[6:]
                    
                    try:
                        chunk_json = json.loads(raw_data)
                        
                        # Forward progress steps to frontend
                        if chunk_json.get("type") == "progress":
                            yield f"data: {json.dumps(chunk_json)}\n\n"
                        
                        # Handle ready payload
                        if chunk_json.get("type") == "lesson_ready":
                            interactive_url = chunk_json.get("interactive_url")
                            yield f"data: {json.dumps(chunk_json)}\n\n"
                    except Exception as json_err:
                        logger.error(f"[SSE FORWARD] Parse error: {json_err} on raw: {raw_data}")

            # Register compiled query results to the global cache
            save_to_global_query_cache(
                raw_query=query,
                class_name=student_profile["class"],
                subject=subject,
                orchestrator_output=out,
                interactive_url=interactive_url
            )

            # Save query turn to standard chat session manager
            turn_data = {
                "query": query,
                "reformulated": out.get("reformulated_query", query),
                "answer": text_script,
                "intent_type": classification,
                "is_clicked_followup": is_clicked_followup,
                "timestamp": datetime.datetime.now().isoformat()
            }
            session_manager.add_turn(session["session_id"], turn_data)

            yield "data: [DONE]\n\n"
            from backend.app.utils.gemini_tracker import print_query_performance_report
            print_query_performance_report()

        except Exception as e:
            logger.error(f"[ORCHESTRATE ROUTE ERROR] Failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            from backend.app.utils.gemini_tracker import print_query_performance_report
            print_query_performance_report()
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/session/history", tags=["Session"])
async def get_session_history(session_id: str = Query(...)):
    """
    Returns complete chat history with metadata for a given session.
    """
    from backend.app.core.redis_service import redis_service
    session = redis_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    total_turns = len(session.get("full_history", []))
    cache_hits = sum(1 for turn in session.get("full_history", []) if turn.get("intent_type") == "USE_CACHED_CONTEXT")
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


@router.get("/api/session/chunks", tags=["Session"])
async def get_current_chunks(session_id: str = Query(...)):
    """
    Returns currently cached chunks for the active topic.
    """
    chunks = session_manager.get_current_topic_chunks(session_id)
    if not chunks:
        return {"chunks": [], "total_count": 0, "message": "No cached chunks for current topic"}
    
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


@router.post("/api/summarize")
async def get_summary(request: SummaryRequest):
    """
    Generates a teacher-like explanation for a specific chapter of a book.
    """
    class_name = request.class_name
    subject = request.subject
    chapter_name = request.chapter_name

    summary_doc = firestore_service.load_summary_from_firestore(class_name, subject)
    
    chapter_summary = None
    if summary_doc and "chapters" in summary_doc:
        for chap in summary_doc["chapters"]:
            if chap.get("chapter_name") == chapter_name:
                chapter_summary = chap.get("summary")
                break
    
    if chapter_summary is None or chapter_summary == "":
        raise HTTPException(status_code=404, detail="Summary not found for this chapter or is being generated.")

    explanation = generate_teacher_explanation(
        class_name=class_name,
        subject=subject,
        chapter_name=chapter_name,
        summary_text=chapter_summary
    )
    
    return {"summary": explanation}


@router.websocket("/ws/conversation/{conversation_id}")
async def websocket_conversation(
    websocket: WebSocket,
    conversation_id: str,
    book_uuid: str,
    uid: str = Query(None),
    class_name: str = Query(None),
    subject: str = Query(None)
):
    """
    Handles conversational voice/text WebSockets with dynamic interruption support.
    """
    await conversation_manager.connect(websocket, conversation_id, book_uuid, uid, class_name, subject)
    print(f"[App] WebSocket handler started for conversation_id={conversation_id}, book_uuid={book_uuid}, uid={uid}, class_name={class_name}, subject={subject}")
    try:
        while True:
            data = await websocket.receive_json()
            print(f"[App] Received WS message for {conversation_id}: {str(data)[:200]}")
            
            if data.get("type") == "query":
                print(f"[App] Dispatching 'query' to ConversationManager for {conversation_id}")
                asyncio.create_task(conversation_manager.process_query(conversation_id, data.get("query", "")))
            elif data.get("type") == "interrupt":
                print(f"[App] Received 'interrupt' for {conversation_id}")
                await conversation_manager.interrupt(conversation_id)
    
    except WebSocketDisconnect:
        print(f"[App] WebSocket disconnected for conversation_id={conversation_id}")
    except Exception as exc:
        print(f"[App] WebSocket error for {conversation_id}: {exc}")
    finally:
        conversation_manager.disconnect(conversation_id)

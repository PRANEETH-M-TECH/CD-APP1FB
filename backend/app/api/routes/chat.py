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
    async def event_generator():
        from backend.app.utils.gemini_tracker import request_stats
        request_stats.set({"calls": [], "start_time": time.time(), "query": query})
        uid = get_user_id_or_default(request)
        start_time = time.time()
        print(f"\n============================================================")
        print(f"USER QUESTION: '{query}'")
        print(f"============================================================")

        try:
            session = session_manager.get_or_create_session(book_uuid, session_id)
            active_context_window = session["active_context_window"]
            
            last_action = None
            if active_context_window:
                last_action = active_context_window[-1].get("intent_type")
            
            action_details = determine_next_action(
                current_query=query,
                conversation_window=active_context_window,
                gemini_client=qdrant.gemini_client,
                generation_model_name=qdrant.generation_model_name,
                embedder=qdrant.local_embedder,
                is_clicked_followup=is_clicked_followup,
                last_action=last_action
            )
            action = action_details.get("action")
            reason = action_details.get("reason", "No reason provided.")
            similarity_score = action_details.get("similarity_score", 0.0)
            tier = action_details.get("tier", "UNKNOWN")
            
            print(f"\nINTENT CLASSIFIER DECISION: Tier {tier} | Action {action}")
            yield f"data: {json.dumps({'type': 'intent', 'intent': action})}\n\n"

            context = ""
            hybrid_results = []
            reformulated_query = query
            keywords = []
            full_answer = ""

            if action == "RETRIEVE_NEW_CONTEXT":
                new_topic_name = action_details.get("new_topic_name", "New Topic")
                session_manager.start_new_topic(session['session_id'], new_topic_name)

                summary_doc = load_summary_from_firestore(class_name, subject)
                chapters = summary_doc.get("chapters", [])
                
                reform = reformulate_with_llm(query, class_name, subject, chapters)
                reformulated_query = reform.get("reformulated_query", query)
                keywords = reform.get("keywords", [])
                chapter_ranking = reform.get("chapter_ranking", [])
                conceptual_score = reform.get("conceptual_score", 0.5)
                
                cleaned_keywords = []
                for kw in keywords:
                    if isinstance(kw, dict):
                        cleaned_keywords.append({"keyword": kw.get("keyword", ""), "importance": kw.get("importance", 0.5)})
                    else:
                        cleaned_keywords.append({"keyword": str(kw), "importance": 0.5})

                top_chapter_names = [ch["chapter_name"] for ch in chapter_ranking[:5]] if chapter_ranking else []
                hybrid_results, _, _ = qdrant.hybrid_search(
                    book_uuid=book_uuid,
                    query=reformulated_query,
                    keywords=cleaned_keywords,
                    conceptual_score=conceptual_score,
                    metadata_filters={"chapter_names": top_chapter_names} if top_chapter_names else {}
                )
                context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
                session_manager.update_topic_chunks(session['session_id'], hybrid_results)

            elif action == "USE_CACHED_CONTEXT":
                cached_chunks = session_manager.get_current_topic_chunks(session['session_id'])
                if cached_chunks:
                    hybrid_results = cached_chunks
                    context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
                else:
                    action = "RETRIEVE_NEW_CONTEXT"
                    summary_doc = load_summary_from_firestore(class_name, subject)
                    chapters = summary_doc.get("chapters", [])
                    reform = reformulate_with_llm(query, class_name, subject, chapters)
                    reformulated_query = reform.get("reformulated_query", query)
                    keywords = reform.get("keywords", [])
                    chapter_ranking = reform.get("chapter_ranking", [])
                    
                    cleaned_keywords = []
                    for kw in keywords:
                        if isinstance(kw, dict):
                            cleaned_keywords.append({"keyword": kw.get("keyword", ""), "importance": kw.get("importance", 0.5)})
                        else:
                            cleaned_keywords.append({"keyword": str(kw), "importance": 0.5})

                    top_chapter_names = [ch["chapter_name"] for ch in chapter_ranking[:5]] if chapter_ranking else []
                    hybrid_results, _, _ = qdrant.hybrid_search(
                        book_uuid=book_uuid, query=reformulated_query, keywords=cleaned_keywords,
                        metadata_filters={"chapter_names": top_chapter_names} if top_chapter_names else {}
                    )
                    context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:10]])
                    session_manager.update_topic_chunks(session['session_id'], hybrid_results)

                reform = context_aware_reformulate(query, active_context_window)
                reformulated_query = reform.get("reformulated_query", query)

            if action in ["RETRIEVE_NEW_CONTEXT", "USE_CACHED_CONTEXT"]:
                conversation_context = "\n\nPREVIOUS CONVERSATION HISTORY:\n"
                for turn in session.get("full_history", [])[-3:]:
                    conversation_context += f"Q: {turn['query']}\nA: {turn.get('answer', 'N/A')[:200]}...\n\n"

                final_prompt = prompt_styler.get_answer_prompt(
                    class_name=class_name,
                    subject=subject,
                    query=reformulated_query,
                    context=context,
                    conversation_context=conversation_context,
                    action=action
                )
            else: # ANSWER_FROM_HISTORY
                context_summary = ""
                for turn in session.get("full_history", []):
                    answer_preview = turn.get('answer', '')[:200]
                    if len(turn.get('answer', '')) > 200:
                        answer_preview += "..."
                    context_summary += f"Q: {turn['query']}\nA: {answer_preview}\n\n"
                
                final_prompt = prompt_styler.get_answer_prompt(
                    class_name=class_name,
                    subject=subject,
                    query=query,
                    context="",
                    conversation_context=context_summary,
                    action=action
                )

            response_stream = qdrant.gemini_client.models.generate_content_stream(
                model=qdrant.generation_model_name,
                contents=final_prompt
            )
            
            for chunk in response_stream:
                text_val = None
                try:
                    if hasattr(chunk, "text") and chunk.text:
                        text_val = chunk.text
                except Exception:
                    text_val = None
                
                if text_val:
                    full_answer += text_val
                    yield f"data: {json.dumps({'display_text': text_val})}\n\n"
                    await asyncio.sleep(0)

            follow_ups = []
            if action != "ANSWER_FROM_HISTORY":
                follow_ups = generate_smart_followups(reformulated_query, full_answer, hybrid_results[:5])
            
            yield f"data: {json.dumps({'type': 'followups', 'followups': follow_ups})}\n\n"

            turn_data = {
                "query": query, "reformulated": reformulated_query, "answer": full_answer,
                "intent_type": action, "is_clicked_followup": is_clicked_followup,
                "tier": tier, "similarity_score": similarity_score,
                "follow_ups": follow_ups, "timestamp": datetime.datetime.now().isoformat()
            }
            if action == "RETRIEVE_NEW_CONTEXT":
                 turn_data["context_cache"] = {"retrieved_chunks": hybrid_results, "context": context, "keywords": keywords}

            session_manager.add_turn(session["session_id"], turn_data)
            
            try:
                chapter_id = None
                chapter_name = "Unknown"
                if hybrid_results and len(hybrid_results) > 0:
                    first_chunk = hybrid_results[0][1]
                    chapter_id = first_chunk.get("chapter_id")
                    chapter_name = first_chunk.get("chapter_name", "Unknown")
                
                mode = "text"
                if request and hasattr(request, "headers"):
                    if request.headers.get("X-Client-Mode") == "voice":
                        mode = "voice"
                
                analytics_service.log_query(
                    uid=uid, class_name=class_name, subject=subject, chapter_id=chapter_id,
                    chapter_name=chapter_name, query=query, reformulated_query=reformulated_query,
                    mode=mode, llm_action=action, answer_length=len(full_answer)
                )
                
                analytics_service.update_user_stats(uid=uid, subject=subject, chapter_id=chapter_id, class_name=class_name)
                
                if chapter_id:
                    analytics_service.update_chapter_stats(class_name=class_name, subject=subject, chapter_id=chapter_id, chapter_name=chapter_name, uid=uid)
                    topics_to_track = keywords if keywords else [reformulated_query[:50]]
                    enhanced_analytics.track_topic_analytics(
                        uid=uid, subject=subject, chapter_id=chapter_id, chapter_name=chapter_name,
                        topics=topics_to_track, difficulty_score=0.5
                    )
                    
                    enhanced_analytics.update_frequent_questions(uid=uid, query=query, chapter_name=chapter_name, subject=subject)
                
                track_cumulative_analytics(uid=uid, query=query, subject=subject, chapter_name=chapter_name)
                
            except Exception as analytics_error:
                logger.error(f"[ANALYTICS] Analytics logging failed: {analytics_error}")

            try:
                mistake_metadata = {"patterns": [], "confusion_topics": [], "recommended_tasks": []}
                analytics_service.update_mistake_patterns(
                    uid=uid, patterns=mistake_metadata["patterns"],
                    confusion_topics=mistake_metadata["confusion_topics"],
                    recommended_tasks=mistake_metadata["recommended_tasks"]
                )
            except Exception as e:
                pass

            updated_history = session_manager.get_full_history(session["session_id"])
            current_turn_number = len(updated_history)
            
            chunks_summary = []
            if hybrid_results:
                for score, doc in hybrid_results[:5]:
                    chunks_summary.append({
                        "chapter_name": doc.get("chapter_name", "Unknown"),
                        "relevance_score": round(score, 3),
                        "text_preview": doc.get("text", "")[:150] + "...",
                        "pdf_pages": f"{doc.get('pdf_startpg', '?')}-{doc.get('pdf_endpg', '?')}",
                        "chapter_pages": f"{doc.get('chpstpage', '?')}-{doc.get('chpendpage', '?')}"
                    })
            
            cache_hit = (action == "USE_CACHED_CONTEXT")
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
                    "retrieval_time_saved_ms": 1500 if cache_hit else 0
                },
                "retrieved_chunks": chunks_summary
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            try:
                formatted_chunks = []
                if hybrid_results:
                    for score, doc in hybrid_results[:5]:
                        formatted_chunks.append({
                            "chunk_id": doc.get("chunk_id", "chunk_unknown"),
                            "text": doc.get("text", "")[:200],
                            "score": round(score, 3)
                        })
                save_chat_log_background(
                    user_query=query,
                    subject=subject,
                    mode=mode if 'mode' in locals() else "text_to_text",
                    session_id=session.get("session_id"),
                    retrieved_rag_chunks=formatted_chunks,
                    llm_response=full_answer,
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            except Exception as log_err:
                logger.error(f"[DeploymentLogger] Failed to log smart_query_engine: {log_err}")

            yield "data: [DONE]\n\n"
            from backend.app.utils.gemini_tracker import print_query_performance_report
            print_query_performance_report()
            
        except Exception as e:
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

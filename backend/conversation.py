"""
Conversation state management and optimized processing for real-time interactions.
"""
import asyncio
from typing import Dict, Optional, Set, List
from dataclasses import dataclass
from datetime import datetime
from cachetools import TTLCache
from fastapi import WebSocket
import json
import logging

from . import qdrant  # NEW: Import qdrant module for generation_model and embedder
from . import analytics_service # Import analytics service

logger = logging.getLogger(__name__) # Initialize logger
from .qdrant import (
    get_book_metadata,
    hybrid_search,
    reformulate_and_classify_query,
    generate_conversational_answer
)
from .session_service import session_manager
from .intent_classifier import determine_next_action

@dataclass
class ConversationState:
    book_uuid: str
    websocket: WebSocket
    last_query_time: datetime
    cached_context: Dict
    is_speaking: bool = False
    should_stop: bool = False
    session_id: Optional[str] = None
    turn_count: int = 0

class ConversationManager:
    def __init__(self):
        self.active_conversations: Dict[str, ConversationState] = {}
    
    async def connect(self, websocket: WebSocket, conversation_id: str, book_uuid: str):
        await websocket.accept()
        print(f"[ConversationManager] WebSocket accepted for conversation_id={conversation_id}, book_uuid={book_uuid}")
        self.active_conversations[conversation_id] = ConversationState(
            book_uuid=book_uuid,
            websocket=websocket,
            last_query_time=datetime.now(),
            cached_context={}
        )

    async def _safe_send(self, websocket: WebSocket, payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except Exception as e:
            print(f"[ConversationManager] Failed to send payload to websocket: {e}")
            return False
    
    def disconnect(self, conversation_id: str):
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
    
    async def interrupt(self, conversation_id: str):
        if conversation_id in self.active_conversations:
            conv = self.active_conversations[conversation_id]
            conv.should_stop = True
            if conv.is_speaking:
                await conv.websocket.send_json({
                    "type": "interrupt_acknowledged",
                    "message": "Stopping current response..."
                })

    async def process_query(self, conversation_id: str, query: str):
        if conversation_id not in self.active_conversations:
            return
        
        conv = self.active_conversations[conversation_id]
        print(f"\n{'='*60}")
        print(f"[CONVERSATION] New message from user")
        print(f"  ID: {conversation_id} | Query: {query}")
        print(f"{'='*60}\n")
        
        conv.should_stop = False
        conv.is_speaking = True
        
        try:
            # 1. Get or create session
            session = session_manager.get_or_create_session(conv.book_uuid, conv.session_id)
            conv.session_id = session["session_id"]
            
            # Extract basic query metadata early
            uid = conv.session_id # Use session_id as uid for WebSocket analytics
            book_metadata = qdrant.get_book_metadata(conv.book_uuid)
            class_name = book_metadata.get("class_name", "Unknown")
            subject = book_metadata.get("subject", "Unknown")
            
            active_context_window = session["active_context_window"]
            
            # Detect client mode
            mode = "text"

            # WebSocket mode detection
            if session and session.get("transport") == "websocket":
                if session.get("input_type") == "voice":
                    mode = "voice"
            
            # Extract last action for context awareness (NEW)
            last_action = None
            if active_context_window:
                last_action = active_context_window[-1].get("intent_type")
            
            # 2. Determine next action with semantic similarity and 5-tier routing
            action_details = determine_next_action(
                current_query=query,
                conversation_window=active_context_window,
                generation_model=qdrant.generation_model,
                embedder=qdrant.local_embedder,  # Pass embedder for similarity analysis
                is_clicked_followup=False,  # WebSocket queries are always typed (not clicked)
                last_action=last_action  # NEW: Previous action for context
            )
            action = action_details.get("action")
            similarity_score = action_details.get("similarity_score", 0.0)
            tier = action_details.get("tier", "UNKNOWN")
            
            print(f"[ACTION] Determined Action: {action}")
            print(f"[ACTION] Tier: {tier}")
            print(f"[ACTION] Reason: {action_details.get('reason')}")
            print(f"[ACTION] Similarity Score: {similarity_score:.3f}\n")
            
            await self._safe_send(conv.websocket, {
                'type': 'intent', 
                'intent_type': action,
                'tier': tier,  # NEW: Send tier information
                'similarity_score': similarity_score,  # NEW: Send similarity score
                'cache_hit': action == "USE_CACHED_CONTEXT"  # NEW: Cache hit indicator
            })
            
            # 3. Execute action
            search_results = []
            context = ""
            chapter_id = None
            chapter_name = "Unknown"
            reformulated_query = query # Initialize reformulated_query
            
            if action == "RETRIEVE_NEW_CONTEXT":
                print(f"[PATH] New topic - Full retrieval\n")
                new_topic_name = action_details.get("new_topic_name", "New Topic")
                session_manager.start_new_topic(conv.session_id, new_topic_name)
                
                # Metadata already retrieved earlier
                processed_query_data = reformulate_and_classify_query(
                    query=query,
                    class_name=class_name,
                    subject=subject
                )
                reformulated_query = processed_query_data.get("reformulated_query", query)
                
                search_results, _, _ = hybrid_search(
                    book_uuid=conv.book_uuid,
                    query=reformulated_query,
                    keywords=processed_query_data.get("keywords", []),
                    conceptual_score=processed_query_data.get("conceptual_score", 0.5)
                )
                print(f"[RETRIEVAL] ✓ Retrieved {len(search_results)} chunks\n")
                
                if search_results:
                    context = "\n\n---\n\n".join([payload['text'] for score, payload in search_results])
                    session_manager.update_topic_chunks(conv.session_id, search_results)
                    # Extract chapter info from first retrieved chunk
                    first_chunk = search_results[0][1] # (score, doc) format
                    chapter_id = first_chunk.get("chapter_id")
                    chapter_name = first_chunk.get("chapter_name", "Unknown")

            elif action == "USE_CACHED_CONTEXT":
                print(f"[PATH] ⚡ Follow-up - Reusing cached context\n")
                cached_chunks = session_manager.get_current_topic_chunks(conv.session_id)
                if cached_chunks:
                    search_results = cached_chunks
                    context = "\n\n---\n\n".join([doc["text"] for score, doc in search_results[:10]])
                    print(f"[REUSE] Using {len(search_results)} cached chunks.\n")
                    
                    if search_results: # Extract chapter info from first retrieved chunk
                        first_chunk = search_results[0][1] # (score, doc) format
                        chapter_id = first_chunk.get("chapter_id")
                        chapter_name = first_chunk.get("chapter_name", "Unknown")
                        
                else:
                    print(f"[WARN] No cached chunks found, falling back to fresh search.\n")
                    # Fallback logic here mirrors RETRIEVE_NEW_CONTEXT
                    action = "RETRIEVE_NEW_CONTEXT" # Update action for logging
                    processed_query_data = reformulate_and_classify_query(query=query, class_name=class_name, subject=subject)
                    reformulated_query = processed_query_data.get("reformulated_query", query)
                    
                    search_results, _, _ = hybrid_search(
                        book_uuid=conv.book_uuid,
                        query=reformulated_query,
                        keywords=processed_query_data.get("keywords", []),
                        conceptual_score=processed_query_data.get("conceptual_score", 0.5)
                    )
                    if search_results:
                        context = "\n\n---\n\n".join([payload['text'] for score, payload in search_results])
                        session_manager.update_topic_chunks(conv.session_id, search_results)
                        
                        first_chunk = search_results[0][1] # (score, doc) format
                        chapter_id = first_chunk.get("chapter_id")
                        chapter_name = first_chunk.get("chapter_name", "Unknown")

            elif action == "ANSWER_FROM_HISTORY":
                 print(f"[PATH] 🗣️ Answering from history.\n")
                 context = "No retrieval needed. Answer from history." # Placeholder

            # 4. Stream the answer
            full_answer = ""
            if context:
                conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
                for turn in active_context_window[-3:]:
                    conversation_context += f"Q: {turn['query']}\nA: {turn.get('answer', 'N/A')[:200]}...\n\n"
                
                if action == "ANSWER_FROM_HISTORY":
                    final_prompt = f"Answer the user's current question based only on the provided conversation history.\n\n{conversation_context}\nCURRENT QUESTION: {query}"
                else:
                    final_prompt = f"{conversation_context}\nCURRENT QUESTION: {query}\n\nRETRIEVED INFORMATION:\n{context}\n\nAnswer the current question using the information."

                async for chunk in self._stream_answer(conv, final_prompt):
                    if conv.should_stop:
                        await self._safe_send(conv.websocket, {"type": "interrupted"})
                        break
                    
                    full_answer += chunk
                    await self._safe_send(conv.websocket, {"type": "chunk", "content": chunk})
                
                if not conv.should_stop:
                    # 5. Generate and send follow-ups
                    from .app import generate_smart_followups
                    followups = generate_smart_followups(query, full_answer, search_results[:5])
                    await self._safe_send(conv.websocket, {"type": "followups", "followups": followups})
                    
                    # 6. Save turn to session
                    turn_data = {
                        "query": query, "answer": full_answer, "intent_type": action,
                        "follow_ups": followups, "timestamp": datetime.now().isoformat()
                    }
                    if action == "RETRIEVE_NEW_CONTEXT":
                        turn_data["context_cache"] = {"retrieved_chunks": search_results, "context": context}
                    
                    session_manager.add_turn(conv.session_id, turn_data)
                    print(f"[SESSION] Saved turn to session {conv.session_id}\n")
            else:
                await self._safe_send(conv.websocket, {"type": "error", "message": "No relevant information found."})
        
        except Exception as e:
            print(f"[ERROR] processing query in ConversationManager: {e}")
            import traceback
            traceback.print_exc()
            await self._safe_send(conv.websocket, {"type": "error", "message": str(e)})
        finally:
            # Consolidate analytics logging for WebSocket
            try:
                analytics_service.log_query(
                    uid=uid,
                    class_name=class_name,
                    subject=subject,
                    chapter_id=chapter_id,
                    chapter_name=chapter_name,
                    query=query,
                    reformulated_query=reformulated_query,
                    mode=mode,
                    llm_action=action,
                    answer_length=len(full_answer)
                )

                analytics_service.update_user_stats(uid, subject, chapter_id, class_name)
                analytics_service.update_chapter_stats(class_name, subject, chapter_id, chapter_name, uid)

                # optional LLM metadata
                analytics_service.update_mistake_patterns(uid, {
                    "patterns": [],
                    "confusion_topics": [],
                    "recommended_tasks": []
                })

            except Exception as e:
                logger.warning(f"[ANALYTICS] WebSocket analytics update failed: {e}")
            
            conv.is_speaking = False
            await self._safe_send(conv.websocket, {"type": "done"})

    async def _stream_answer(self, conv: ConversationState, prompt: str):
        """Stream the answer with interrupt checking."""
        print(f"[ConversationManager] Starting to stream answer for book {conv.book_uuid}")
        
        response_stream = qdrant.generation_model.generate_content(prompt, stream=True)
        
        for chunk in response_stream:
            if conv.should_stop:
                break
            if chunk.text:
                yield chunk.text
            await asyncio.sleep(0.05)

conversation_manager = ConversationManager()
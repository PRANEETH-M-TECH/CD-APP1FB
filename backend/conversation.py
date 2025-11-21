"""
Conversation state management and optimized processing for real-time interactions.
"""
import asyncio
from typing import Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from cachetools import TTLCache
from fastapi import WebSocket
import json

from .qdrant import (
    get_book_metadata,
    hybrid_search,
    reformulate_and_classify_query,
    generate_conversational_answer
)

@dataclass
class ConversationState:
    book_uuid: str
    websocket: WebSocket
    last_query_time: datetime
    cached_context: Dict
    is_speaking: bool = False
    should_stop: bool = False

class ConversationManager:
    def __init__(self):
        self.active_conversations: Dict[str, ConversationState] = {}
        self.context_cache = TTLCache(maxsize=100, ttl=300)  # Cache contexts for 5 minutes
    
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
        """Attempt to send JSON over websocket. Returns False if send fails (client disconnected).

        This centralizes send error handling so we don't try to send after the socket is closed.
        """
        try:
            await websocket.send_json(payload)
            return True
        except Exception as e:
            # Common exceptions are WebSocketDisconnect or send after close; log and return False
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
    
    def get_cached_context(self, book_uuid: str, query: str) -> Optional[dict]:
        cache_key = f"{book_uuid}:{query}"
        return self.context_cache.get(cache_key)
    
    def cache_context(self, book_uuid: str, query: str, context: dict):
        cache_key = f"{book_uuid}:{query}"
        self.context_cache[cache_key] = context

    async def process_query(self, conversation_id: str, query: str):
        if conversation_id not in self.active_conversations:
            return
        
        conv = self.active_conversations[conversation_id]
        print(f"\n{'='*60}")
        print(f"[CONVERSATION] New message from user")
        print(f"[CONVERSATION] Conversation ID: {conversation_id}")
        print(f"[CONVERSATION] User input: {query}")
        print(f"[CONVERSATION] Book UUID: {conv.book_uuid[:16]}...")
        print(f"{'='*60}\n")
        
        conv.should_stop = False
        conv.is_speaking = True
        
        try:
            # Check cache first
            cached_context = self.get_cached_context(conv.book_uuid, query)
            if cached_context:
                search_results = cached_context.get("search_results", [])
                print(f"[CONVERSATION] ✓ Using cached search results\n")
            else:
                print(f"[CONVERSATION] Performing new search...")
                # Get fresh context if not cached
                metadata = get_book_metadata(conv.book_uuid)
                
                # Use lighter query processing for follow-ups
                processed_query_data = reformulate_and_classify_query(
                    query=query,
                    class_name=metadata.get("class_name"),
                    subject=metadata.get("subject")
                )
                
                search_results, _, _ = hybrid_search(
                    book_uuid=conv.book_uuid,
                    query=processed_query_data.get("reformulated_query", query),
                    keywords=processed_query_data.get("keywords", []),
                    conceptual_score=processed_query_data.get("conceptual_score", 0.5)
                )
                
                print(f"[CONVERSATION] ✓ Retrieved {len(search_results)} chunks\n")
                
                # Cache the results
                context_to_cache = {
                    "search_results": search_results,
                    "metadata": metadata
                }
                self.cache_context(conv.book_uuid, query, context_to_cache)
            
            # Stream the answer
            if search_results:
                context = "\n\n---\n\n".join([payload['text'] for score, payload in search_results])
                print(f"[CONVERSATION] Streaming answer to user...\n")
                async for chunk in self._stream_answer(conv, query, context):
                    if conv.should_stop:
                        print(f"[ConversationManager] Conversation {conversation_id} interrupted by user")
                        # try to inform client; if it fails, stop sending
                        ok = await self._safe_send(conv.websocket, {
                            "type": "interrupted",
                            "message": "Response stopped."
                        })
                        if not ok:
                            break
                        break
                    # Log that we're sending a chunk (trimmed preview)
                    try:
                        preview = (chunk[:120] + '...') if len(chunk) > 120 else chunk
                    except Exception:
                        preview = '<non-printable chunk>'
                    print(f"[ConversationManager] Sending chunk to {conversation_id}: {preview}")
                    ok = await self._safe_send(conv.websocket, {
                        "type": "chunk",
                        "content": chunk
                    })
                    if not ok:
                        # Client disconnected, stop processing
                        print(f"[ConversationManager] Stop streaming to {conversation_id} because send failed")
                        break
            else:
                await self._safe_send(conv.websocket, {
                    "type": "error",
                    "message": "No relevant information found."
                })
        
        except Exception as e:
            await conv.websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        finally:
            conv.is_speaking = False
            # Attempt to send done; ignore if client disconnected
            await self._safe_send(conv.websocket, {"type": "done"})

    async def _stream_answer(self, conv: ConversationState, query: str, context: str):
        """Stream the answer with interrupt checking."""
        metadata = get_book_metadata(conv.book_uuid)
        book_details = {
            "class_name": metadata.get("class_name"),
            "subject": metadata.get("subject")
        }
        print(f"[ConversationManager] Starting to stream answer for book {conv.book_uuid}")
        
        for chunk in generate_conversational_answer(query, book_details, context):
            if conv.should_stop:
                break
            # Optionally log chunk size
            try:
                print(f"[ConversationManager] Generated chunk (len={len(chunk)})")
            except Exception:
                pass
            yield chunk
            # Small delay to allow interrupt checking
            await asyncio.sleep(0.1)

conversation_manager = ConversationManager()
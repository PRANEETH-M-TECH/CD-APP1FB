"""
Smart session management with book-scoped context.
Maintains rolling conversation window (last 5 turns) per book.
"""
from datetime import datetime
from typing import Dict, List, Optional
import time


class SmartSessionManager:
    """
    Manages conversation sessions with book-scoped context.
    Each session is tied to a specific book (book_uuid).
    Maintains a rolling window of the last N conversation turns.
    """
    
    def __init__(self):
        self.memory_cache: Dict[str, dict] = {}  # {session_id: session_data}
        self.ttl = 3600  # 1 hour session timeout
    
    def get_or_create_session(self, book_uuid: str, session_id: Optional[str] = None) -> dict:
        """
        Get existing session or create new one (book-scoped).
        
        Args:
            book_uuid: UUID of the current book
            session_id: Optional existing session ID
        
        Returns:
            Session dictionary with conversation window
        """
        # Create new session if no ID provided or session doesn't exist
        if not session_id or session_id not in self.memory_cache:
            session_id = f"{book_uuid}_{int(time.time())}"
            session = {
                "session_id": session_id,
                "book_uuid": book_uuid,
                "conversation_window": [],  # Last N turns
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "max_window_size": 5  # Keep last 5 turns
            }
            self.memory_cache[session_id] = session
            print(f"[SESSION] ✨ Created new session: {session_id}")
            return session
        
        session = self.memory_cache[session_id]
        
        # Validate book match - if book changed, create new session
        if session["book_uuid"] != book_uuid:
            print(f"[SESSION] 📚 Book changed from {session['book_uuid'][:16]}... to {book_uuid[:16]}...")
            print(f"[SESSION] Creating new session for new book")
            return self.get_or_create_session(book_uuid, None)
        
        print(f"[SESSION] ♻️ Reusing existing session: {session_id}")
        return session
    
    def add_turn(self, session_id: str, turn_data: dict):
        """
        Add a new conversation turn and maintain rolling window.
        
        Args:
            session_id: Session ID
            turn_data: Dictionary containing turn information
        """
        if session_id not in self.memory_cache:
            print(f"[SESSION] ⚠️ Session {session_id} not found")
            return
        
        session = self.memory_cache[session_id]
        turn_data["turn"] = len(session["conversation_window"]) + 1
        session["conversation_window"].append(turn_data)
        
        # Rolling window: keep last N turns only
        max_size = session.get("max_window_size", 5)
        if len(session["conversation_window"]) > max_size:
            removed_turn = session["conversation_window"].pop(0)
            print(f"[SESSION] 🗑️ Removed turn {removed_turn['turn']} (window full)")
        
        session["last_updated"] = datetime.now().isoformat()
        print(f"[SESSION] ✅ Added turn {turn_data['turn']} (window size: {len(session['conversation_window'])})")
    
    def get_window(self, session_id: str) -> List[dict]:
        """
        Get conversation window for a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            List of conversation turns
        """
        if session_id not in self.memory_cache:
            return []
        return self.memory_cache[session_id]["conversation_window"]
    
    def get_turn(self, session_id: str, turn_number: int) -> Optional[dict]:
        """
        Get a specific turn from the session.
        
        Args:
            session_id: Session ID
            turn_number: Turn number (1-indexed)
        
        Returns:
            Turn data or None if not found
        """
        window = self.get_window(session_id)
        for turn in window:
            if turn["turn"] == turn_number:
                return turn
        return None
    
    def cleanup_expired_sessions(self):
        """
        Remove expired sessions based on TTL.
        Called periodically to free memory.
        """
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.memory_cache.items():
            last_updated = datetime.fromisoformat(session["last_updated"])
            age = (current_time - last_updated).total_seconds()
            
            if age > self.ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.memory_cache[session_id]
            print(f"[SESSION] 🧹 Cleaned up expired session: {session_id}")
        
        if expired_sessions:
            print(f"[SESSION] Removed {len(expired_sessions)} expired sessions")


# Global session manager instance
session_manager = SmartSessionManager()

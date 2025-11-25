"""
Smart query intent classification for conversational context.
Determines if a query is independent, follow-up, or clarification.
"""
from typing import List, Dict, Optional
import json
import re


def classify_query_intent(
    current_query: str,
    conversation_window: List[dict],
    book_uuid: str,
    is_clicked_followup: bool = False,
    generation_model = None
) -> dict:
    """
    Determines query type and decides retrieval strategy.
    
    Uses three-tier detection:
    1. Pattern matching (fast, obvious cases)
    2. Clicked follow-up flag (frontend tells us)
    3. LLM-based detection (smart, handles edge cases)
    
    Args:
        current_query: User's current question
        conversation_window: List of previous turns
        book_uuid: Current book UUID
        is_clicked_followup: Whether user clicked a suggested follow-up
        generation_model: Gemini model for LLM-based detection
    
    Returns:
        {
            "type": "independent" | "followup" | "clarification",
            "needs_retrieval": bool,
            "reuse_turn": int | null,
            "context_needed": bool,
            "reason": str
        }
    """
    
    # CASE 1: Empty conversation → Independent
    if not conversation_window or len(conversation_window) == 0:
        return {
            "type": "independent",
            "needs_retrieval": True,
            "reuse_turn": None,
            "context_needed": False,
            "reason": "First query in session"
        }
    
    # CASE 2: Clicked follow-up button → Reuse context
    if is_clicked_followup:
        last_turn = conversation_window[-1]["turn"]
        return {
            "type": "followup",
            "needs_retrieval": False,
            "reuse_turn": last_turn,
            "context_needed": True,
            "reason": "User clicked suggested follow-up button"
        }
    
    # CASE 2.5: Check if spoken query matches any suggested follow-ups (FUZZY MATCH)
    # This helps children who ask follow-ups in their own words
    last_turn_data = conversation_window[-1] if conversation_window else None
    if last_turn_data and "follow_ups" in last_turn_data:
        suggested_followups = last_turn_data.get("follow_ups", [])
        matched_followup = _fuzzy_match_followup(current_query, suggested_followups)
        
        if matched_followup:
            last_turn = conversation_window[-1]["turn"]
            return {
                "type": "followup",
                "needs_retrieval": False,
                "reuse_turn": last_turn,
                "context_needed": True,
                "reason": f"Fuzzy matched to suggested follow-up: '{matched_followup}'"
            }
    
    # CASE 3: Pattern matching for obvious follow-ups
    followup_patterns = [
        r"\bexplain (that|it|this)\b",
        r"\belaborate\b",
        r"\bmore about\b",
        r"\btell me more\b",
        r"\bsimpler (terms|words)\b",
        r"\bgive (me )?(an? )?example\b",
        r"\bcan you (explain|clarify|elaborate)\b",
        r"\bwhat about (that|it|this)\b",
        r"\bhow about (that|it|this)\b",
        r"\bclarify (that|it|this)\b",
        r"\bin detail\b",
        r"\bexpand on\b",
        r"\b(what|how) (does|is|are) (that|it|this)\b"
    ]
    
    query_lower = current_query.lower()
    for pattern in followup_patterns:
        if re.search(pattern, query_lower):
            last_turn = conversation_window[-1]["turn"]
            matched_pattern = pattern.replace("\\b", "").replace("(", "").replace(")", "")
            return {
                "type": "followup",
                "needs_retrieval": False,
                "reuse_turn": last_turn,
                "context_needed": True,
                "reason": f"Pattern match: '{matched_pattern}' in query"
            }
    
    # CASE 4: LLM-based smart detection (handles complex cases)
    if generation_model:
        return _llm_based_detection(
            current_query,
            conversation_window,
            generation_model
        )
    else:
        # Fallback: assume independent if no LLM available
        return {
            "type": "independent",
            "needs_retrieval": True,
            "reuse_turn": None,
            "context_needed": False,
            "reason": "No generation model available, defaulting to independent"
        }


def _llm_based_detection(
    current_query: str,
    conversation_window: List[dict],
    generation_model
) -> dict:
    """
    Use LLM to detect if query is follow-up or new topic.
    Handles complex cases like topic switching.
    """
    
    # Get last 2 turns for context
    recent_turns = conversation_window[-2:] if len(conversation_window) >= 2 else conversation_window
    context_summary = ""
    
    for turn in recent_turns:
        context_summary += f"Turn {turn['turn']}: {turn['query']}\n"
    
    prompt = f"""You are analyzing if a student's question is a FOLLOW-UP to previous questions or a NEW TOPIC.

PREVIOUS CONVERSATION:
{context_summary}

CURRENT QUERY: "{current_query}"

RULES:
1. It's a FOLLOW-UP if:
   - References previous answers ("that", "it", "those")
   - Same subject/topic as previous questions
   - Asks for clarification, examples, or elaboration
   
2. It's a NEW TOPIC if:
   - Completely different subject (e.g., photosynthesis → motion)
   - No connection to previous questions
   - Fresh question unrelated to context

Respond ONLY with this JSON format:
{{
  "is_followup": true/false,
  "is_same_topic": true/false,
  "reasoning": "brief explanation (1 sentence)"
}}

Examples:

Previous: "What is photosynthesis?"
Current: "What is Newton's law?"
Response: {{"is_followup": false, "is_same_topic": false, "reasoning": "Different topics - biology vs physics"}}

Previous: "What is photosynthesis?"
Current: "How do plants use chlorophyll?"
Response: {{"is_followup": true, "is_same_topic": true, "reasoning": "Same topic, asking about component of photosynthesis"}}

Previous: "Explain democracy"
Current: "give me an example"
Response: {{"is_followup": true, "is_same_topic": true, "reasoning": "Requesting example of previous topic"}}
"""
    
    try:
        response = generation_model.generate_content(prompt)
        raw = response.text.strip()
        
        # Extract JSON from response
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        # Clean up any markdown or extra text
        raw = raw.strip()
        
        result = json.loads(raw)
        
        # Validate response structure
        if "is_followup" not in result or "is_same_topic" not in result:
            raise ValueError("Invalid LLM response structure")
        
        # Determine intent based on LLM analysis
        if result.get("is_followup") and result.get("is_same_topic"):
            last_turn = conversation_window[-1]["turn"]
            return {
                "type": "followup",
                "needs_retrieval": False,
                "reuse_turn": last_turn,
                "context_needed": True,
                "reason": f"LLM: {result.get('reasoning', 'Follow-up detected')}"
            }
        else:
            return {
                "type": "independent",
                "needs_retrieval": True,
                "reuse_turn": None,
                "context_needed": False,
                "reason": f"LLM: {result.get('reasoning', 'New topic detected')}"
            }
    
    except Exception as e:
        print(f"[INTENT] ⚠️ LLM detection failed: {e}")
        # Fallback: assume independent on error
        return {
            "type": "independent",
            "needs_retrieval": True,
            "reuse_turn": None,
            "context_needed": False,
            "reason": f"LLM check failed ({str(e)}), defaulting to independent"
        }


def extract_voice_command(query: str) -> Optional[str]:
    """
    Extract voice commands from user query.
    
    Commands:
    - "new topic" - Force new search
    - "repeat" - Repeat last answer
    - "stop" - Stop current response
    
    Args:
        query: User's voice input
    
    Returns:
        Command name or None
    """
    query_lower = query.lower().strip()
    
    # New topic command
    if re.search(r"\bnew topic\b", query_lower):
        return "new_topic"
    
    # Repeat command
    if re.search(r"\b(repeat|say (that|it) again)\b", query_lower):
        return "repeat"
    
    # Stop command
    if re.search(r"\b(stop|cancel|nevermind)\b", query_lower):
        return "stop"
    
    return None


def _fuzzy_match_followup(user_query: str, suggested_followups: List[str]) -> Optional[str]:
    """
    Check if user's query (in their own words) matches any suggested follow-up.
    
    This helps children who cannot speak the exact suggested follow-up text.
    
    Example:
        Suggested: "What are the products of photosynthesis?"
        Child says: "what does it make?"
        → Should match!
    
    Args:
        user_query: What the child actually said
        suggested_followups: List of suggested follow-up questions
    
    Returns:
        Matched follow-up string or None
    """
    if not suggested_followups or not user_query:
        return None
    
    user_query_clean = user_query.lower().strip()
    
    # Extract keywords from user query (remove common words)
    stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'can', 'may', 'might', 'must', 'of', 'in', 'on', 'at', 'to',
                 'for', 'with', 'from', 'by', 'about', 'as', 'into', 'through', 'during',
                 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
                 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 'that', 'this', 'these', 'those'}
    
    user_words = [w for w in re.findall(r'\b\w+\b', user_query_clean) if w not in stopwords]
    
    # Check each suggested follow-up for similarity
    best_match = None
    best_score = 0
    
    for suggested in suggested_followups:
        suggested_clean = suggested.lower().strip()
        suggested_words = [w for w in re.findall(r'\b\w+\b', suggested_clean) if w not in stopwords]
        
        # Calculate keyword overlap
        if not user_words or not suggested_words:
            continue
        
        common_words = set(user_words) & set(suggested_words)
        overlap_ratio = len(common_words) / max(len(user_words), len(suggested_words))
        
        # Boost score if question words match (what, how, why, where, when, who)
        question_words = {'what', 'how', 'why', 'where', 'when', 'who', 'which'}
        user_question_words = set(user_query_clean.split()) & question_words
        suggested_question_words = set(suggested_clean.split()) & question_words
        
        question_match = len(user_question_words & suggested_question_words) > 0
        
        # Calculate final score
        score = overlap_ratio
        if question_match:
            score += 0.2  # Boost for matching question type
        
        # Special case: user says very short query like "what does it make?"
        # Check if it's asking about the same core concept
        if len(user_words) <= 4 and overlap_ratio > 0.4:
            score += 0.3  # Boost for short, simple questions
        
        if score > best_score:
            best_score = score
            best_match = suggested
    
    # Threshold: need at least 50% similarity
    if best_score >= 0.5:
        print(f"[FUZZY MATCH] User query '{user_query}' matched to '{best_match}' (score: {best_score:.2f})")
        return best_match
    
    return None

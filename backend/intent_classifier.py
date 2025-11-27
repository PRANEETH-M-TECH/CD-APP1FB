"""
Intelligent conversation action classifier.
Determines the next best action for the system to take based on conversational context.
Enhanced with semantic similarity scoring for intelligent cache reuse.
"""
from typing import List, Dict, Optional
import json


def calculate_query_similarity(
    current_query: str,
    previous_queries: List[str],
    embedder
) -> List[float]:
    """
    Calculate cosine similarity between current and previous queries.
    
    Args:
        current_query: The user's current query
        previous_queries: List of previous queries to compare against
        embedder: Sentence transformer model for encoding
    
    Returns:
        List of similarity scores (0.0 to 1.0) for each previous query
    """
    try:
        from sentence_transformers import util
        
        if not previous_queries:
            return []
        
        current_embedding = embedder.encode(current_query, convert_to_tensor=True)
        scores = []
        
        for prev_query in previous_queries:
            prev_embedding = embedder.encode(prev_query, convert_to_tensor=True)
            similarity = util.cos_sim(current_embedding, prev_embedding)[0][0].item()
            scores.append(similarity)
        
        return scores
    except Exception as e:
        print(f"[SIMILARITY] Error calculating similarity: {e}")
        return []


def determine_next_action(
    current_query: str,
    conversation_window: List[dict],
    generation_model,
    embedder=None,
    is_clicked_followup: bool = False,  # NEW: Flag for pre-generated follow-up clicks
    last_action: str = None  # NEW: Previous action for context awareness
) -> dict:
    """
    Determine the next action using 5-TIER intelligent routing:
    
    TIER 1: Clicked follow-ups (absolute priority - always use cache)
    TIER 2: Empty conversation (must retrieve)
    TIER 3: Meta-conversational queries (answer from history)
    TIER 4: Semantic similarity analysis (smart cache/retrieval)
    TIER 5: LLM fallback for edge cases
    
    Args:
        current_query: User's current query
        conversation_window: List of recent conversation turns
        generation_model: LLM for classification
        embedder: Sentence transformer for similarity
        is_clicked_followup: True if user clicked a pre-generated follow-up
        last_action: Previous action taken (for context)
    
    Returns:
        {
            "action": "USE_CACHED_CONTEXT" | "RETRIEVE_NEW_CONTEXT" | "ANSWER_FROM_HISTORY",
            "reason": str,
            "similarity_score": float,
            "tier": str,
            "new_topic_name": str (optional)
        }
    """
    
    # === TIER 1: ABSOLUTE PRIORITY FOR CLICKED FOLLOW-UPS ===
    if is_clicked_followup:
        # If parent query used cache, follow-up definitely can too
        if last_action == "USE_CACHED_CONTEXT":
            print(f"[TIER 1] ⚡ Clicked follow-up + cached parent → Guaranteed cache reuse")
            return {
                "action": "USE_CACHED_CONTEXT",
                "reason": "Pre-generated follow-up with cached parent context",
                "similarity_score": 1.0,
                "tier": "ABSOLUTE_PRIORITY",
                "confidence": "GUARANTEED"
            }
        
        # Even if parent did retrieval, follow-ups are generated from those chunks
        print(f"[TIER 1] ✓ Clicked follow-up → Strong cache preference")
        return {
            "action": "USE_CACHED_CONTEXT",
            "reason": "Pre-generated follow-ups are contextually guaranteed to be related",
            "similarity_score": 0.95,
            "tier": "STRONG_PREFERENCE",
            "confidence": "HIGH"
        }
    
    # Similarity thresholds for cache decisions
    HIGH_SIMILARITY_THRESHOLD = 0.75  # Very similar → use cache
    MEDIUM_SIMILARITY_THRESHOLD = 0.50  # Somewhat similar → ask LLM
    
    # === TIER 2: EMPTY CONVERSATION ===
    if not conversation_window:
        print(f"[TIER 2] 🆕 First query → Retrieval required")
        return {
            "action": "RETRIEVE_NEW_CONTEXT",
            "new_topic_name": current_query[:50],  # Use query as initial topic name
            "reason": "First query in conversation - no context available",
            "similarity_score": 0.0,
            "tier": "INITIAL_QUERY"
        }

    # === TIER 3: META-CONVERSATIONAL QUERIES ===
    # Queries about the conversation itself, not about the topic
    meta_patterns = [
        "what was", "what did", "earlier", "previous", "before",
        "remind me", "first question", "last", "summarize", "review"
    ]
    query_lower = current_query.lower()
    if any(pattern in query_lower for pattern in meta_patterns):
        print(f"[TIER 3] 💬 Meta-conversational query detected → Answer from history")
        return {
            "action": "ANSWER_FROM_HISTORY",
            "new_topic_name": None,
            "reason": "User asking about previous conversation content",
            "similarity_score": 0.0,
            "tier": "META_QUERY"
        }

    # === TIER 4: SEMANTIC SIMILARITY ANALYSIS ===
    max_similarity = 0.0
    similarity_scores = []
    
    if embedder is not None:
        try:
            # Get recent queries for comparison
            recent_queries = [turn['query'] for turn in conversation_window[-3:]]
            similarity_scores = calculate_query_similarity(current_query, recent_queries, embedder)
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            
            print(f"[TIER 4] 🔍 Semantic similarity analysis:")
            print(f"[TIER 4]   Current query: '{current_query[:50]}...'")
            print(f"[TIER 4]   Comparing with {len(recent_queries)} recent queries")
            for i, (prev_q, score) in enumerate(zip(recent_queries, similarity_scores)):
                print(f"[TIER 4]     {i+1}. '{prev_q[:40]}...' → {score:.3f}")
            print(f"[TIER 4]   Max similarity: {max_similarity:.3f}")
        except Exception as e:
            print(f"[TIER 4] ⚠️ Error during similarity calculation: {e}")
            max_similarity = 0.0
    else:
        print(f"[TIER 4] ⚠️ No embedder provided, skipping similarity check")
    
    # High similarity → Cache reuse
    if max_similarity >= HIGH_SIMILARITY_THRESHOLD:
        print(f"[TIER 4] ⚡ High similarity ({max_similarity:.3f}) → Cache reuse")
        return {
            "action": "USE_CACHED_CONTEXT",
            "new_topic_name": None,
            "reason": f"High semantic similarity ({max_similarity:.2f}) with recent queries - using cached context for speed",
            "similarity_score": max_similarity,
            "tier": "HIGH_SIMILARITY"
        }
    
    # Low similarity + substantial history → Likely new topic
    # But double-check it's not a meta-query first
    if max_similarity < MEDIUM_SIMILARITY_THRESHOLD and len(conversation_window) >= 2:
        if not any(p in query_lower for p in meta_patterns):
            print(f"[TIER 4] 🔍 Low similarity ({max_similarity:.3f}) → New retrieval")
            return {
                "action": "RETRIEVE_NEW_CONTEXT",
                "new_topic_name": current_query[:50],
                "reason": f"Low similarity ({max_similarity:.2f}) suggests topic change",
                "similarity_score": max_similarity,
                "tier": "LOW_SIMILARITY"
            }


    # === TIER 5: LLM FALLBACK FOR EDGE CASES ===
    # Medium similarity (0.50-0.75) or uncertain cases
    print(f"[TIER 5] 🤖 LLM classifier for edge case (similarity: {max_similarity:.3f})")
    
    # Build a summary of the last few turns for the LLM prompt.
    context_summary = ""
    for turn in conversation_window[-3:]: # Use last 3 turns
        answer_preview = turn.get('answer', 'No answer was given.')[:200]
        if len(turn.get('answer', '')) > 200:
            answer_preview += "..."
        context_summary += f"Q: {turn['query']}\nA: {answer_preview}\n\n"

    # Include similarity score in the prompt for better LLM decision-making
    similarity_context = ""
    if max_similarity > 0:
        similarity_context = f"\n## Semantic Similarity Analysis:\nThe current query has a semantic similarity score of {max_similarity:.2f} with recent queries (0.0 = completely different, 1.0 = identical).\n"

    prompt = f"""You are an AI assistant that analyzes a user's query within an ongoing conversation to decide the next best action.

## Conversation History:
{context_summary}
{similarity_context}
## User's New Query:
"{current_query}"

## Available Actions:
1.  `USE_CACHED_CONTEXT`: Choose this if the user's query is a direct follow-up that can be answered using the same information retrieved for the previous question. Examples: "explain that in more detail," "give me an example," "what does that mean?"

2.  `RETRIEVE_NEW_CONTEXT`: Choose this if the user is asking about a completely new topic, or a related but distinctly different topic that requires searching the textbook for new information. Examples: "Okay, now tell me about photosynthesis," "What about the French Revolution?," "How are magnets different from electricity?"

3.  `ANSWER_FROM_HISTORY`: Choose this if the query can be answered directly from the `Conversation History` provided above, without needing the textbook. Examples: "What was the first question I asked?," "Summarize what we just talked about."

## Your Task:
Analyze the user's intent and respond in the following JSON format. Choose ONLY ONE action.

{{
  "analysis": "A brief analysis of the user's intent.",
  "action": "The single best action to take from the list above.",
  "new_topic_name": "If the action is 'RETRIEVE_NEW_CONTEXT', provide a short name for the new topic (e.g., 'Photosynthesis'). Otherwise, null."
}}
"""

    try:
        response = generation_model.generate_content(prompt)

        # Safety check: Ensure the response has content.
        if not response.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            print(f"[ACTION_CLASSIFIER] LLM returned an empty response. Finish Reason: {finish_reason}. Defaulting to new retrieval.")
            return {
                "action": "RETRIEVE_NEW_CONTEXT",
                "new_topic_name": current_query[:50],
                "reason": f"LLM response was empty or blocked (finish reason: {finish_reason}).",
                "similarity_score": max_similarity
            }

        response_text = response.text.strip()
        
        # --- ROBUST JSON EXTRACTION ---
        try:
            # Find the start and end of the JSON object
            start_index = response_text.find('{')
            end_index = response_text.rfind('}') + 1
            
            if start_index == -1 or end_index == 0:
                raise ValueError("No JSON object found in the response.")
            
            # Extract and parse the JSON
            json_text = response_text[start_index:end_index]
            result = json.loads(json_text)
        except (ValueError, json.JSONDecodeError) as json_e:
             # If parsing fails, it's a critical error with the LLM's output.
             print(f"[ACTION_CLASSIFIER] JSON parsing failed: {json_e}")
             print(f"[ACTION_CLASSIFIER] Raw LLM response:\n---\n{response_text}\n---")
             raise ValueError(f"Failed to parse JSON from LLM: {json_e}")

        # Validate the response from the LLM
        if "action" not in result or result["action"] not in ["USE_CACHED_CONTEXT", "RETRIEVE_NEW_CONTEXT", "ANSWER_FROM_HISTORY"]:
             raise ValueError("LLM response missing or has invalid 'action'.")

        print(f"[LLM CLASSIFIER] ✓ Action determined: {result['action']}")
        
        return {
            "action": result["action"],
            "new_topic_name": result.get("new_topic_name"),
            "reason": result.get("analysis", "LLM-based action determination."),
            "similarity_score": max_similarity,
            "tier": "LLM_FALLBACK"
        }

    except Exception as e:
        print(f"[TIER 5] ❌ LLM classification error: {e}")
        # Safe fallback: default to safe retrieval
        return {
            "action": "RETRIEVE_NEW_CONTEXT",
            "new_topic_name": current_query[:50],
            "reason": f"Classification error - defaulting to safe retrieval (error: {str(e)[:100]})",
            "similarity_score": max_similarity,
            "tier": "ERROR_FALLBACK"
        }
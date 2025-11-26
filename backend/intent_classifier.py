"""
Intelligent conversation action classifier.
Determines the next best action for the system to take based on conversational context.
"""
from typing import List, Dict
import json

def determine_next_action(
    current_query: str,
    conversation_window: List[dict],
    generation_model
) -> dict:
    """
    Determines the next best action for the system using an LLM-based router.

    Args:
        current_query: The user's latest query.
        conversation_window: The last few turns of the conversation.

    Returns:
        A dictionary containing the chosen action and any related metadata.
        Example:
        {
            "action": "USE_CACHED_CONTEXT",
            "new_topic_name": None,
            "reason": "The user is asking a direct follow-up question."
        }
    """
    # If there's no history, the only action is to retrieve new context.
    if not conversation_window:
        return {
            "action": "RETRIEVE_NEW_CONTEXT",
            "new_topic_name": current_query, # Use the query as the initial topic name
            "reason": "This is the first query in the conversation."
        }

    # Build a summary of the last few turns for the LLM prompt.
    context_summary = ""
    for turn in conversation_window[-3:]: # Use last 3 turns
        answer_preview = turn.get('answer', 'No answer was given.')[:200]
        if len(turn.get('answer', '')) > 200:
            answer_preview += "..."
        context_summary += f"Q: {turn['query']}\nA: {answer_preview}\n\n"

    prompt = f"""You are an AI assistant that analyzes a user's query within an ongoing conversation to decide the next best action.

## Conversation History:
{context_summary}

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
                "new_topic_name": current_query,
                "reason": f"LLM response was empty or blocked (finish reason: {finish_reason})."
            }

        response_text = response.text.strip()
        
        # Extract JSON from response, handling markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        # Validate the response from the LLM
        if "action" not in result or result["action"] not in ["USE_CACHED_CONTEXT", "RETRIEVE_NEW_CONTEXT", "ANSWER_FROM_HISTORY"]:
             raise ValueError("LLM response missing or has invalid 'action'.")

        return {
            "action": result["action"],
            "new_topic_name": result.get("new_topic_name"),
            "reason": result.get("analysis", "LLM-based action determination.")
        }

    except Exception as e:
        print(f"[ACTION_CLASSIFIER] ⚠️ Action determination failed: {e}. Defaulting to new retrieval.")
        # In case of any failure, the safest fallback is to re-run the retrieval.
        return {
            "action": "RETRIEVE_NEW_CONTEXT",
            "new_topic_name": current_query,
            "reason": f"Classifier failed ({str(e)[:100]}), defaulting to new retrieval."
        }
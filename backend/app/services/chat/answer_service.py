import os
import json
import logging
from typing import List, Dict, Optional

from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.prompts import get_teacher_explanation_prompt, templates

def __getattr__(name: str):
    if name == 'openai_client':
        return qdrant.openai_client
    if name == 'generation_model_name':
        return qdrant.generation_model_name
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

logger = logging.getLogger(__name__)


def reformulate_and_classify_query(query: str, class_name: Optional[str] = None, subject: Optional[str] = None, chapter_list: Optional[List] = None) -> Dict:
    """
    Use the generative model to reformulate the query, extract keywords and
    return a conceptual_score. Returns a dict.
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    raw_query = query
    summary_context = ""

    if class_name and subject:
        summary_dir = "summary"
        json_filename_summary = f"{subject.lower()}{class_name.replace(' ', '')}.json"
        json_filepath_summary = os.path.join(summary_dir, json_filename_summary)

        if os.path.exists(json_filepath_summary):
            try:
                with open(json_filepath_summary, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                # Extract chapter summaries to provide as context
                chapter_summaries = []
                for chapter in summary_data.get("chapters", []):
                    chapter_summaries.append(f"Chapter: {chapter.get('chapter_name')}\nSummary: {chapter.get('summary')}\n---")
                summary_context = "\n".join(chapter_summaries)
            except Exception as e:
                print(f"Error reading summary file {json_filepath_summary}: {e}")
                summary_context = ""

    base_prompt = templates.REFORMULATE_AND_CLASSIFY_QUERY_BASE

    if summary_context:
        base_prompt += templates.REFORMULATE_AND_CLASSIFY_QUERY_SUMMARY.format(
            summary_context=summary_context,
            raw_query=raw_query
        )
    else:
        base_prompt += templates.REFORMULATE_AND_CLASSIFY_QUERY_NO_SUMMARY.format(
            raw_query=raw_query
        )

    if not openai_client:
        # Fallback: simple deterministic extraction if no model available
        result = {
            "reformulated_query": raw_query,
            "keywords": [],
            "conceptual_score": 0.5,
            "classification": "conceptual",
        }
        if summary_context:
            result["classified_chapter"] = "None"
        return result

    try:
        response = openai_client.models.generate_content(
            model=generation_model_name,
            contents=base_prompt
        )
        json_text = response.text.strip()
        json_start = json_text.find("{")
        json_end = json_text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            clean_json = json_text[json_start:json_end]
            parsed_json = json.loads(clean_json)
            conceptual_score = parsed_json.get("conceptual_score", 0.5)
            parsed_json["classification"] = "conceptual" if conceptual_score > 0.5 else "factual"
            return parsed_json
        else:
            result = {
                "reformulated_query": raw_query,
                "keywords": [],
                "conceptual_score": 0.5,
                "classification": "conceptual",
            }
            if summary_context:
                result["classified_chapter"] = "None"
            return result
    except Exception as e:
        print(f"Error in reformulate_and_classify_query: {e}")
        result = {
            "reformulated_query": raw_query,
            "keywords": [],
            "conceptual_score": 0.5,
            "classification": "conceptual",
        }
        if summary_context:
            result["classified_chapter"] = "None"
        return result


def generate_answer(raw_query: str, book_details: Dict, context: str):
    """
    Use the generative model (OpenAI) with a teacher-system prompt to answer the query.
    This function is a generator that yields chunks of the response (both for display and TTS).
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized.")

    system_prompt = templates.GENERATE_ANSWER_SYSTEM
    user_prompt = templates.GENERATE_ANSWER_USER.format(
        class_name=book_details.get('class_name', 'N/A'),
        subject=book_details.get('subject', 'N/A'),
        raw_query=raw_query,
        context=context
    )

    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    response = openai_client.models.generate_content_stream(
        model=generation_model_name,
        contents=combined_prompt
    )
    for chunk in response:
        yield chunk.text


def generate_conversational_answer(raw_query: str, book_details: Dict, context: str):
    """
    Use the generative model (OpenAI) with a conversational system prompt to answer the query.
    This is designed for the real-time conversational mode.
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized.")

    system_prompt = templates.GENERATE_CONVERSATIONAL_ANSWER_SYSTEM
    user_prompt = templates.GENERATE_CONVERSATIONAL_ANSWER_USER.format(
        class_name=book_details.get('class_name', 'N/A'),
        subject=book_details.get('subject', 'N/A'),
        raw_query=raw_query,
        context=context
    )

    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    response = openai_client.models.generate_content_stream(
        model=generation_model_name,
        contents=combined_prompt
    )
    for chunk in response:
        yield chunk.text


def generate_teacher_explanation(class_name: str, subject: str, chapter_name: str, summary_text: str) -> str:
    """
    Uses the generative model to create a teacher-like explanation from a chapter summary,
    tailored specifically for Indian students of that class level.
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized.")

    combined_prompt = get_teacher_explanation_prompt(
        class_name=class_name,
        subject=subject,
        chapter_name=chapter_name,
        summary_text=summary_text
    )
    
    response = openai_client.models.generate_content(
        model=generation_model_name,
        contents=combined_prompt
    )
    return response.text


def generate_chapter_summary(class_name: str, subject_name: str, chapter_name: str, chapter_chunks: List[str]) -> str:
    """
    Generates a summary for a single chapter using the generative model.
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized.")

    # Combine chunks into a single text
    full_chapter_text = "\n\n".join(chapter_chunks)

    # Construct the prompt
    prompt = templates.GENERATE_CHAPTER_SUMMARY_PROMPT.format(
        class_name=class_name,
        subject_name=subject_name,
        chapter_name=chapter_name,
        full_chapter_text=full_chapter_text
    )

    try:
        response = openai_client.models.generate_content(
            model=generation_model_name,
            contents=prompt
        )
        # Extract the JSON part from the response
        json_text = response.text.strip()
        json_start = json_text.find("{")
        json_end = json_text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            clean_json = json_text[json_start:json_end]
            parsed_json = json.loads(clean_json)
            return parsed_json.get("summary", "")
        else:
            return "Could not generate summary."
    except Exception as e:
        print(f"Error generating summary for chapter {chapter_name}: {e}")
        return "Could not generate summary."


def generate_chapters_from_json(pdf_json: List[Dict]) -> str:
    """
    Builds a prompt string containing the JSON page list for the LLM to extract chapters.
    Returns the prompt as a plain string.
    """
    json_text = json.dumps(pdf_json, ensure_ascii=False)
    prompt = templates.GENERATE_CHAPTERS_FROM_JSON_PROMPT.format(
        json_text=json_text
    )
    return prompt


def generate_chapters_from_text(json_path: str) -> str:
    """
    Read the page JSON file (json_path), construct prompt and ask the generative model to extract chapters.
    Returns a JSON-string representation of the parsed LLM output or a safe default.
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    if not openai_client:
        return json.dumps({"pdf_offset": 0, "chapters": []})

    with open(json_path, "r", encoding="utf-8") as f:
        pdf_pages_data = json.load(f)

    prompt = generate_chapters_from_json(pdf_pages_data)

    try:
        response = openai_client.models.generate_content(
            model=generation_model_name,
            contents=prompt
        )
        text = response.text.strip()
        
        # DEBUG: Log the raw LLM response
        print(f"[CHAPTER EXTRACTION] LLM Raw Response (first 100 chars):")
        print(repr(text[:100]))
        print(f"[CHAPTER EXTRACTION] Full response length: {len(text)} characters")
        
        # Strip markdown code fences if present (more robust)
        if text.startswith("```json"):
            first_newline = text.find('\n')
            if first_newline != -1:
                text = text[first_newline+1:]
        elif text.startswith("```"):
            first_newline = text.find('\n')
            if first_newline != -1:
                text = text[first_newline+1:]
        
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        
        text = text.strip()
        
        print(f"[CHAPTER EXTRACTION] After fence stripping, length: {len(text)} characters")
        
        try:
            data = json.loads(text)
            num_chapters = len(data.get('chapters', []))
            print(f"[CHAPTER EXTRACTION] âœ… Successfully parsed JSON with {num_chapters} chapters")
            return json.dumps(data)
        except json.JSONDecodeError as e:
            print(f"[CHAPTER EXTRACTION] âŒ JSON decode failed: {e}")
            return json.dumps({"pdf_offset": 0, "chapters": []})

    except Exception as e:
        print(f"[CHAPTER EXTRACTION] âŒ Exception during LLM call: {e}")
        return json.dumps({"pdf_offset": 0, "chapters": []})


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
    from backend.app.core.firebase.firebase_init import db
    doc_ref = db.collection("summaries").document(key)
    doc = doc_ref.get()

    if not doc.exists:
        raise Exception(f"Summary document not found: summaries/{key}")

    data = doc.to_dict()
    SUMMARY_CACHE[key] = data  # cache it

    print(f"[CACHE] Loaded summary -> summaries/{key}")

    return data


def extract_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != -1 and end > start:
        return text[start:end]
    return None


def reformulate_with_llm(raw_query: str, class_name: str, subject: str, chapters):
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    # Extract only chapter names
    chapter_names = [chapter.get("chapter_name") for chapter in chapters if chapter.get("chapter_name")]
    chapter_names_str = json.dumps(chapter_names, ensure_ascii=False, indent=2)

    prompt = templates.REFORMULATE_WITH_LLM_PROMPT.format(
        class_name=class_name,
        subject=subject,
        raw_query=raw_query,
        chapter_names=chapter_names_str
    )

    # LLM Call
    try:
        response = openai_client.models.generate_content(
            model=generation_model_name,
            contents=prompt
        )
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


def context_aware_reformulate(query: str, conversation_window: List[dict]) -> dict:
    """
    Reformulate query using previous conversation context.
    Expands vague references like "that", "it", "more" using previous Q&A.
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    if not conversation_window:
        return {
            "reformulated_query": query,
            "keywords": []
        }
    
    recent_turns = conversation_window[-2:] if len(conversation_window) >= 2 else conversation_window
    context_summary = ""
    
    for turn in recent_turns:
        answer_preview = turn.get('answer', '')[:200]
        if len(turn.get('answer', '')) > 200:
            answer_preview += "..."
        context_summary += f"Q: {turn['query']}\nA: {answer_preview}\n\n"
    
    prompt = templates.CONTEXT_AWARE_REFORMULATE_PROMPT.format(
        context_summary=context_summary,
        query=query
    )
    
    try:
        response = openai_client.models.generate_content(
            model=generation_model_name,
            contents=prompt
        )
        raw = response.text.strip()
        
        json_text = extract_json_block(raw)
        if not json_text:
            json_text = raw
        
        result = json.loads(json_text)
        
        if "reformulated_query" not in result:
            raise ValueError("Missing reformulated_query in response")
        
        print(f"[REFORM] Context-aware reformulation successful")
        print(f"[REFORM] Original: {query}")
        print(f"[REFORM] Reformulated: {result['reformulated_query']}")
        
        return result
    
    except Exception as e:
        print(f"[REFORM] âš ï¸ Context-aware reformulation failed: {e}")
        return {
            "reformulated_query": query,
            "keywords": []
        }


def generate_smart_followups(query: str, answer: str, top_chunks: List) -> List[str]:
    """
    Generate answer-specific follow-up questions tailored for Indian students.
    Questions are age-appropriate, in simple English, and contextually relevant.
    """
    openai_client = qdrant.openai_client
    generation_model_name = qdrant.generation_model_name
    try:
        chapter_names = []
        class_level = None
        subject = None
        
        # print(f"\n{'='*80}")
        # print(f"[FOLLOWUPS] GENERATING FOLLOW-UPS USING {len(top_chunks)} CHUNKS")
        for idx, item in enumerate(top_chunks, 1):
            if isinstance(item, tuple) and len(item) >= 2:
                payload = item[1]
                chapter_name = payload.get("chapter_name", "Unknown")
                
                # Suppressed verbose logging to reduce terminal spam
                # chunk_text = payload.get("text", "")
                # print(f"[FOLLOWUPS]   --- Chunk {idx} (Chapter: {chapter_name}) ---")
                # clean_text = chunk_text.replace('\n', ' ').strip()
                # if len(clean_text) > 300:
                #     print(f"[FOLLOWUPS]   {clean_text[:300]}...\n")
                # else:
                #     print(f"[FOLLOWUPS]   {clean_text}\n")
                
                if chapter_name not in chapter_names and chapter_name != "Unknown":
                    chapter_names.append(chapter_name)
                
                if not class_level:
                    class_level = payload.get("class_name", None)
                if not subject:
                    subject = payload.get("subject", None)
        # print(f"{'='*80}\n")
        
        if class_level:
            try:
                class_num = int(str(class_level).replace("class", "").replace("Class", "").strip())
            except:
                class_num = 8
        else:
            class_num = 8

        if class_num <= 5:
            language_level = "very simple words, short sentences (like talking to a 10-year-old)"
            complexity = "basic concepts only, use everyday examples"
        elif class_num <= 8:
            language_level = "simple, clear English that a 13-year-old understands easily"
            complexity = "moderate depth, relatable examples from daily life"
        else:
            language_level = "clear, straightforward English (not complicated academic words)"
            complexity = "detailed but still clear, real-world applications"
        
        answer_preview = answer[:500]
        if len(answer) > 500:
            answer_preview += "..."
        
        # Use class_num/class_num+2 for formatting
        class_num_plus_2 = class_num + 2
        prompt = templates.GENERATE_SMART_FOLLOWUPS_PROMPT.format(
            class_level=class_level or 'middle school',
            class_num=class_num,
            class_num_plus_2=class_num_plus_2,
            subject=subject or 'the topic',
            query=query,
            answer_preview=answer_preview,
            chapter_names=chapter_names if chapter_names else ['General'],
            language_level=language_level,
            complexity=complexity
        )
        
        response = openai_client.models.generate_content(
            model=generation_model_name,
            contents=prompt
        )
        
        if not response.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            print(f"[FOLLOWUPS] Warning: LLM returned an empty response. Finish Reason: {finish_reason}.")
            raise ValueError(f"Empty response from LLM (finish reason: {finish_reason})")

        raw = response.text.strip()
        
        json_text = extract_json_block(raw)
        if not json_text:
            json_text = raw
        
        result = json.loads(json_text)
        followups = result.get("followups", [])
        
        if not followups or not isinstance(followups, list):
            raise ValueError("Invalid followups format")
        
        followups = followups[:3]
        
        print(f"[FOLLOWUPS] Generated {len(followups)} age-appropriate follow-ups for Class {class_level}")
        for i, f in enumerate(followups, 1):
            print(f"[FOLLOWUPS]   {i}. {f}")
        
        return followups
    
    except Exception as e:
        print(f"[FOLLOWUPS] Warning: Generation failed: {e}")
        return [
            f"Can you explain more about this?",
            f"What is an example of this?",
            f"Why is this important?"
        ]


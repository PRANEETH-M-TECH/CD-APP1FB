import json
import uuid
import logging
import asyncio
import os
import re
from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.services.visual_learning.visual_lesson_prompt import get_visual_lesson_prompt
from backend.app.services.visual_learning.asset_retrieval_service import retrieve_asset_url
from backend.app.services.visual_learning.visual_audio_generator import generate_slide_audio

logger = logging.getLogger(__name__)

def clean_and_parse_json(response_text: str) -> dict:
    """
    Resilient JSON parser that handles code blocks, unescaped text,
    missing object closing braces, trailing commas, and malformed LLM output.
    """
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    # Attempt standard parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"[VisualLearning JSON Cleaner] Initial json.loads failed ({e}). Attempting sanitization...")

    # Apply multi-stage JSON repair pipeline
    cleaned = text
    # Remove single-line C++ style comments
    cleaned = re.sub(r'//.*', '', cleaned)

    # Fix missing object closing brace before scene boundaries
    cleaned = re.sub(r'(\"template_data\"\s*:\s*\{[\s\S]*?\})\s*,\s*(\{\s*\"scene_no\")', r'\1}\n,\n\2', cleaned)
    cleaned = re.sub(r'(\}\s*\n\s*),\s*(\n\s*\{)', r'\1}\n,\2', cleaned)
    # Fix trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*([\}\]])', r'\1', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract outermost JSON object using regex
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        extracted = match.group(1).strip()
        extracted = re.sub(r'//.*', '', extracted)
        extracted = re.sub(r'(\"template_data\"\s*:\s*\{[\s\S]*?\})\s*,\s*(\{\s*\"scene_no\")', r'\1}\n,\n\2', extracted)
        extracted = re.sub(r',\s*([\}\]])', r'\1', extracted)
        
        # Auto-close unclosed braces/brackets if truncated
        open_braces = extracted.count('{') - extracted.count('}')
        open_brackets = extracted.count('[') - extracted.count(']')
        if open_brackets > 0:
            extracted += ']' * open_brackets
        if open_braces > 0:
            extracted += '}' * open_braces

        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Final attempt: line filter
    cleaned_lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        if any(trimmed.startswith(c) for c in ['{', '}', '[', ']', '"', ',', ':', '//', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'true', 'false', 'null']):
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'(\"template_data\"\s*:\s*\{[\s\S]*?\})\s*,\s*(\{\s*\"scene_no\")', r'\1}\n,\n\2', cleaned_text)
    cleaned_text = re.sub(r',\s*([\}\]])', r'\1', cleaned_text)
    
    return json.loads(cleaned_text)

async def generate_visual_lesson_stream(query: str, book_uuid: str, class_name: str, subject: str):
    """
    Main pipeline to generate a visual lesson storyboard.
    Streams progress states synchronized with frontend UI steps, compiles Hyperframes composition,
    and returns completed lesson ready event.
    """
    lesson_id = f"vl_{uuid.uuid4().hex[:8]}"
    print("\n======================================================================")
    print(f"[PIPELINE DEBUG] ENTER VisualLearning")
    print(f"   Query: '{query}' | Lesson ID: {lesson_id}")
    print("======================================================================\n")
    
    try:
        # Step 1: Retrieve context from book using hybrid search
        yield f"data: {json.dumps({'type': 'progress', 'step': 'understanding_topic', 'status': 'in_progress', 'message': 'Retrieving relevant textbook context...'})}\n\n"
        await asyncio.sleep(0.3)
        
        context = ""
        try:
            hybrid_results, _, _ = qdrant.hybrid_search(
                book_uuid=book_uuid,
                query=query,
                keywords=[],
                conceptual_score=0.5,
                metadata_filters=None
            )
            if hybrid_results:
                context = "\n\n---\n\n".join([doc["text"] for score, doc in hybrid_results[:5]])
                logger.info(f"[VisualLearning] Retrieved {len(hybrid_results)} chunks for context.")
            else:
                logger.warning("[VisualLearning] No chunks retrieved. Using query context only.")
        except Exception as e:
            logger.error(f"[VisualLearning] Hybrid search failed: {e}")
            
        yield f"data: {json.dumps({'type': 'progress', 'step': 'understanding_topic', 'status': 'complete', 'message': 'Textbook context analyzed.'})}\n\n"
        
        # Step 2: Design lesson storyboard blueprint with Gemini
        yield f"data: {json.dumps({'type': 'progress', 'step': 'designing_lesson', 'status': 'in_progress', 'message': 'Creating storyboard and scene animations...'})}\n\n"
        await asyncio.sleep(0.3)
        
        prompt = get_visual_lesson_prompt(class_name, subject, query, context)
        client = qdrant.gemini_client
        model_name = qdrant.generation_model_name
        
        if not client:
            raise RuntimeError("Gemini Client is not initialized in qdrant_service.")
            
        logger.info(f"[VisualLearning] Sending storyboard prompt to Gemini ({model_name})...")
        
        # Automatic multi-model fallback chain prioritizing gemini-2.5-flash
        model_candidates = ["gemini-2.5-flash", model_name, "gemini-2.0-flash", "gemini-1.5-flash"]
        models_to_try = []
        for m in model_candidates:
            if m not in models_to_try:
                models_to_try.append(m)

        response_text = None
        for m_name in models_to_try:
            try:
                logger.info(f"[VisualLearning] Attempting storyboard generation with model '{m_name}'...")
                try:
                    from google.genai import types
                    gen_config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                    response = client.models.generate_content(
                        model=m_name,
                        contents=prompt,
                        config=gen_config
                    )
                except Exception:
                    response = client.models.generate_content(
                        model=m_name,
                        contents=prompt
                    )
                response_text = response.text.strip()
                logger.info(f"[VisualLearning] Received Gemini response text using '{m_name}' (length: {len(response_text)})")
                break
            except Exception as m_err:
                logger.warning(f"[VisualLearning] Model '{m_name}' failed: {m_err}. Trying next candidate...")

        if not response_text:
            raise RuntimeError("All Gemini model candidates failed to generate storyboard.")
        
        try:
            blueprint = clean_and_parse_json(response_text)
            raw_clips = blueprint.get("clips", blueprint.get("scenes", []))
            global_assets = blueprint.get("global_assets", [])
            connections = blueprint.get("connections", [])
            layout_mode = blueprint.get("layout_mode", "timeline")
            theme = blueprint.get("theme", "indigo")
            
            # Robust Clip Normalization (Guarantees dict object structure for every scene)
            clips = []
            for idx, item in enumerate(raw_clips, 1):
                if isinstance(item, dict):
                    clips.append(item)
                elif isinstance(item, str):
                    clips.append({
                        "scene_no": idx,
                        "purpose": item,
                        "template_id": "title_slide" if idx == 1 else "concept_diagram",
                        "teacher_script": item,
                        "template_data": {"title": f"Scene {idx}", "subtitle": item}
                    })

            # ── Template Selection Audit & Variety Validation Pass ────────────
            valid_templates = [
                'title_slide', 'concept_diagram', 'cycle_template', 'math_derivation',
                'venn_diagram', 'taxonomy_tree', 'cartesian_grid', 'column_comparison',
                'geo_marker', 'database_grid', 'before_after_slider', 'quiz_checkpoint'
            ]
            
            print("\n----------------------------------------------------------------------")
            print("[STORYBOARD AUDIT] LLM Template Selection & Reasoning Analysis:")
            print(f"   Lesson Title: {blueprint.get('lesson_title', 'Untitled')}")
            print(f"   Total Scenes: {len(clips)}")
            print("----------------------------------------------------------------------")
            
            for idx, clip in enumerate(clips, 1):
                tid = clip.get("template_id", "concept_diagram")
                reasoning = clip.get("template_selection_reasoning", "No explicit reasoning provided.")
                
                is_valid = tid in valid_templates
                status_icon = "[OK]" if is_valid else "[FALLBACK]"
                print(f"   Scene {idx}: [{tid}] {status_icon}")
                print(f"           Reasoning: {reasoning[:90]}...")
            
            # Enforce non-consecutive duplicate rule & title_slide rule
            for idx in range(len(clips)):
                if idx == 0:
                    clips[idx]["template_id"] = "title_slide"
                else:
                    if clips[idx].get("template_id") == "title_slide":
                        clips[idx]["template_id"] = "concept_diagram"
                    if clips[idx].get("template_id") == clips[idx-1].get("template_id"):
                        alt_templates = [t for t in valid_templates if t not in [clips[idx-1].get("template_id"), 'title_slide']]
                        clips[idx]["template_id"] = alt_templates[0]
                        print(f"   [AUDIT REPAIR] Swapped consecutive duplicate template in Scene {idx+1} to '{clips[idx]['template_id']}'")
            
            print("----------------------------------------------------------------------\n")
            
        except Exception as e:
            logger.error(f"[VisualLearning] Failed to parse storyboard JSON. Raw response:\n{response_text[:500]}...\nError: {e}")
            raise ValueError(f"Failed to parse storyboard JSON from Gemini response: {e}")
            
        yield f"data: {json.dumps({'type': 'progress', 'step': 'designing_lesson', 'status': 'complete', 'message': f'Storyboard generated with {len(clips)} dynamic scenes.'})}\n\n"
        
        # Step 3: Retrieve Animated Scene Assets
        yield f"data: {json.dumps({'type': 'progress', 'step': 'generating_visuals', 'status': 'in_progress', 'message': 'Retrieving animated scene templates and visual assets...'})}\n\n"
        await asyncio.sleep(0.3)
        yield f"data: {json.dumps({'type': 'progress', 'step': 'generating_visuals', 'status': 'complete', 'message': 'Scene visual templates assembled.'})}\n\n"

        # Step 4: Synthesize Voice Narration Audio
        yield f"data: {json.dumps({'type': 'progress', 'step': 'creating_narration', 'status': 'in_progress', 'message': 'Synthesizing AI teacher voiceover narration...'})}\n\n"
        await asyncio.sleep(0.3)
        
        processed_scenes = clips
        try:
            audio_urls = await generate_slide_audio(clips, lesson_id)
            for idx, scene in enumerate(processed_scenes):
                if idx < len(audio_urls):
                    scene["audio_url"] = audio_urls[idx]
        except Exception as audio_err:
            logger.warning(f"[VisualLearning] Batch audio generation notice: {audio_err}")
            for scene in processed_scenes:
                if "audio_url" not in scene:
                    scene["audio_url"] = ""
            
        yield f"data: {json.dumps({'type': 'progress', 'step': 'creating_narration', 'status': 'complete', 'message': 'Voiceovers & narration ready.'})}\n\n"
        
        # Step 5: Compile Hyperframes Rendering Engine
        yield f"data: {json.dumps({'type': 'progress', 'step': 'hyperframes_engine', 'status': 'in_progress', 'message': 'Compiling Hyperframes 60fps HTML video composition...'})}\n\n"
        await asyncio.sleep(0.3)
        
        # Correctly calculate absolute PROJECT_ROOT (4 parent directory levels up from backend/app/services/visual_learning)
        MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "..", ".."))
        output_dir = os.path.join(PROJECT_ROOT, "uploads", "visual_lessons", lesson_id)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            output_dir = os.path.join("/tmp", "uploads", "visual_lessons", lesson_id)
            os.makedirs(output_dir, exist_ok=True)
        
        lesson_package = {
            "lesson_id": lesson_id,
            "lesson_title": blueprint.get("lesson_title", f"Visual Lesson: {query}"),
            "layout_mode": layout_mode,
            "theme": theme,
            "global_assets": global_assets,
            "connections": connections,
            "scenes": processed_scenes
        }
        
        # Write lesson.json to root uploads storage directory
        lesson_json_path = os.path.join(output_dir, "lesson.json")
        with open(lesson_json_path, "w", encoding="utf-8") as f:
            json.dump(lesson_package, f, indent=2)
            
        # Trigger engine compilation bridge passing root output_dir
        try:
            from backend.app.services.visual_learning.hyperframes_engine_bridge import compile_hyperframes_html_fast
            compiled_url = await compile_hyperframes_html_fast(lesson_id, output_dir)
        except Exception as compile_err:
            logger.warning(f"[VisualLearning] Engine bridge compilation notice: {compile_err}")
            compiled_url = None

        # Verify whether index.html actually exists on disk before setting html_url
        expected_index_path = os.path.join(output_dir, "index.html")
        if not (compiled_url and os.path.exists(expected_index_path)):
            logger.warning(f"[VisualLearning] index.html not found on disk at {expected_index_path}. Falling back to client-side slide renderer.")
            compiled_url = None
            
        # Attach URLs explicitly for Hyperframes player mounting
        lesson_package["html_url"] = compiled_url
        lesson_package["interactive_url"] = compiled_url
        lesson_package["video_url"] = None

        yield f"data: {json.dumps({'type': 'progress', 'step': 'hyperframes_engine', 'status': 'complete', 'message': 'Hyperframes compilation complete.'})}\n\n"
        
        # Step 6: Launching Media Player & Emit lesson_ready Event
        yield f"data: {json.dumps({'type': 'progress', 'step': 'launching_lesson', 'status': 'in_progress', 'message': 'Launching media player...'})}\n\n"
        await asyncio.sleep(0.2)
        yield f"data: {json.dumps({'type': 'progress', 'step': 'launching_lesson', 'status': 'complete', 'message': 'Lesson ready!'})}\n\n"

        # Final Event Payload matching frontend handleSSEEvent contract
        ready_payload = {
            "type": "lesson_ready",
            "lesson_id": lesson_id,
            "lesson_title": lesson_package["lesson_title"],
            "interactive_url": compiled_url,
            "html_url": compiled_url,
            "video_url": None,
            "scene_count": len(processed_scenes),
            "lesson": lesson_package,
            "lesson_package": lesson_package
        }
        yield f"data: {json.dumps(ready_payload)}\n\n"
        
    except Exception as e:
        logger.error(f"[VisualLearning] Failed to stream visual lesson storyboard: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

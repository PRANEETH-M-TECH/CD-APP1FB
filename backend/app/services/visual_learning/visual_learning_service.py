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
    and hallucinated prose paragraphs injected by the LLM.
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

    # Extract outermost JSON object using regex
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        extracted = match.group(1).strip()
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Filter out hallucinated non-JSON prose lines (e.g. random textbook paragraphs)
    cleaned_lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        # Check if line looks like valid JSON structure
        if any(trimmed.startswith(c) for c in ['{', '}', '[', ']', '"', ',', ':', '//', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'true', 'false', 'null']):
            cleaned_lines.append(line)
        else:
            logger.warning(f"[VisualLearning JSON Cleaner] Stripping non-JSON line: '{trimmed[:60]}...'")

    cleaned_text = "\n".join(cleaned_lines)
    return json.loads(cleaned_text)

async def generate_visual_lesson_stream(query: str, book_uuid: str, class_name: str, subject: str):
    """
    Main pipeline to generate a visual lesson storyboard.
    Streams progress states and returns the completed lesson JSON package.
    """
    lesson_id = f"vl_{uuid.uuid4().hex[:8]}"
    print("\n======================================================================")
    print("🚀 [PIPELINE DEBUG] ENTER VisualLearning")
    print(f"   Query: '{query}' | Lesson ID: {lesson_id}")
    print("======================================================================\n")
    
    try:
        # Step 1: Retrieve context from book using hybrid search
        yield f"data: {json.dumps({'type': 'progress', 'step': 'understanding_topic', 'status': 'in_progress', 'message': 'Retrieving relevant textbook context...'})}\n\n"
        await asyncio.sleep(0.4)
        
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
        await asyncio.sleep(0.4)
        
        prompt = get_visual_lesson_prompt(class_name, subject, query, context)
        client = qdrant.gemini_client
        model_name = qdrant.generation_model_name
        
        if not client:
            raise RuntimeError("Gemini Client is not initialized in qdrant_service.")
            
        logger.info(f"[VisualLearning] Sending storyboard prompt to Gemini ({model_name})...")
        try:
            # Enforce strict JSON output mode in API config to prevent LLM hallucinations
            try:
                from google.genai import types
                gen_config = types.GenerateContentConfig(response_mime_type="application/json")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=gen_config
                )
            except Exception as cfg_err:
                logger.warning(f"[VisualLearning] Custom config failed ({cfg_err}), falling back to standard generate_content.")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            response_text = response.text.strip()
            logger.info(f"[VisualLearning] Received Gemini response text (length: {len(response_text)})")
        except Exception as e:
            logger.error(f"[VisualLearning] Gemini API storyboard generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API storyboard generation failed: {e}")
        
        try:
            blueprint = clean_and_parse_json(response_text)
            clips = blueprint.get("clips", blueprint.get("scenes", []))
            global_assets = blueprint.get("global_assets", [])
            connections = blueprint.get("connections", [])
            layout_mode = blueprint.get("layout_mode", "timeline")
            theme = blueprint.get("theme", "indigo")
            
            # ── Template Selection Audit & Variety Validation Pass ────────────
            valid_templates = [
                'title_slide', 'concept_diagram', 'cycle_template', 'math_derivation',
                'venn_diagram', 'taxonomy_tree', 'cartesian_grid', 'column_comparison',
                'geo_marker', 'database_grid', 'before_after_slider', 'quiz_checkpoint'
            ]
            
            print("\n----------------------------------------------------------------------")
            print("📋 [STORYBOARD AUDIT] LLM Template Selection & Reasoning Analysis:")
            prev_template = None
            concept_diagram_count = 0
            
            for idx, clip in enumerate(clips):
                scene_num = clip.get("scene_no", idx + 1)
                orig_template = clip.get("template_id", "general_scene")
                reasoning = clip.get("template_selection_reasoning", "No LLM reasoning provided.")
                
                # Enforce valid template_id
                if orig_template not in valid_templates:
                    orig_template = "concept_diagram" if scene_num > 1 else "title_slide"
                
                # Rule 1: First scene MUST be title_slide; last scene MUST be quiz_checkpoint (if >2 scenes)
                if idx == 0:
                    orig_template = "title_slide"
                elif idx == len(clips) - 1 and len(clips) >= 3 and orig_template == "concept_diagram":
                    orig_template = "quiz_checkpoint"
                
                # Rule 2: Prevent consecutive duplicate templates
                if orig_template == prev_template and orig_template in ('concept_diagram', 'column_comparison', 'cycle_template'):
                    alternatives = ['column_comparison', 'cycle_template', 'horizontal_timeline', 'venn_diagram']
                    for alt in alternatives:
                        if alt != prev_template:
                            logger.info(f"[Template Validator] Re-mapped Scene {scene_num} from '{orig_template}' to '{alt}' to enforce template variety.")
                            orig_template = alt
                            break
                
                # Rule 3: Cap concept_diagram usage
                if orig_template == 'concept_diagram':
                    concept_diagram_count += 1
                    if concept_diagram_count > 1 and len(clips) >= 4:
                        orig_template = 'column_comparison' if (scene_num % 2 == 0) else 'horizontal_timeline'
                        logger.info(f"[Template Validator] Capped concept_diagram over-use: Re-mapped Scene {scene_num} to '{orig_template}'.")
                
                clip["template_id"] = orig_template
                prev_template = orig_template
                
                # Auto-synthesize dynamic camera parameters if missing from LLM
                if "camera" not in clip or not isinstance(clip["camera"], dict):
                    # Synthesize cinematic camera motion per scene
                    zoom_val = 1.0 + (0.05 * (scene_num % 3))
                    pan_x_val = -15 if scene_num % 2 == 0 else 15
                    clip["camera"] = {
                        "zoom": round(zoom_val, 2),
                        "pan_x": pan_x_val,
                        "pan_y": 0,
                        "target_node": f"scene_node_{scene_num}"
                    }
                
                print(f"   Scene {scene_num}: [{orig_template}]")
                print(f"     └─ LLM Reasoning: \"{reasoning}\"")
                print(f"     └─ Camera Framing: Zoom={clip['camera'].get('zoom', 1.0)}x, PanX={clip['camera'].get('pan_x', 0)}")
            
            print("----------------------------------------------------------------------\n")

            logger.info(f"[VisualLearning] Parsed and validated storyboard successfully with {len(clips)} clips.")
            print("\n======================================================================")
            print("🚀 [PIPELINE DEBUG] ENTER Storyboard")
            print(f"   Generated storyboard ID: {lesson_id} with {len(clips)} scenes")
            print("======================================================================\n")
        except Exception as e:
            logger.error(f"[VisualLearning] Failed to parse storyboard JSON. Raw response:\n{response_text}\nError: {e}", exc_info=True)
            raise ValueError(f"Failed to parse storyboard JSON from Gemini response: {e}")
        
        if not clips:
            raise ValueError("Gemini storyboard does not contain any clips.")
            
        yield f"data: {json.dumps({'type': 'progress', 'step': 'designing_lesson', 'status': 'complete', 'message': f'Designed storyboard with {len(clips)} clips.'})}\n\n"
        
        # Step 3: Retrieve visual assets for scenes (both global and local clip assets)
        yield f"data: {json.dumps({'type': 'progress', 'step': 'generating_visuals', 'status': 'in_progress', 'message': 'Retrieving educational visual assets...'})}\n\n"
        await asyncio.sleep(0.4)
        
        # Resolve global assets
        for asset in global_assets:
            asset_type = asset.get("type", "image")
            if asset_type in ("image", "icon"):
                query_str = asset.get("search_query", "")
                logger.info(f"[VisualLearning] Searching global asset: '{query_str}' (type: {asset_type})")
                asset_url = await retrieve_asset_url(query_str, asset_type=asset_type, theme=theme)
                asset["asset_url"] = asset_url
            else:
                asset["asset_url"] = ""

        # Resolve local clip image and icon assets
        for clip in clips:
            clip_no = clip.get("clip_no", clip.get("scene_no", 1))
            local_assets = clip.get("local_assets", clip.get("assets", []))
            for asset in local_assets:
                asset_type = asset.get("type", "image")
                query_str = asset.get("search_query", "")
                
                if asset_type in ("image", "icon"):
                    logger.info(f"[VisualLearning] Searching asset for clip {clip_no}: '{query_str}' (type: {asset_type})")
                    asset_url = await retrieve_asset_url(query_str, asset_type=asset_type, theme=theme)
                    asset["asset_url"] = asset_url
                else:
                    asset["asset_url"] = ""
        
        yield f"data: {json.dumps({'type': 'progress', 'step': 'generating_visuals', 'status': 'complete', 'message': 'Visual assets loaded successfully.'})}\n\n"
        
        # Step 4: Generate voice narration
        yield f"data: {json.dumps({'type': 'progress', 'step': 'creating_narration', 'status': 'in_progress', 'message': 'Synthesizing teacher audio narration...'})}\n\n"
        await asyncio.sleep(0.4)
        
        # Ensure slide generator compatibility
        for clip in clips:
            if "scene_no" not in clip:
                clip["scene_no"] = clip.get("clip_no", 1)
            if "teacher_script" not in clip:
                clip["teacher_script"] = clip.get("teacher_script", "")

        audio_urls = await generate_slide_audio(clips, lesson_id)
        
        # Map generated audio URLs to the clips
        for idx, clip in enumerate(clips):
            clip["audio_url"] = audio_urls[idx] if idx < len(audio_urls) else ""
            
        yield f"data: {json.dumps({'type': 'progress', 'step': 'creating_narration', 'status': 'complete', 'message': 'Teacher narration synthesized.'})}\n\n"
        
        # Save the lesson package JSON to disk first
        MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "..", ".."))
        lesson_dir = os.path.join(PROJECT_ROOT, "uploads", "visual_lessons", lesson_id)
        os.makedirs(lesson_dir, exist_ok=True)
        json_path = os.path.join(lesson_dir, "lesson.json")
        
        lesson_package = {
            "lesson_title": blueprint.get("lesson_title", "Visual Lesson"),
            "layout_mode": layout_mode,
            "theme": theme,
            "global_assets": global_assets,
            "connections": connections,
            "lesson_id": lesson_id,
            "scenes": clips
        }
        
        json_path = os.path.join(lesson_dir, "lesson.json")
        storyboard_path = os.path.join(lesson_dir, "storyboard.json")
        
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(lesson_package, f, indent=2, ensure_ascii=False)
            with open(storyboard_path, "w", encoding="utf-8") as f:
                json.dump(lesson_package, f, indent=2, ensure_ascii=False)
            print(f"\n======================================================================")
            print(f"🎬 [STORYBOARD JSON GENERATED & SAVED]")
            print(f"📂 Location: {json_path}")
            print(f"======================================================================\n")
            logger.info(f"[VisualLearning] Saved scene lesson blueprint JSON to {json_path} & {storyboard_path}")
        except Exception as json_err:
            logger.error(f"[VisualLearning] Failed to save lesson.json: {json_err}")
        
        # Step 5: Execute Hyperframes Engine Master Composition Compilation (< 0.1s fast-path)
        print("\n======================================================================")
        print("🚀 [PIPELINE DEBUG] ENTER Hyperframes")
        print("======================================================================\n")
        yield f"data: {json.dumps({'type': 'progress', 'step': 'hyperframes_engine', 'status': 'in_progress', 'message': '[Hyperframes Engine] Compiling master composition & scene templates...'})}\n\n"
        
        try:
            from backend.app.services.visual_learning.hyperframes_engine_bridge import compile_hyperframes_html_fast
            
            # Compile master HTML composition (< 0.1s)
            html_url = await compile_hyperframes_html_fast(lesson_id, lesson_dir)
            if html_url:
                lesson_package["html_url"] = html_url
                logger.info(f"[VisualLearning] Hyperframes HTML master composition successfully attached: {html_url}")
                yield f"data: {json.dumps({'type': 'progress', 'step': 'hyperframes_engine', 'status': 'complete', 'message': f'[Hyperframes Success] Master composition compiled: {html_url}'})}\n\n"
            # Check for custom_lesson.mp4 video file
            expected_mp4 = os.path.join(lesson_dir, "custom_lesson.mp4")
            if os.path.exists(expected_mp4):
                lesson_package["video_url"] = f"/uploads/visual_lessons/{lesson_id}/custom_lesson.mp4"
                logger.info(f"[VisualLearning] Hyperframes MP4 video file attached: {lesson_package['video_url']}")

        except Exception as hf_err:
            logger.error(f"[VisualLearning] Hyperframes execution encountered error: {hf_err}", exc_info=True)
            yield f"data: {json.dumps({'type': 'progress', 'step': 'hyperframes_engine', 'status': 'warn', 'message': f'[Hyperframes Notice] Notice: {hf_err}'})}\n\n"

        # Step 6: Launching lesson package with Hyperframes player
        yield f"data: {json.dumps({'type': 'progress', 'step': 'launching_lesson', 'status': 'complete', 'message': 'Launching Hyperframes player...'})}\n\n"
        await asyncio.sleep(0.2)
        
        yield f"data: {json.dumps({'type': 'lesson_ready', 'lesson': lesson_package})}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"[VisualLearning] Failed to stream visual lesson storyboard: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to generate visual lesson: {str(e)}'})}\n\n"

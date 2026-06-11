import json
import uuid
import logging
import asyncio
import os
from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.services.visual_learning.visual_lesson_prompt import get_visual_lesson_prompt
from backend.app.services.visual_learning.asset_retrieval_service import retrieve_asset_url
from backend.app.services.visual_learning.visual_audio_generator import generate_slide_audio

logger = logging.getLogger(__name__)

async def generate_visual_lesson_stream(query: str, book_uuid: str, class_name: str, subject: str):
    """
    Main pipeline to generate a visual lesson storyboard.
    Streams progress states and returns the completed lesson JSON package.
    """
    lesson_id = f"vl_{uuid.uuid4().hex[:8]}"
    
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
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text
                
            blueprint = json.loads(json_str)
            scenes = blueprint.get("scenes", [])
            logger.info(f"[VisualLearning] Parsed storyboard successfully with {len(scenes)} scenes.")
        except Exception as e:
            logger.error(f"[VisualLearning] Failed to parse storyboard JSON. Raw response:\n{response_text}\nError: {e}", exc_info=True)
            raise ValueError(f"Failed to parse storyboard JSON from Gemini response: {e}")
        
        if not scenes:
            raise ValueError("Gemini storyboard does not contain any scenes.")
            
        yield f"data: {json.dumps({'type': 'progress', 'step': 'designing_lesson', 'status': 'complete', 'message': f'Designed storyboard with {len(scenes)} scenes.'})}\n\n"
        
        # Step 3: Retrieve visual assets for scenes
        yield f"data: {json.dumps({'type': 'progress', 'step': 'generating_visuals', 'status': 'in_progress', 'message': 'Retrieving educational visual assets...'})}\n\n"
        await asyncio.sleep(0.4)
        
        # Resolve search queries using Wikimedia / Openverse Asset Retrieval Engine
        for scene in scenes:
            scene_no = scene.get("scene_no", 1)
            assets = scene.get("assets", [])
            for asset in assets:
                asset_type = asset.get("type", "image")
                query_str = asset.get("search_query", "")
                
                if asset_type == "image":
                    logger.info(f"[VisualLearning] Searching asset for scene {scene_no}: '{query_str}'")
                    asset_url = await retrieve_asset_url(query_str)
                    asset["asset_url"] = asset_url
                else:
                    # Lottie preset elements will be resolved in the client player
                    asset["asset_url"] = ""
        
        yield f"data: {json.dumps({'type': 'progress', 'step': 'generating_visuals', 'status': 'complete', 'message': 'Visual assets loaded successfully.'})}\n\n"
        
        # Step 4: Generate voice narration
        yield f"data: {json.dumps({'type': 'progress', 'step': 'creating_narration', 'status': 'in_progress', 'message': 'Synthesizing teacher audio narration...'})}\n\n"
        await asyncio.sleep(0.4)
        
        audio_urls = await generate_slide_audio(scenes, lesson_id)
        
        # Map generated audio URLs to the scenes
        for idx, scene in enumerate(scenes):
            scene["audio_url"] = audio_urls[idx] if idx < len(audio_urls) else ""
            
        yield f"data: {json.dumps({'type': 'progress', 'step': 'creating_narration', 'status': 'complete', 'message': 'Teacher narration synthesized.'})}\n\n"
        
        # Step 5: Assembling package
        yield f"data: {json.dumps({'type': 'progress', 'step': 'launching_lesson', 'status': 'in_progress', 'message': 'Launching media player...'})}\n\n"
        await asyncio.sleep(0.4)
        
        lesson_package = {
            "lesson_title": blueprint.get("lesson_title", "Visual Lesson"),
            "lesson_type": blueprint.get("lesson_type", "conceptual"),
            "lesson_id": lesson_id,
            "scenes": scenes
        }
        
        # Save the lesson package JSON to disk
        MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "..", ".."))
        lesson_dir = os.path.join(PROJECT_ROOT, "uploads", "visual_lessons", lesson_id)
        os.makedirs(lesson_dir, exist_ok=True)
        json_path = os.path.join(lesson_dir, "lesson.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(lesson_package, f, indent=2, ensure_ascii=False)
            logger.info(f"[VisualLearning] Saved scene lesson blueprint JSON to {json_path}")
        except Exception as json_err:
            logger.error(f"[VisualLearning] Failed to save lesson.json: {json_err}")
        
        yield f"data: {json.dumps({'type': 'lesson_ready', 'lesson': lesson_package})}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"[VisualLearning] Failed to stream visual lesson storyboard: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to generate visual lesson: {str(e)}'})}\n\n"

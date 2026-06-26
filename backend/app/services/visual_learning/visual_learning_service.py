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
            clips = blueprint.get("clips", blueprint.get("scenes", []))
            global_assets = blueprint.get("global_assets", [])
            connections = blueprint.get("connections", [])
            layout_mode = blueprint.get("layout_mode", "timeline")
            theme = blueprint.get("theme", "indigo")
            
            logger.info(f"[VisualLearning] Parsed storyboard successfully with {len(clips)} clips.")
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
        
        # Step 5: Assembling package
        yield f"data: {json.dumps({'type': 'progress', 'step': 'launching_lesson', 'status': 'in_progress', 'message': 'Launching media player...'})}\n\n"
        await asyncio.sleep(0.4)
        
        lesson_package = {
            "lesson_title": blueprint.get("lesson_title", "Visual Lesson"),
            "layout_mode": layout_mode,
            "theme": theme,
            "global_assets": global_assets,
            "connections": connections,
            "lesson_id": lesson_id,
            "scenes": clips
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
            
            # Copy hook to standalone Remotion test app
            try:
                import shutil
                remotion_dest_dir = os.path.join(PROJECT_ROOT, "remotion_test_app", "public", "uploads", "visual_lessons", lesson_id)
                os.makedirs(os.path.dirname(remotion_dest_dir), exist_ok=True)
                if os.path.exists(remotion_dest_dir):
                    shutil.rmtree(remotion_dest_dir)
                shutil.copytree(lesson_dir, remotion_dest_dir)
                logger.info(f"[VisualLearning] Successfully copied storyboard and audio files to Remotion public uploads: {remotion_dest_dir}")
            except Exception as copy_err:
                logger.error(f"[VisualLearning] Failed to copy lesson files to Remotion app: {copy_err}")
                
        except Exception as json_err:
            logger.error(f"[VisualLearning] Failed to save lesson.json: {json_err}")
        
        yield f"data: {json.dumps({'type': 'lesson_ready', 'lesson': lesson_package})}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"[VisualLearning] Failed to stream visual lesson storyboard: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to generate visual lesson: {str(e)}'})}\n\n"

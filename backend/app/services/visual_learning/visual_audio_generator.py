import os
import re
import base64
import logging
import httpx

logger = logging.getLogger(__name__)

SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_MAX_CHARS = 2500

class AudioGenerationError(Exception):
    """Custom exception raised when TTS audio generation fails."""
    pass

def _split_text(text: str, max_chars: int = SARVAM_MAX_CHARS) -> list[str]:
    """
    Splits slide script text into chunks at sentence boundaries 
    to fit within Sarvam's API character limits.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.?!])\s+', text)

    for sentence in sentences:
        if len(sentence) > max_chars:
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
            continue

        if len(current) + len(sentence) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current.strip())

    return chunks

async def generate_slide_audio(slides: list, lesson_id: str) -> list:
    """
    Generates narration audio for all scenes/slides.
    Saves WAV files to disk and returns a list of relative audio URLs.
    """
    import time
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "..", ".."))
    lesson_dir = os.path.join(PROJECT_ROOT, "uploads", "visual_lessons", lesson_id)
    try:
        os.makedirs(lesson_dir, exist_ok=True)
        test_file = os.path.join(lesson_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("1")
        os.remove(test_file)
    except Exception:
        lesson_dir = os.path.join("/tmp", "uploads", "visual_lessons", lesson_id)
        os.makedirs(lesson_dir, exist_ok=True)
    
    api_key = os.getenv("SARVAM_API_KEY", "")
    
    # 1-second silent WAV base64 bytes for offline fallback testing
    dummy_wav_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQQAAAAAAA=="
    audio_urls = []
    
    for slide_idx, slide in enumerate(slides, 1):
        slide_no = slide.get("scene_no", slide_idx)
        if not isinstance(slide_no, int):
            slide_no = slide_idx
        text = slide.get("teacher_script", "").strip()
        if not text:
            text = f"Scene {slide_no}."
            
        wav_filename = f"scene_{slide_no}.wav"
        wav_path = os.path.join(lesson_dir, wav_filename)
        
        # Soft fallback for local development if api key is missing
        if not api_key:
            logger.warning(f"[AudioGen] SARVAM_API_KEY not configured. Saving mock WAV for scene {slide_no}.")
            print(f"⚠️ [AudioGen] SARVAM_API_KEY is missing. Saving mock silent audio for scene {slide_no}.")
            with open(wav_path, "wb") as f:
                f.write(base64.b64decode(dummy_wav_b64))
            audio_urls.append(f"/uploads/visual_lessons/{lesson_id}/{wav_filename}")
            continue
            
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }
        
        chunks = _split_text(text)
        logger.info(f"[AudioGen] Scene {slide_no} narration script has {len(text)} characters. Split into {len(chunks)} chunks.")
        print(f"   [AudioGen] Scene {slide_no} script: '{text[:60]}...' ({len(text)} chars)")
        all_audio_bytes = b""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for idx, chunk in enumerate(chunks):
                    payload = {
                        "text": chunk,
                        "target_language_code": "en-IN",
                        "speaker": "ritu", # Using the default teacher speaker
                        "model": "bulbul:v3",
                        "enable_preprocessing": True,
                    }
                    logger.info(f"[AudioGen] Calling Sarvam Bulbul v3 API for scene {slide_no}, chunk {idx+1}/{len(chunks)} ({len(chunk)} chars)...")
                    print(f"   [AudioGen] Calling Sarvam TTS (scene {slide_no}, chunk {idx+1}/{len(chunks)})...")
                    
                    start_time = time.time()
                    response = await client.post(SARVAM_API_URL, headers=headers, json=payload)
                    duration = time.time() - start_time
                    
                    logger.info(f"[AudioGen] Sarvam Response: status={response.status_code} | time={duration:.2f}s")
                    print(f"   [AudioGen] Sarvam Response: status={response.status_code} | time={duration:.2f}s")
                    
                    if response.status_code != 200:
                        err_msg = f"Sarvam API error: Status {response.status_code} - {response.text}"
                        logger.error(f"[AudioGen] {err_msg}")
                        print(f"❌ [AudioGen ERROR] {err_msg}")
                        raise AudioGenerationError(err_msg)
                        
                    data = response.json()
                    audios = data.get("audios", [])
                    if not audios:
                        logger.error(f"[AudioGen] Sarvam returned empty audios array for scene {slide_no}")
                        raise AudioGenerationError("Sarvam returned empty audio list.")
                    
                    all_audio_bytes += base64.b64decode(audios[0])
            
            with open(wav_path, "wb") as f:
                f.write(all_audio_bytes)
                
            # Upload to Supabase Cloud Storage (with local fallback)
            from backend.app.core.supabase_storage import upload_file_to_supabase
            cloud_audio_url = upload_file_to_supabase(wav_path, f"{lesson_id}/{wav_filename}")
            final_audio_url = cloud_audio_url or f"/uploads/visual_lessons/{lesson_id}/{wav_filename}"
            audio_urls.append(final_audio_url)
            
            logger.info(f"[RENDER LOG] [AUDIO TTS SUCCESS] Scene {slide_no} audio ready ({len(all_audio_bytes)} bytes) -> {final_audio_url}")
            try:
                print(f"[RENDER LOG] [AUDIO TTS SUCCESS] Scene {slide_no} audio ready ({len(all_audio_bytes)} bytes) -> {final_audio_url}")
            except Exception:
                pass
            
        except Exception as e:
            logger.warning(f"[AudioGen] Notice/fallback generating audio for scene {slide_no}: {e}")
            try:
                print(f"[RENDER LOG] [AUDIO TTS NOTICE] Scene {slide_no} using silent fallback audio: {e}")
            except Exception:
                pass
            try:
                with open(wav_path, "wb") as f:
                    f.write(base64.b64decode(dummy_wav_b64))
                from backend.app.core.supabase_storage import upload_file_to_supabase
                cloud_audio_url = upload_file_to_supabase(wav_path, f"{lesson_id}/{wav_filename}")
                final_audio_url = cloud_audio_url or f"/uploads/visual_lessons/{lesson_id}/{wav_filename}"
                audio_urls.append(final_audio_url)
            except Exception as write_err:
                logger.error(f"[AudioGen] Failed writing fallback audio: {write_err}")
                audio_urls.append(None)
            
    return audio_urls

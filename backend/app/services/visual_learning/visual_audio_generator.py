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
    Generates narration audio for all slides.
    Saves WAV files to disk and returns a list of relative audio URLs.
    """
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "..", ".."))
    lesson_dir = os.path.join(PROJECT_ROOT, "uploads", "visual_lessons", lesson_id)
    os.makedirs(lesson_dir, exist_ok=True)
    
    api_key = os.getenv("SARVAM_API_KEY", "")
    
    # 1-second silent WAV base64 bytes for offline fallback testing
    dummy_wav_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQQAAAAAAA=="
    audio_urls = []
    
    for slide in slides:
        slide_no = slide.get("slide_no", 1)
        text = slide.get("teacher_script", "").strip()
        if not text:
            text = f"Slide {slide_no}."
            
        wav_filename = f"slide_{slide_no}.wav"
        wav_path = os.path.join(lesson_dir, wav_filename)
        
        # Soft fallback for local development if api key is missing
        if not api_key:
            logger.warning(f"[AudioGen] SARVAM_API_KEY not configured. Saving mock WAV for slide {slide_no}.")
            with open(wav_path, "wb") as f:
                f.write(base64.b64decode(dummy_wav_b64))
            audio_urls.append(f"/uploads/visual_lessons/{lesson_id}/{wav_filename}")
            continue
            
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }
        
        chunks = _split_text(text)
        all_audio_bytes = b""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for chunk in chunks:
                    payload = {
                        "text": chunk,
                        "target_language_code": "en-IN",
                        "speaker": "ritu", # Using the default teacher speaker
                        "model": "bulbul:v3",
                        "enable_preprocessing": True,
                    }
                    logger.info(f"[AudioGen] Calling Sarvam for slide {slide_no}...")
                    response = await client.post(SARVAM_API_URL, headers=headers, json=payload)
                    
                    if response.status_code != 200:
                        raise AudioGenerationError(f"Sarvam API error: {response.status_code} - {response.text}")
                        
                    data = response.json()
                    audios = data.get("audios", [])
                    if not audios:
                        raise AudioGenerationError("Sarvam returned empty audio list.")
                    
                    all_audio_bytes += base64.b64decode(audios[0])
            
            with open(wav_path, "wb") as f:
                f.write(all_audio_bytes)
                
            audio_urls.append(f"/uploads/visual_lessons/{lesson_id}/{wav_filename}")
            logger.info(f"[AudioGen] Successfully generated audio for slide {slide_no}")
            
        except Exception as e:
            logger.error(f"[AudioGen] Error generating audio for slide {slide_no}: {e}")
            raise AudioGenerationError(f"Failed to generate audio for slide {slide_no}: {str(e)}")
            
    return audio_urls

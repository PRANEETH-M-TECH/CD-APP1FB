import os
import httpx
import hashlib
import logging
import base64
import re
from typing import Tuple
from backend.app.core.redis_service import redis_service

logger = logging.getLogger(__name__)

SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"
TTS_CACHE_TTL = 604800  # 7 days expiration for cached audio bytes

def _split_text(text: str, max_chars: int = 2000) -> list[str]:
    """Splits text into chunks of at most max_chars, trying to split on sentences."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks = []
    current_chunk = ""

    for s in sentences:
        if len(current_chunk) + len(s) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + s).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = s.strip()

    if current_chunk:
        chunks.append(current_chunk)

    # Hard boundary fallback for extremely long sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i:i+max_chars])
        else:
            final_chunks.append(chunk)

    return final_chunks

async def synthesize_text_cached(
    text: str,
    language: str = "en-IN",
    speaker: str = "ritu",
    model: str = "bulbul:v3"
) -> Tuple[bytes, str]:
    """
    Unified text-to-speech synthesis function with Redis-backed caching.
    Prevents calling the external Sarvam API twice for identical text strings.
    
    Returns (raw_audio_bytes, format_string).
    """
    text = text.strip()
    if not text:
        return b"", "wav"

    api_key = os.getenv("SARVAM_API_KEY", "")
    if not api_key:
        logger.warning("[TTS Service] SARVAM_API_KEY missing. Returning mock silent audio.")
        # Return a silent WAV header fallback
        dummy_wav_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQQAAAAAAA=="
        return base64.b64decode(dummy_wav_b64), "wav"

    chunks = _split_text(text)
    all_audio_bytes = b""

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            # Generate a unique cache key based on text content, speaker, and model details
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            cache_key = f"tts_cache:{model}:{speaker}:{chunk_hash}"
            
            cached_audio = None
            try:
                # Retrieve from Redis
                cached_data = redis_service.r.get(cache_key)
                if cached_data:
                    # Redis stores bytes or strings; handle appropriately
                    if isinstance(cached_data, str):
                        cached_audio = base64.b64decode(cached_data)
                    else:
                        cached_audio = cached_data
            except Exception as cache_err:
                logger.warning(f"[TTS Service] Cache read error: {cache_err}")

            if cached_audio:
                logger.info(f"[TTS CACHE HIT] Reusing cached audio for chunk: '{chunk[:40]}...'")
                print(f"🚀 [TTS CACHE HIT] Reusing cached audio for chunk: '{chunk[:40]}...'", flush=True)
                all_audio_bytes += cached_audio
            else:
                # Call Sarvam Bulbul API
                logger.info(f"[TTS CACHE MISS] Synthesizing via Sarvam for: '{chunk[:40]}...'")
                print(f"📡 [TTS CACHE MISS] Call Sarvam API for chunk: '{chunk[:40]}...'", flush=True)
                
                payload = {
                    "text": chunk,
                    "target_language_code": language,
                    "speaker": speaker,
                    "model": model,
                    "enable_preprocessing": True,
                }
                
                headers = {
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json",
                }
                
                try:
                    response = await client.post(SARVAM_API_URL, headers=headers, json=payload)
                    if response.status_code != 200:
                        raise Exception(f"Sarvam API status {response.status_code}: {response.text}")
                    
                    data = response.json()
                    audios = data.get("audios", [])
                    if not audios:
                        raise Exception("Sarvam API returned empty audio array.")
                    
                    chunk_audio_bytes = base64.b64decode(audios[0])
                    all_audio_bytes += chunk_audio_bytes
                    
                    # Store in Redis cache
                    try:
                        redis_service.r.setex(cache_key, TTS_CACHE_TTL, chunk_audio_bytes)
                    except Exception as cache_write_err:
                        logger.warning(f"[TTS Service] Cache write error: {cache_write_err}")
                        
                except Exception as tts_err:
                    logger.error(f"[TTS Service] Synthesis error for chunk: {tts_err}")
                    raise

    return all_audio_bytes, "wav"

"""
TTS (Text-to-Speech) proxy endpoint.
Keeps all API keys server-side and routes requests to the selected TTS provider.

Supported models (added incrementally):
  - sarvam   : Sarvam AI Bulbul v3  ✅ (Step 1)
  - google   : Google Cloud TTS     🔲 (Step 2)
  - azure    : Microsoft Azure TTS  🔲 (Step 3)
  - bhashini : Bhashini Digital     🔲 (Step 4)
  - indictts : AI4Bharat IndicTTS   🔲 (Step 5)
"""

import os
import base64
import logging
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    model: str = "sarvam"          # which TTS provider to use
    language: str = "en-IN"        # BCP-47 language code
    speaker: str = "meera"         # voice/speaker name (provider-specific)


class TTSResponse(BaseModel):
    audio_base64: str              # base64-encoded audio bytes
    format: str                    # "wav" | "mp3"
    model: str                     # which model was actually used


# ---------------------------------------------------------------------------
# Sarvam AI — Bulbul v3
# ---------------------------------------------------------------------------

SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"

# Max chars Sarvam accepts per request
SARVAM_MAX_CHARS = 2500

# Available Sarvam speakers for bulbul:v3
SARVAM_SPEAKERS = [
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan", 
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", 
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand", 
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani", 
    "mohit", "kavitha", "rehan", "soham", "rupali", "niharika"
]


def _split_text(text: str, max_chars: int = SARVAM_MAX_CHARS) -> list[str]:
    """
    Split long text into chunks at sentence boundaries so each chunk
    fits within the provider's character limit.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""

    # Split on sentence-ending punctuation
    import re
    sentences = re.split(r'(?<=[.?!])\s+', text)

    for sentence in sentences:
        # If a single sentence exceeds the limit, hard-split it
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


async def _call_sarvam(text: str, language: str, speaker: str) -> tuple[bytes, str]:
    """
    Call Sarvam AI Bulbul v3 API.
    Returns (raw_audio_bytes, format_string).
    Handles chunking for text > 2500 chars automatically.
    """
    api_key = os.getenv("SARVAM_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY not configured on server."
        )

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    # Validate / default speaker
    if speaker not in SARVAM_SPEAKERS:
        print(f"[Sarvam] Unknown speaker '{speaker}', defaulting to 'ritu'")
        speaker = "ritu"

    chunks = _split_text(text)
    print(f"   [SARVAM API] Splitting text into {len(chunks)} chunk(s).")
    logger.info(f"[Sarvam] Sending {len(chunks)} chunk(s) | lang={language} | speaker={speaker}")

    all_audio_bytes = b""

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, chunk in enumerate(chunks):
            payload = {
                "text": chunk,
                "target_language_code": language,
                "speaker": speaker,
                "model": "bulbul:v3",
                "enable_preprocessing": True,
            }

            print(f"   [SARVAM API] Chunk {i+1}/{len(chunks)}: Calling https://api.sarvam.ai/text-to-speech with speaker='{speaker}'")
            logger.info(f"[Sarvam] Chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
            response = await client.post(SARVAM_API_URL, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"❌ [SARVAM API ERROR] Status {response.status_code}: {response.text}")
                logger.error(f"[Sarvam] API error {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Sarvam API returned {response.status_code}: {response.text}"
                )

            data = response.json()
            print(f"   [SARVAM API] Chunk {i+1} OK (200). Decoding audio...")

            # Sarvam returns: { "audios": ["<base64_wav_string>"] }
            audios = data.get("audios", [])
            if not audios:
                raise HTTPException(status_code=502, detail="Sarvam returned empty audio list.")

            # Decode base64 WAV and concatenate
            chunk_audio = base64.b64decode(audios[0])
            all_audio_bytes += chunk_audio

    return all_audio_bytes, "wav"


# ---------------------------------------------------------------------------
# Microsoft Azure TTS
# ---------------------------------------------------------------------------

async def _call_azure(text: str, language: str, speaker: str) -> tuple[bytes, str]:
    """
    Call Microsoft Azure TTS API.
    Returns (raw_audio_bytes, format_string).
    Constructs SSML and POSTs to the Azure REST API.
    """
    api_key = os.getenv("AZURE_TTS_API_KEY", "")
    region = os.getenv("AZURE_TTS_REGION", "")
    
    if not api_key or not region:
        raise HTTPException(
            status_code=503,
            detail="AZURE_TTS_API_KEY or AZURE_TTS_REGION not configured on server."
        )

    # Azure requires SSML (Speech Synthesis Markup Language)
    ssml = f"""
    <speak version="1.0" xml:lang="{language}">
        <voice name="{speaker}">
            {text}
        </voice>
    </speak>
    """

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
        "User-Agent": "ChaduvuGuruTTS"
    }

    # Azure endpoint is dynamic based on region
    azure_url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    print(f"   [AZURE API] Calling {azure_url} with speaker='{speaker}'")
    logger.info(f"[Azure] Sending request | lang={language} | speaker={speaker}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(azure_url, headers=headers, content=ssml)

        if response.status_code != 200:
            print(f"❌ [AZURE API ERROR] Status {response.status_code}: {response.text}")
            logger.error(f"[Azure] API error {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Azure API returned {response.status_code}: {response.text}"
            )

        print(f"   [AZURE API] OK (200). Received MP3 audio stream.")
        return response.content, "mp3"


# ---------------------------------------------------------------------------
# Main /api/tts endpoint

# ---------------------------------------------------------------------------

@router.post("/api/tts", tags=["TTS"])
async def text_to_speech(request: TTSRequest):
    """
    Proxy endpoint for TTS. Accepts text + model selection,
    returns base64-encoded audio.

    Supported models: sarvam (more coming in steps 2-5)
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    model = request.model.lower()
    
    print("\n" + "="*60)
    print(f"🔊 [TTS TRIGGERED] User requested speech")
    print(f"   Model:   {model}")
    print(f"   Speaker: {request.speaker}")
    print(f"   Chars:   {len(text)}")
    print(f"   Cost:    Estimated {len(text)} characters billed to {model.upper()}")
    print("="*60 + "\n")

    logger.info(f"[TTS] Request: model={model}, lang={request.language}, speaker={request.speaker}, chars={len(text)}")

    try:
        if model == "sarvam":
            from backend.app.services.chat.tts_service import synthesize_text_cached
            audio_bytes, fmt = await synthesize_text_cached(
                text=text,
                language=request.language,
                speaker=request.speaker,
            )
        elif model == "azure":
            print(f"📡 [AZURE API] Preparing payload for Microsoft Azure TTS...")
            audio_bytes, fmt = await _call_azure(
                text=text,
                language=request.language,
                speaker=request.speaker,
            )
            print(f"✅ [AZURE API] Success! Received {len(audio_bytes)} bytes of audio.")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown TTS model: '{model}'. Supported: sarvam"
            )

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        logger.info(f"[TTS] Success: model={model}, audio_size={len(audio_bytes)} bytes")
        return JSONResponse(content={
            "audio_base64": audio_b64,
            "format": fmt,
            "model": model,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TTS] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /api/tts/voices — return available voices for a given model
# ---------------------------------------------------------------------------

@router.get("/api/tts/voices", tags=["TTS"])
async def get_tts_voices(model: str = "sarvam"):
    """
    Returns available voices/speakers for the given TTS model.
    Frontend uses this to populate the voice dropdown.
    """
    model = model.lower()

    if model == "sarvam":
        return {
            "model": "sarvam",
            "voices": [
                {"id": "ritu",      "name": "Ritu",      "gender": "Female"},
                {"id": "priya",     "name": "Priya",     "gender": "Female"},
                {"id": "neha",      "name": "Neha",      "gender": "Female"},
                {"id": "pooja",     "name": "Pooja",     "gender": "Female"},
                {"id": "simran",    "name": "Simran",    "gender": "Female"},
                {"id": "shruti",    "name": "Shruti",    "gender": "Female"},
                {"id": "kavitha",   "name": "Kavitha",   "gender": "Female"},
                {"id": "aditya",    "name": "Aditya",    "gender": "Male"},
                {"id": "rahul",     "name": "Rahul",     "gender": "Male"},
                {"id": "rohan",     "name": "Rohan",     "gender": "Male"},
                {"id": "amit",      "name": "Amit",      "gender": "Male"},
                {"id": "vijay",     "name": "Vijay",     "gender": "Male"},
            ]
        }
    
    if model == "azure":
        return {
            "model": "azure",
            "voices": [
                {"id": "en-IN-NeerjaNeural",  "name": "Neerja",  "gender": "Female"},
                {"id": "en-IN-KavitaNeural",  "name": "Kavita",  "gender": "Female"},
                {"id": "en-IN-AashiNeural",   "name": "Aashi",   "gender": "Female"},
                {"id": "en-IN-PrabhatNeural", "name": "Prabhat", "gender": "Male"},
                {"id": "en-IN-AaravNeural",   "name": "Aarav",   "gender": "Male"},
                {"id": "en-IN-RehaanNeural",  "name": "Rehaan",  "gender": "Male"},
            ]
        }

    return {"model": model, "voices": []}

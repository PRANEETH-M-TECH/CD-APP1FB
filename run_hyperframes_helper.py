#!/usr/bin/env python3
import os
import sys
import json
import uuid
import base64
import time
import httpx
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import urllib.request
import urllib.error

# Force UTF-8 stdout encoding to avoid Windows console crashes with special characters
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

BACKEND_URL = "http://localhost:8000"
SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"

def get_books():
    print(f"Fetching books from {BACKEND_URL}/api/books...")
    try:
        req = urllib.request.Request(f"{BACKEND_URL}/api/books")
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"\n[WARNING] Could not connect to the backend server at {BACKEND_URL}.")
        print("System Mode requires the FastAPI backend to be running.")
        print("If you want to run offline, choose General Mode [2] instead.\n")
        return None
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch books: {e}")
        return []

async def generate_sarvam_tts(text: str, lesson_id: str, scene_no: int, api_key: str, dest_dir: Path):
    """
    Directly calls the Sarvam AI TTS API for General Mode
    """
    if not api_key:
        # Save a mock WAV if no API key is present
        print(f"   [TTS] No SARVAM_API_KEY. Creating mock audio for scene {scene_no}")
        dummy_wav_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQQAAAAAAA=="
        audio_bytes = base64.b64decode(dummy_wav_b64)
        wav_path = dest_dir / f"scene_{scene_no}.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
        return f"/uploads/visual_lessons/{lesson_id}/scene_{scene_no}.wav"

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "target_language_code": "en-IN",
        "speaker": "ritu",
        "model": "bulbul:v3",
        "enable_preprocessing": True,
    }

    print(f"   [TTS] Generating audio for scene {scene_no} using Sarvam AI...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        start_time = time.time()
        response = await client.post(SARVAM_API_URL, headers=headers, json=payload)
        duration = time.time() - start_time
        
        if response.status_code != 200:
            print(f"   ❌ [TTS ERROR] Sarvam API returned status {response.status_code}: {response.text}")
            # fallback
            dummy_wav_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQQAAAAAAA=="
            audio_bytes = base64.b64decode(dummy_wav_b64)
            wav_path = dest_dir / f"scene_{scene_no}.wav"
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
            return f"./scene_{scene_no}.wav"
            
        data = response.json()
        audios = data.get("audios", [])
        if not audios:
            print(f"   ❌ [TTS ERROR] Empty audio array received.")
            raise RuntimeError("Empty audio array from Sarvam")
            
        audio_bytes = base64.b64decode(audios[0])
        wav_path = dest_dir / f"scene_{scene_no}.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
        print(f"   ✅ [TTS SUCCESS] Saved audio to {wav_path.name} ({len(audio_bytes)} bytes, {duration:.2f}s)")
        return f"./scene_{scene_no}.wav"

async def run_general_mode():
    print("\n--------------------------------------------------")
    print("                RUNNING GENERAL MODE              ")
    print("--------------------------------------------------")
    
    gemini_key = os.getenv("GOOGLE_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    
    if not gemini_key:
        print("\n[ERROR] GOOGLE_API_KEY is not defined in your .env file.")
        print("General Mode requires a Gemini API Key to generate storyboards.")
        return
        
    query = input("\nEnter your storyboard topic (e.g. 'explain molecular structure of water'): ").strip()
    while not query:
        query = input("Topic cannot be empty. Enter topic: ").strip()

    theme_choice = input("Enter theme (Science/Math/History/Civics/General) [Default: Science]: ").strip()
    if not theme_choice:
        theme_choice = "Science"
        
    print("\n1. Generating storyboard JSON using Gemini...")
    
    # Load system prompt
    prompt_file = Path(__file__).parent / 'hyperframes_test_app' / 'system_prompt.md'
    if prompt_file.exists():
        system_instruction = prompt_file.read_text(encoding='utf-8')
    else:
        system_instruction = "Generate an educational storyboard lesson JSON."

    client = genai.Client(api_key=gemini_key)
    
    full_prompt = f"Create a storyboard about: '{query}'. Use the theme '{theme_choice}'."
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json"
            }
        )
        storyboard_json = json.loads(response.text)
    except Exception as e:
        print(f"❌ Failed to generate storyboard JSON from Gemini: {e}")
        return

    lesson_id = f"vl_{uuid.uuid4().hex[:8]}"
    dest_dir = Path(__file__).parent / 'hyperframes_test_app' / 'outputs' / lesson_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Generated storyboard ID: {lesson_id}")
    print(f"   Saved temporary files in: {dest_dir}")
    
    # 2. Generate Audio via Sarvam TTS for each scene
    print("\n2. Generating scene audio using Sarvam TTS...")
    scenes = storyboard_json.get("scenes", [])
    if not scenes:
        scenes = storyboard_json.get("clips", [])
        storyboard_json["scenes"] = scenes

    for scene in scenes:
        scene_no = scene.get("scene_no", 1)
        script = scene.get("teacher_script", "")
        audio_url = await generate_sarvam_tts(script, lesson_id, scene_no, sarvam_key, dest_dir)
        scene["audio_url"] = audio_url

    # Save final JSON package
    storyboard_json["lesson_id"] = lesson_id
    storyboard_json["theme"] = theme_choice
    storyboard_json["lesson_title"] = storyboard_json.get("lesson_title", query)
    
    json_path = dest_dir / 'lesson.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(storyboard_json, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ All assets generated successfully!")
    print(f"   Storyboard JSON package written to: {json_path}")
    print(f"\nNext Steps:")
    print(f"  1. Navigate to: cd hyperframes_test_app")
    print(f"  2. Run: node run-storyboard.js")
    print(f"  3. Select action [1] to Preview, or [2] to Render.")

def run_system_mode(books):
    selected_book = None
    if books:
        print("\nAvailable Books in Database:")
        for idx, book in enumerate(books):
            book_id = book.get("book_uuid", book.get("id", "Unknown"))
            title = book.get("title", book.get("book_name", "Untitled"))
            class_name = book.get("class_name", "Unknown Class")
            subject = book.get("subject", "Unknown Subject")
            print(f"[{idx + 1}] Title: {title}")
            print(f"    Class: {class_name} | Subject: {subject}")
            print(f"    UUID: {book_id}")
            print("-" * 50)
            
        while True:
            choice = input(f"\nSelect a book number (1-{len(books)}): ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(books):
                    selected_book = books[choice_idx]
                    break
            except ValueError:
                pass
            print("Invalid selection. Please enter a valid number.")
    else:
        print("\n[WARNING] No books found in the database. Using fallback book parameters.")
        fallback_uuid = input("Enter standard Book UUID (or press Enter for default 'fallback_book_uuid'): ").strip()
        selected_book = {
            "book_uuid": fallback_uuid if fallback_uuid else "fallback_book_uuid",
            "class_name": "Class 9",
            "subject": "Science"
        }

    book_uuid = selected_book.get("book_uuid", selected_book.get("id"))
    class_name = selected_book.get("class_name")
    subject = selected_book.get("subject")

    print(f"\nConfigured parameters:")
    print(f"  - Book UUID: {book_uuid}")
    print(f"  - Class: {class_name}")
    print(f"  - Subject: {subject}")

    query = input("\nEnter your storyboard topic query (e.g. 'explain structure of neuron'): ").strip()
    while not query:
        query = input("Query cannot be empty. Enter topic query: ").strip()

    payload = {
        "query": query,
        "book_uuid": book_uuid,
        "class_name": class_name,
        "subject": subject
    }

    print(f"\nGenerating storyboard for: '{query}'...")
    
    req = urllib.request.Request(
        f"{BACKEND_URL}/api/visual_learning",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                if line_str.startswith('data:'):
                    data_content = line_str[5:].strip()
                    if data_content == '[DONE]':
                        print("\nFinished stream.")
                        break
                    
                    try:
                        event = json.loads(data_content)
                        event_type = event.get("type")
                        
                        if event_type == "progress":
                            step_msg = event.get("message", "")
                            status = event.get("status", "")
                            if status == "in_progress":
                                print(f" [*] {step_msg}")
                            elif status == "complete":
                                print(f" [OK] {step_msg}")
                        elif event_type == "lesson_ready":
                            lesson = event.get("lesson", {})
                            lesson_id = lesson.get("lesson_id", "")
                            title = lesson.get("lesson_title", "")
                            scenes_count = len(lesson.get("scenes", []))
                            print("\n==============================================")
                            print("         STORYBOARD GENERATION SUCCESS!       ")
                            print("==============================================")
                            print(f" Lesson ID: {lesson_id}")
                            print(f" Title: {title}")
                            print(f" Scenes: {scenes_count}")
                            print("==============================================")
                            print(f"\nSuccess! The storyboard has been generated and saved.")
                            print(f"Go to 'hyperframes_test_app' and run:")
                            print(f"  node run-storyboard.js")
                            print(f"to preview or render the video.")
                        elif event_type == "error":
                            print(f"\n[ERROR] Generation failed: {event.get('message')}")
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as e:
        print(f"\n[HTTP ERROR] Backend responded with status {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

async def main():
    print("==================================================")
    print("      HYPERFRAMES VISUAL STORYBOARD GENERATOR     ")
    print("==================================================")
    print("[1] System Mode (Fetch books & use backend server)")
    print("[2] General Mode (Custom prompt offline generator)")
    print("--------------------------------------------------")
    
    choice = input("Select generation mode (1-2): ").strip()
    if choice == "2":
        await run_general_mode()
    else:
        books = get_books()
        if books is not None:
            run_system_mode(books)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

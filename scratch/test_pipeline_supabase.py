import asyncio
import json
from dotenv import load_dotenv
load_dotenv(override=True)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.services.visual_learning.visual_learning_service import generate_visual_lesson_stream

async def main():
    print("Initializing qdrant service...")
    try:
        qdrant.initialize()
    except Exception as e:
        print("Qdrant init warning:", e)
        
    print("Starting full end-to-end Visual Learning pipeline test...")
    async for raw_event in generate_visual_lesson_stream("what types of foods contain vitamin c?", "test_book_uuid", "Class 10", "Biology"):
        if "lesson_ready" in raw_event:
            print("\n" + "="*80)
            print("[SUCCESS] LESSON READY EVENT RECEIVED:")
            data_json = raw_event.replace("data: ", "").strip()
            parsed = json.loads(data_json)
            print("Lesson ID:", parsed.get("lesson_id"))
            print("Lesson Title:", parsed.get("lesson_title"))
            print("Interactive HTML URL:", parsed.get("interactive_url"))
            print("HTML URL:", parsed.get("html_url"))
            print("Scene Count:", parsed.get("scene_count"))
            print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

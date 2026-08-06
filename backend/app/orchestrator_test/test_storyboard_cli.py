"""
Standalone no-TTS storyboard test harness.

Unlike test_orchestrator_cli.py (classification + RAG only - the current
prompt schema no longer emits a video_storyboard), this calls
generate_visual_lesson_stream() directly - the real storyboard generator
used by the live app - against real Firestore/Qdrant content, and stops
right after the `storyboard_ready` SSE event (before Step 4's Sarvam TTS
call, which isn't reachable in this sandbox). Output is the full scene list
(template_id + template_data) for template/diagram-selection quality
auditing, saved to test_outputs/ alongside the orchestrator reports.

Usage: python -m backend.app.orchestrator_test.test_storyboard_cli
"""
import os
import sys
import json
import asyncio
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.app.orchestrator_test.test_runner import _get_classes_subjects_docs, OUTPUTS_DIR
from backend.app.services.retrieval import qdrant_service
from backend.app.services.visual_learning.visual_learning_service import generate_visual_lesson_stream

qdrant_service.initialize()


def resolve_book_uuid(grade: int, subject: str) -> str:
    for sid, data in _get_classes_subjects_docs(grade):
        if sid.strip().lower() == subject.strip().lower():
            book_uuid = data.get("book_uuid") or ""
            if book_uuid and qdrant_service.book_has_content(book_uuid):
                return book_uuid
    raise ValueError(f"No content-bearing book_uuid found for grade={grade} subject={subject}")


async def generate_storyboard_only(query: str, book_uuid: str, class_name: str, subject: str) -> dict:
    """Drives generate_visual_lesson_stream and returns the storyboard_ready
    payload, without letting the generator proceed into Step 4 TTS."""
    storyboard = None
    async for event in generate_visual_lesson_stream(query, book_uuid, class_name, subject):
        line = event.strip()
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[len("data: "):])
        if payload.get("type") == "storyboard_ready":
            storyboard = payload
            break  # stop consuming - do not let the generator reach Step 4 TTS
        if payload.get("type") == "error":
            raise RuntimeError(payload.get("message", "Unknown storyboard generation error"))
    if storyboard is None:
        raise RuntimeError("Generator finished without emitting storyboard_ready")
    return storyboard


def audit_scene(scene: dict) -> str:
    tid = scene.get("template_id")
    beat = scene.get("beat_shape")
    data = scene.get("template_data") or {}
    tags = []
    if tid == "illustrated_scene":
        if data.get("_curated_diagram_id") or scene.get("_curated_diagram_id"):
            tags.append("curated")
        elif data.get("primitive_shape") or scene.get("_primitive_shape"):
            tags.append(f"primitive:{data.get('primitive_shape') or scene.get('_primitive_shape')}")
        else:
            tags.append("freehand")
        n = len((data.get("elements") or []))
        tags.append(f"elements={n}")
    return f"[{tid}] beat={beat} {' '.join(tags)}"


async def run_case(grade: int, subject: str, query: str):
    book_uuid = resolve_book_uuid(grade, subject)
    print(f"\n{'='*70}\nQUERY ({subject}, grade {grade}): {query}\nbook_uuid={book_uuid}\n{'='*70}")
    storyboard = await generate_storyboard_only(query, book_uuid, str(grade), subject)
    scenes = storyboard.get("scenes", [])
    print(f"Lesson: {storyboard.get('lesson_title')}  |  {len(scenes)} scenes")
    for i, scene in enumerate(scenes, 1):
        print(f"  Scene {i}: {audit_scene(scene)}  purpose={scene.get('purpose', '')[:70]}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subject = subject.replace(" ", "_")
    out_path = os.path.join(OUTPUTS_DIR, f"storyboard_report_{safe_subject}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)
    print(f"  saved -> {out_path}")
    return storyboard


async def main():
    cases = [
        (10, "science", "Explain how the human excretory system removes waste from the body"),
        (10, "social", "Give a timeline of the major events of the Indian nationalist movement between 1920 and 1947"),
        (10, "maths", "Derive the quadratic formula from ax^2 + bx + c = 0 by completing the square"),
    ]
    for grade, subject, query in cases:
        try:
            await run_case(grade, subject, query)
        except Exception as e:
            print(f"  [FAILED] {subject}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

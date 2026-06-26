#!/usr/bin/env python3
import os
import sys
import json
import httpx
import base64
import subprocess
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Reconfigure stdout to UTF-8 to prevent Windows terminal character crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Load Environment Variables
PROJECT_ROOT = Path(__file__).parent.resolve()
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Add project root to sys.path to allow importing backend services
sys.path.append(str(PROJECT_ROOT))
from backend import local_chap_service
from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.services.chat.answer_service import reformulate_with_llm

class TimelineStage(BaseModel):
    step_no: int = Field(description="Step sequence number (1, 2, 3)")
    label: str = Field(description="Short caption for this step (max 2-3 words)")
    icon_name: str = Field(description="Lucide icon name matching context")

class ColumnData(BaseModel):
    header: str = Field(description="Title of this comparison column")
    bullets: List[str] = Field(description="2-3 comparison bullet points")

# SVG element for dynamic illustration generation
class SvgElement(BaseModel):
    type: Literal["circle", "rect", "ellipse", "line", "path"] = Field(description="SVG element type")
    # Circle attributes
    cx: Optional[float] = Field(default=None, description="Center X (circle/ellipse)")
    cy: Optional[float] = Field(default=None, description="Center Y (circle/ellipse)")
    r: Optional[float] = Field(default=None, description="Radius (circle)")
    # Rect attributes
    x: Optional[float] = Field(default=None, description="X position (rect)")
    y: Optional[float] = Field(default=None, description="Y position (rect)")
    width: Optional[float] = Field(default=None, description="Width (rect)")
    height: Optional[float] = Field(default=None, description="Height (rect)")
    rx: Optional[float] = Field(default=None, description="Corner radius (rect) / X-radius (ellipse)")
    # Ellipse
    ry: Optional[float] = Field(default=None, description="Y-radius (ellipse)")
    # Line attributes
    x1: Optional[float] = Field(default=None, description="Start X (line)")
    y1: Optional[float] = Field(default=None, description="Start Y (line)")
    x2: Optional[float] = Field(default=None, description="End X (line)")
    y2: Optional[float] = Field(default=None, description="End Y (line)")
    # Path
    d: Optional[str] = Field(default=None, description="SVG path data string (for complex shapes)")
    # Common styling
    fill: Optional[str] = Field(default="#3b82f6", description="Fill color (hex)")
    stroke: Optional[str] = Field(default="#60a5fa", description="Stroke color (hex)")
    stroke_width: Optional[float] = Field(default=2, description="Stroke width")
    # Animation flag
    animate: Optional[bool] = Field(default=False, description="True if this element should animate during action phase")
    label: Optional[str] = Field(default=None, description="Text label near the element")

class TemplatePayload(BaseModel):
    # Used for 'title_slide'
    title: Optional[str] = Field(default=None, description="Centered big title")
    subtitle: Optional[str] = Field(default=None, description="Topic subtitle summary")
    icon_name: Optional[str] = Field(default=None, description="Lucide icon name")

    # Used for 'concept_diagram'
    left_title: Optional[str] = Field(default=None, description="Header of bullet points")
    left_bullets: Optional[List[str]] = Field(default=None, description="1-3 bullet definitions")
    central_node: Optional[str] = Field(default=None, description="Central Entity block label")
    leaf_nodes: Optional[List[str]] = Field(default=None, description="Connected Attribute node labels")

    # Used for 'horizontal_timeline'
    timeline_title: Optional[str] = Field(default=None, description="Dashed line process title")
    stages: Optional[List[TimelineStage]] = Field(default=None, description="Timeline stages")

    # Used for 'column_comparison'
    left_column: Optional[ColumnData] = Field(default=None, description="Left comparison card")
    right_column: Optional[ColumnData] = Field(default=None, description="Right comparison card")

    # Used for 'database_grid'
    table_title: Optional[str] = Field(default=None, description="Title of database table")
    headers: Optional[List[str]] = Field(default=None, description="Table column header names")
    rows: Optional[List[List[str]]] = Field(default=None, description="Rows cell values")
    highlight_row_idx: Optional[int] = Field(default=-1, description="Row index to highlight, -1 if none")
    highlight_col_idx: Optional[int] = Field(default=-1, description="Col index to highlight, -1 if none")

    # Used for 'illustrated_scene' — DYNAMIC SVG generation
    svg_elements: Optional[List[SvgElement]] = Field(default=None, description="Array of SVG elements to render dynamically")
    animation_action: Optional[Literal["rise", "fall", "spin", "scale_up", "slide_left", "slide_right", "none"]] = Field(default="none", description="Animation action for illustrated elements")
    canvas_color: Optional[str] = Field(default=None, description="Background accent color for the SVG canvas (hex)")

class Scene(BaseModel):
    scene_no: int = Field(description="Sequential scene number starting at 1")
    teacher_script: str = Field(description="Narrator audio script (2-3 short sentences, matching class level)")
    template_id: Literal["title_slide", "concept_diagram", "horizontal_timeline", "column_comparison", "database_grid", "illustrated_scene"] = Field(
        description="Choose template matching scene visual data."
    )
    template_data: TemplatePayload = Field(description="Payload details matching the selected template_id")

class Storyboard(BaseModel):
    lesson_title: str = Field(description="Main title of the lesson")
    theme: Literal["indigo", "gold", "emerald", "rose"] = Field(description="Color theme matching subject matter")
    scenes: List[Scene] = Field(description="Chronological scene storyboard list")

# 3. Helper function to retrieve textbook context with query reformulation
def retrieve_textbook_context(book_uuid: str, query: str, class_name: str, subject: str) -> str:
    print("\n[*] Initializing Search Engine (Qdrant)...")
    qdrant.initialize()

    # Step A: Perform LLM query reformulation matching standard chat backend
    print(f"[*] Fetching chapters for Class {class_name} - {subject}...")
    chapters = local_chap_service.get_chapters(class_name, subject)
    
    print(f"[*] Reformulating student query using LLM...")
    reformulated_query = query
    try:
        reform = reformulate_with_llm(query, class_name, subject, chapters)
        reformulated_query = reform.get("reformulated_query", query)
        print(f" [OK] Original Query:    \"{query}\"")
        print(f" [OK] Reformulated Query:\"{reformulated_query}\"")
    except Exception as e:
        print(f" [WARNING] Reformulation failed: {e}. Using raw query.")

    # Step B: Retrieve context using reformulated query
    print(f"[*] Running hybrid search for query: '{reformulated_query}'...")
    context = ""
    try:
        results, _, _ = qdrant.hybrid_search(
            book_uuid=book_uuid,
            query=reformulated_query,
            keywords=[],
            conceptual_score=0.5,
            metadata_filters=None
        )
        if results:
            context = "\n\n---\n\n".join([doc["text"] for score, doc in results[:5]])
            print(f"[OK] Retrieved {len(results)} context blocks from database.")
        else:
            print("[WARNING] No textbook chunks found. Using reformulated query only as context.")
    except Exception as e:
        print(f"[ERROR] Search failed: {e}. Proceeding with empty context.")
    
    return context

# 4. Helper function to call Sarvam AI Text-to-Speech API
def generate_sarvam_audio(text: str, output_path: Path, speaker: str, api_key: str) -> bool:
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "target_language_code": "en-IN",
        "speaker": speaker,
        "model": "bulbul:v3",
        "enable_preprocessing": True
    }
    
    try:
        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                audios = data.get("audios", [])
                if audios:
                    audio_b64 = audios[0]
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Ensure destination directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    return True
                else:
                    print(f"  [ERROR] Sarvam returned empty audios array: {data}")
            else:
                print(f"  [ERROR] Sarvam TTS responded with {response.status_code}: {response.text}")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch Sarvam audio: {e}")
    
    return False

# 5. Main pipeline execution
def main():
    print("==================================================")
    print("    VISUAL LEARNING TEMPLATE-DRIVEN RENDERING     ")
    print("==================================================")
    
    # Check APIs
    google_key = os.getenv("GOOGLE_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    
    if not google_key:
        print("[ERROR] GOOGLE_API_KEY is missing from .env.")
        sys.exit(1)
    if not sarvam_key:
        print("[ERROR] SARVAM_API_KEY is missing from .env.")
        sys.exit(1)
        
    # Select Storyboard Generation Mode
    print("\nSelect Storyboard Generation Mode:")
    print("[1] System Mode (Retrieve context from textbook database)")
    print("[2] General Mode (Skip retrieval, generate using general knowledge)")
    
    while True:
        mode = input("\nEnter selection (1 or 2): ").strip()
        if mode in ('1', '2'):
            break
        print("Invalid selection. Try again.")

    if mode == '1':
        # Get available books
        books = local_chap_service.get_books()
        if not books:
            print("[ERROR] No books found in chapters_cache.json.")
            sys.exit(1)
            
        print("\nAvailable Books:")
        for idx, book in enumerate(books):
            print(f"[{idx + 1}] Class: {book.get('class_name')} | Subject: {book.get('subject')} | File: {book.get('filename')}")
            
        while True:
            choice = input(f"\nSelect a book number (1-{len(books)}): ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(books):
                    selected_book = books[choice_idx]
                    break
            except ValueError:
                pass
            print("Invalid selection. Try again.")

        book_uuid = selected_book.get("id")
        class_name = selected_book.get("class_name")
        subject = selected_book.get("subject")
        
        query = input("\nEnter your storyboard topic query (e.g. 'explain paper making steps'): ").strip()
        while not query:
            query = input("Query cannot be empty: ").strip()

        # Step A: Retrieve Context (with reformulation)
        context = retrieve_textbook_context(book_uuid, query, class_name, subject)
    else:
        class_name = "8"
        subject = "Science / Mathematics / General"
        query = input("\nEnter your general topic query (e.g. 'explain photosynthesis' or 'how gravity works'): ").strip()
        while not query:
            query = input("Query cannot be empty: ").strip()
        
        print("\n[*] Skipping retrieval. Generating storyboard from general knowledge...")
        context = "No textbook context provided. Generate the educational storyboard using your own broad general knowledge."

    
    # Step B: Call Gemini with strict Pydantic Response Schema
    from google import genai
    from google.genai import types
    
    print("\n[*] Initializing Gemini Client...")
    client = genai.Client(api_key=google_key)
    
    system_prompt = f"""You are CHADUVU-GURU, an intelligent educational video storyboard designer for Class {class_name} ({subject}).
Your task is to explain the user's question using textbook context by organizing it into a step-by-step visual storyboard.
You must strictly format your output to conform to the requested JSON response schema.

For each scene, choose the single best template_id:
- Choose 'title_slide' for the introductory scene.
- Choose 'concept_diagram' to explain a key term, structure, or concept that has properties, parts, or attributes.
- Choose 'horizontal_timeline' if the scene explains a chronological sequence of steps, events, or logical order.
- Choose 'column_comparison' when contrasting two ideas or showing differences/pros-cons.
- Choose 'database_grid' when explaining tables, structured fields, rows, columns, or values (like NULL).
- Choose 'illustrated_scene' to dynamically show and animate vector drawings of physical objects, structures, processes, organs, machines, or systems (using circles, rects, lines, paths, etc.). Use this template when a visual diagram helps explain the concept.

Rules:
1. MANDATORY ILLUSTRATION RULE: Whenever a scene visualizes physical objects, mechanical parts, biology structures/organs, nature processes (like water cycle, photosynthesis, digestive system organs), or dynamic physical transitions, you MUST choose the 'illustrated_scene' template.
2. SVG CANVAS & BOUNDS: When using 'illustrated_scene', you must provide a list of 'svg_elements' to be drawn on a 500x400 canvas.
   - All coordinate values (cx, cy, x, y, x1, x2, y1, y2) must fit inside the 500x400 viewport.
   - Keep shapes centered and organized. Do not overlap shapes messily unless intended (e.g. foliage on a tree trunk).
   - Use simple primitives: 'circle' for sun/nodes/dots/organs, 'rect' for boxes/ground/trucks, 'ellipse' for horizontal clouds/leaves, 'line' for rain/rays/connections, 'path' for custom drawings.
3. COLOR PALETTE: Choose vibrant, harmonious colors for your fills and strokes. Use valid hex codes (e.g. '#ef4444' for stomach/heart, '#3b82f6' for water, '#eab308' for sun, '#22c55e' for leaves/plants, '#d97706' for wood).
4. ANIMATION ACTION & CHOREOGRAPHY: Set 'animation_action' ('rise', 'fall', 'spin', 'scale_up', 'slide_left', 'slide_right', or 'none').
   - Identify the elements that should move during this action, and set `animate=True` on them. All other elements will remain static.
   - For example: if action is 'rise', set `animate=True` on the steam/vapor elements to show evaporation. If action is 'fall', set `animate=True` on the raindrops or falling food. If action is 'spin', set `animate=True` on gears or wheels.
5. Provide labels: Use the 'label' field on key SVG elements to render text labels next to them (e.g. "Stomach", "Esophagus", "Sun", "Water", "Vapor").
6. The 'teacher_script' must be a concise, easy-to-read narration script (maximum 2-3 short sentences).
7. Clean Lucide icon names for 'icon_name' keys (e.g. 'book-open', 'globe', 'settings', 'database', 'shield', 'bell').
"""


    user_content = f"""Student Query: "{query}"

Textbook Context Chunks:
---
{context}
---

Generate the storyboard now.
"""
    
    print("[*] Generating template-structured storyboard using Gemini...")
    model_name = qdrant.generation_model_name or "gemini-2.5-flash"
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Storyboard,
                temperature=0.2
            )
        )
        storyboard_data = json.loads(response.text)
        print("[OK] Storyboard successfully generated.")
    except Exception as e:
        print(f"[ERROR] Gemini storyboard generation failed: {e}")
        sys.exit(1)
        
    # Step C: Log Details of Selected Templates
    print("\n==================================================")
    print("            GENERATED STORYBOARD LOGS             ")
    print("==================================================")
    print(f"Lesson Title: {storyboard_data.get('lesson_title')}")
    print(f"Theme: {storyboard_data.get('theme')}")
    print(f"Total Scenes: {len(storyboard_data.get('scenes', []))}")
    print("--------------------------------------------------")
    
    scenes = storyboard_data.get("scenes", [])
    for scene in scenes:
        s_no = scene.get("scene_no")
        t_id = scene.get("template_id")
        script = scene.get("teacher_script")
        print(f"Scene #{s_no} | Selected Template: '{t_id}'")
        print(f"  Narration Script: \"{script}\"")
        # Print template specific details
        payload = scene.get("template_data", {})
        if t_id == "title_slide":
            print(f"  Details: Title='{payload.get('title')}' | Icon='{payload.get('icon_name')}'")
        elif t_id == "concept_diagram":
            print(f"  Details: Central Node='{payload.get('central_node')}' | Attributes={payload.get('leaf_nodes')}")
        elif t_id == "horizontal_timeline":
            stages_summary = [f"{s.get('step_no')}:{s.get('label')}" for s in payload.get("stages", [])]
            print(f"  Details: Title='{payload.get('timeline_title')}' | Stages={stages_summary}")
        elif t_id == "column_comparison":
            print(f"  Details: LeftHeader='{payload.get('left_column', {}).get('header')}' vs RightHeader='{payload.get('right_column', {}).get('header')}'")
        elif t_id == "database_grid":
            print(f"  Details: Grid='{payload.get('table_title')}' | Rows={len(payload.get('rows', []))} | Highlight=Row {payload.get('highlight_row_idx')}, Col {payload.get('highlight_col_idx')}")
        elif t_id == "illustrated_scene":
            print(f"  Details: SVG Elements count={len(payload.get('svg_elements', []) or [])} | Action='{payload.get('animation_action')}'")
        print("--------------------------------------------------")

    # Step D: Choose Speaker for Sarvam AI bulbul:v3 Model
    print("\n==================================================")
    print("          SARVAM TTS SPEAKER SELECTION            ")
    print("==================================================")
    print("Available Speakers: ritu, anushka, abhilash, manisha, vidya, rahul, rohan, kavya, vijay")
    speaker = input("Select a speaker (Press Enter for default 'ritu'): ").strip().lower()
    if not speaker:
        speaker = "ritu"

    print(f"\n[*] Starting audio synthesis using speaker '{speaker}'...")
    test_lesson_id = "test_remotion_lesson"
    relative_upload_dir = Path("uploads") / "visual_lessons" / test_lesson_id
    full_upload_dir = PROJECT_ROOT / "remotion_test_app" / "public" / relative_upload_dir
    
    # Clear directory to keep clean
    if full_upload_dir.exists():
        import shutil
        shutil.rmtree(full_upload_dir)
    full_upload_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, scene in enumerate(scenes):
        s_no = scene.get("scene_no")
        script = scene.get("teacher_script")
        
        audio_filename = f"scene_{s_no}.wav"
        output_path = full_upload_dir / audio_filename
        
        print(f" [*] Synthesizing audio for Scene #{s_no}...")
        success = generate_sarvam_audio(script, output_path, speaker, sarvam_key)
        if success:
            # Set audio URL relative to public folder in Remotion
            scene["audio_url"] = f"/{relative_upload_dir.as_posix()}/{audio_filename}"
            print(f"  [OK] Saved audio to {output_path.name}")
        else:
            print(f"  [WARNING] Failed to generate audio. Remotion will fallback to text-length timing estimation.")
            scene["audio_url"] = ""

    # Step E: Save Final Storyboard JSON for Remotion
    storyboard_data["scenes"] = scenes
    storyboard_data["lesson_id"] = test_lesson_id
    
    props_json_path = full_upload_dir / "lesson.json"
    with open(props_json_path, "w", encoding="utf-8") as f:
        json.dump(storyboard_data, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved final storyboard properties to: {props_json_path}")

    # Step F: Run Remotion Render or Preview Subprocess
    print("\n==================================================")
    print("             REMOTION EXECUTION CHOICE            ")
    print("==================================================")
    print("[1] View dynamically in Local Web Player (Preview)")
    print("[2] Render/Compile to output_test.mp4 file")
    
    choice = input("\nSelect rendering action (1 or 2): ").strip()
    while choice not in ('1', '2'):
        choice = input("Invalid choice. Select 1 or 2: ").strip()

    relative_props_path = f"public/{relative_upload_dir.as_posix()}/lesson.json"
    cmd = 'npx'
    
    if choice == '1':
        print("\n[*] Launching Remotion Player preview. Press Ctrl+C in this terminal to stop.")
        args = ['remotion', 'preview', 'src/index.ts', f'--props={relative_props_path}']
    else:
        out_name = f"output_{test_lesson_id}.mp4"
        print(f"\n[*] Rendering MP4 video to: remotion_test_app/{out_name}...")
        args = ['remotion', 'render', 'src/index.ts', 'StoryboardVideo', out_name, f'--props={relative_props_path}']
        
    try:
        subprocess.run(
            [cmd] + args,
            shell=True,
            cwd=str(PROJECT_ROOT / "remotion_test_app")
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to execute Remotion process: {e}")

if __name__ == "__main__":
    main()

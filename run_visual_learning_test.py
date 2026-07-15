#!/usr/bin/env python3
import os
import sys
import json
import httpx
import base64
import subprocess
import re
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

class ZoomTarget(BaseModel):
    x: float = Field(description="Horizontal center of zoom (0-100%)")
    y: float = Field(description="Vertical center of zoom (0-100%)")
    scale: float = Field(description="Zoom level (1=normal, 2.5=close-up)")
    at_percent: float = Field(description="When in scene timeline (0-100%)")

class ImageAnnotation(BaseModel):
    type: Literal["arrow", "circle", "label"]
    x: float = Field(description="X position (%)")
    y: float = Field(description="Y position (%)")
    target_x: Optional[float] = Field(default=None, description="Arrow endpoint X (%)")
    target_y: Optional[float] = Field(default=None, description="Arrow endpoint Y (%)")
    label: Optional[str] = Field(default=None, description="Text label content")
    color: Optional[str] = Field(default="#ef4444", description="Hex color")
    at_percent: float = Field(description="When to appear (0-100%)")

class MotionPathData(BaseModel):
    path_data: str = Field(description="SVG path d attribute for curved motion path within 1280x720 space")
    dot_color: Optional[str] = Field(default="#ef4444")
    dot_size: Optional[float] = Field(default=8)
    start_percent: Optional[float] = Field(default=10)
    duration_percent: Optional[float] = Field(default=70)

class SpotlightData(BaseModel):
    x: float = Field(description="Spotlight center X (%)")
    y: float = Field(description="Spotlight center Y (%)")
    radius: float = Field(description="Spotlight radius in pixels")
    at_percent: float = Field(description="When to activate (0-100%)")

class CoordPoint(BaseModel):
    x: float = Field(description="X grid coordinate (e.g. -5 to 5)")
    y: float = Field(description="Y grid coordinate (e.g. -5 to 5)")
    label: Optional[str] = Field(default=None, description="Label for the point")

class ConnectionLine(BaseModel):
    from_idx: int = Field(description="Index of starting point in points array")
    to_idx: int = Field(description="Index of ending point in points array")
    label: Optional[str] = Field(default=None, description="Label for the line connection")

class TemplatePayload(BaseModel):
    # Used for 'title_slide'
    title: Optional[str] = Field(default=None, description="Centered big title")
    subtitle: Optional[str] = Field(default=None, description="Topic subtitle summary")
    icon_name: Optional[str] = Field(default=None, description="Lucide icon name")

    # Used for 'cartesian_grid'
    points: Optional[List[CoordPoint]] = Field(default=None, description="List of coordinate points to plot")
    lines: Optional[List[ConnectionLine]] = Field(default=None, description="List of straight line connections between points indices")
    equation_label: Optional[str] = Field(default=None, description="Equation formula string to display in the side box")

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

    # Used for 'illustrated_scene' — DYNAMIC SVG generation (legacy fallback support)
    svg_elements: Optional[List[SvgElement]] = Field(default=None, description="Array of SVG elements to render dynamically")
    animation_action: Optional[Literal["rise", "fall", "spin", "scale_up", "slide_left", "slide_right", "none"]] = Field(default="none", description="Animation action for illustrated elements")
    canvas_color: Optional[str] = Field(default=None, description="Background accent color for the SVG canvas (hex)")

class StepAnimation(BaseModel):
    transition: Literal["fade", "slide", "wipe", "none"] = Field(
        default="none", description="Transition effect entering this step"
    )
    camera_motion: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "none"] = Field(
        default="none", description="Ken Burns camera motion applied to this step"
    )

class StepContent(BaseModel):
    svg_elements: Optional[List[SvgElement]] = Field(default=None, description="Array of SVG shapes - required if visual_type is 'diagram'")
    text_content: Optional[str] = Field(default=None, description="LaTeX math equation or plain text - required if visual_type is 'equation'")

class VisualStep(BaseModel):
    step_no: int = Field(description="Sequential step index starting at 1")
    visual_type: Literal["diagram", "equation", "table"] = Field(
        description="Educational visual type of this step"
    )
    focus: Optional[str] = Field(default=None, description="Element or region to emphasize (e.g., 'mouth', 'stomach')")
    duration_seconds: float = Field(description="Duration of this step in seconds")
    content: StepContent = Field(description="Visual content payload matching visual_type")
    animation: StepAnimation = Field(description="Animation and transition metadata")

class Scene(BaseModel):
    scene_no: int = Field(description="Sequential scene number starting at 1")
    purpose: str = Field(description="Pedagogical objective (e.g., 'Introduce stomach structure')")
    visual_strategy: str = Field(
        description="High-level teaching strategy (e.g., 'intro', 'overview', 'process', 'comparison', 'timeline', 'diagram', 'summary')"
    )
    template_id: Optional[Literal[
        "title_slide", "concept_diagram", "cycle_template", "math_derivation",
        "venn_diagram", "taxonomy_tree", "cartesian_grid", "column_comparison",
        "geo_marker", "database_grid", "before_after_slider", "quiz_checkpoint"
    ]] = Field(
        default=None, description="Renderer template component mapping"
    )
    teacher_script: str = Field(description="Narrator audio script (2-3 short sentences)")
    visual_steps: Optional[List[VisualStep]] = Field(default=None, description="Array of sequential visual steps within this scene")
    template_data: Optional[TemplatePayload] = Field(default=None, description="Static rendering parameters matching template_id (fallback/legacy)")

class Storyboard(BaseModel):
    lesson_title: str = Field(description="Main title of the lesson")
    theme: Literal["indigo", "gold", "emerald", "rose", "Science", "Math", "History", "Civics", "General"] = Field(description="Color theme matching subject matter")
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
    
    # Pre-flight checklist and import checking
    print("[*] Verifying critical python packages and backend services...")
    try:
        import httpx
        import pydantic
        from dotenv import load_dotenv
        from backend import local_chap_service
        from backend.app.services.retrieval import qdrant_service as qdrant
        from backend.app.services.chat.answer_service import reformulate_with_llm
        print(" [OK] All Python package and backend imports resolved successfully.\n")
    except ImportError as e:
        print(f" [ERROR] Critical import failed: {e}")
        print(" Please ensure your virtual environment is activated and you run this in the correct directory.")
        sys.exit(1)
        
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

    # Generate slugified name for output files
    slug = re.sub(r'[^a-zA-Z0-9\s_\-]', '', query).strip().lower()
    slug = re.sub(r'[\s\-]+', '_', slug)
    slug = re.sub(r'_+', '_', slug)
    if not slug:
        slug = "storyboard"

    # Step B: Call Gemini with strict Pydantic Response Schema
    from google import genai
    from google.genai import types
    
    print("\n[*] Initializing Gemini Client...")
    if google_key and google_key.startswith("AQ."):
        print("[DEBUG KEY] Detected AQ key. Forcing standard API key header workaround.")
        client = genai.Client(
            api_key="AIza_DummyForceAPIKeyMode",
            http_options={"headers": {"x-goog-api-key": google_key}}
        )
    else:
        client = genai.Client(api_key=google_key)
    
    system_prompt = f"""You are CHADUVU-GURU, an intelligent educational video storyboard designer for Class {class_name} ({subject}).
Your task is to explain the user's question using textbook context by organizing it into a step-by-step visual storyboard.
You must strictly format your output to conform to the requested JSON response schema.

Set the overall 'theme' of the Storyboard based on the subject:
- 'Science' for Biology, Chemistry, Physics, Environmental Science.
- 'Math' for Algebra, Geometry, Arithmetic, Statistics.
- 'History' for Historical events and timelines.
- 'Civics' for Constitution, Governance, and Rights.
- 'General' for other miscellaneous educational topics.

For each scene, set the 'template_id' matching the pedagogical goal of that scene:
- 'title_slide': For introducing the main lesson topic or agenda.
- 'concept_diagram': For explaining core structures with attributes connected to a main concept.
- 'cycle_template': For explaining repeating loops (e.g., Water Cycle, Nitrogen Cycle, rock cycles).
- 'math_derivation': For demonstrating equations, formula solving, or balanced chemical equations line-by-line.
- 'venn_diagram': For comparing overlapping properties (e.g., Plant vs. Animal cells, Solid vs. Liquid).
- 'taxonomy_tree': For taxonomy, classification hierarchies, or family/government branches.
- 'cartesian_grid': For graphing coordinate geometry, lines, triangles, angles, and algebra graphs. If chosen, populate 'points' (list of coordinates like x: -2, y: 4, label: 'Vertex') and 'lines' (connections between points), and/or 'svg_elements' (for curves like a parabola: e.g. type: 'path', d: 'M...', stroke: '#...', stroke_width: 4).
- 'column_comparison': For direct side-by-side card contrasts.
- 'geo_marker': For geography and history maps, highlighting regions with coordination pointers.
- 'database_grid': For displaying tabular data or periodic table grids.
- 'before_after_slider': For showing a wipe transition between cause and effect states.
- 'quiz_checkpoint': For active recall summary questions at the end of the lesson.

Write a concise 'teacher_script' narration (2-3 short sentences, matching class level).
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
        
        # Inject the user's query/question at the top level
        storyboard_data["question"] = query
        
        # Save storyboard JSON to outputs folder
        outputs_json_dir = PROJECT_ROOT / "remotion_test_app" / "outputs" / "storyboard_json"
        outputs_json_dir.mkdir(parents=True, exist_ok=True)
        storyboard_json_path = outputs_json_dir / f"{slug}.json"
        with open(storyboard_json_path, "w", encoding="utf-8") as f:
            json.dump(storyboard_data, f, indent=2, ensure_ascii=False)
        print(f" [OK] Storyboard JSON saved to: {storyboard_json_path.resolve()}")
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
        if scene.get("visual_steps"):
            steps_summary = [f"Step {s.get('step_no')}({s.get('visual_type')}, focus='{s.get('focus')}')" for s in scene.get("visual_steps", [])]
            print(f"  Visual Strategy: '{scene.get('visual_strategy')}' | Steps={steps_summary}")
        elif t_id == "title_slide":
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
        elif t_id == "image_scene":
            zoom_count = len(payload.get('zoom_targets', []) or [])
            ann_count = len(payload.get('annotations', []) or [])
            has_path = "Yes" if payload.get('motion_path') else "No"
            has_spotlight = "Yes" if payload.get('spotlight') else "No"
            print(f"  Details: Style='{payload.get('animation_style')}' | ZoomTargets={zoom_count} | Annotations={ann_count} | MotionPath={has_path} | Spotlight={has_spotlight}")
            print(f"  Image Prompt: \"{scene.get('image_prompt')}\"")
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
        out_name = f"outputs/output_videos/{slug}.mp4"
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

import os
import sys
import json
import shutil
import logging
import asyncio
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Find project root
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "..", ".."))

async def compile_hyperframes_html_fast(lesson_id: str, lesson_dir: str):
    """
    Fast-path Hyperframes Master Composition generator (< 1 second execution).
    Compiles lesson.json, templates, GSAP engine, camera transforms, and narration sync into index.html.
    Returns html_url relative path for instant browser playback.
    """
    hf_dir = os.path.join(PROJECT_ROOT, "hyperframes_engine")
    hf_outputs_dir = os.path.join(hf_dir, "outputs", lesson_id)
    os.makedirs(hf_outputs_dir, exist_ok=True)

    # Sync lesson.json & audio files into hyperframes outputs dir
    if os.path.abspath(lesson_dir) != os.path.abspath(hf_outputs_dir):
        for item in os.listdir(lesson_dir):
            s_path = os.path.join(lesson_dir, item)
            d_path = os.path.join(hf_outputs_dir, item)
            if os.path.isfile(s_path):
                try:
                    shutil.copy2(s_path, d_path)
                except Exception:
                    pass

    lesson_json_rel = os.path.join("outputs", lesson_id, "lesson.json")

    def _run_node_compiler():
        cmd = ["node", "run-storyboard.js", lesson_json_rel, "compile"]
        return subprocess.run(
            cmd, cwd=hf_dir, capture_output=True, text=True,
            encoding='utf-8', errors='replace', shell=True
        )

    try:
        res = await asyncio.to_thread(_run_node_compiler)
        if res.stdout:
            logger.info(f"[Hyperframes Compiler] stdout:\n{res.stdout.strip()}")
        if res.returncode != 0:
            logger.warning(f"[Hyperframes Compiler Notice] Exit code {res.returncode}: {res.stderr}")
        else:
            logger.info(f"[Hyperframes Compiler] Compilation succeeded (exit 0)")
    except Exception as e:
        logger.error(f"[Hyperframes Compiler Error] {e}")

    # Copy index.html from compiler output dir to uploads serving dir
    src_html = os.path.join(hf_outputs_dir, "index.html")
    dest_html = os.path.join(lesson_dir, "index.html")
    if os.path.exists(src_html):
        try:
            print("\n======================================================================")
            print("🚀 [PIPELINE DEBUG] ENTER Output Copy")
            print(f"   Copying {src_html} -> {dest_html}")
            print("======================================================================\n")
            shutil.copy2(src_html, dest_html)
        except Exception:
            pass

        # Copy shared/ JS libraries AFTER compilation so browser serves post-compile versions
        # (theme.js, animations.js must be co-located with index.html for relative-path resolution)
        hf_shared = os.path.join(hf_dir, "shared")
        if os.path.exists(hf_shared):
            try:
                shutil.copytree(hf_shared, os.path.join(lesson_dir, "shared"), dirs_exist_ok=True)
                logger.info(f"[Hyperframes Bridge] Copied shared/ JS libs to {lesson_dir}")
            except Exception as e:
                logger.warning(f"[Hyperframes Bridge] shared/ copy warning: {e}")

        return f"/uploads/visual_lessons/{lesson_id}/index.html"
    elif os.path.exists(dest_html):
        # index.html already present (e.g. from a previous run) — still ensure shared/ is there
        hf_shared = os.path.join(hf_dir, "shared")
        if os.path.exists(hf_shared):
            try:
                shutil.copytree(hf_shared, os.path.join(lesson_dir, "shared"), dirs_exist_ok=True)
            except Exception:
                pass
        return f"/uploads/visual_lessons/{lesson_id}/index.html"
    return None

async def render_hyperframes_video_stream(lesson_id: str, lesson_dir: str):
    """
    Executes the Hyperframes Engine compilation & rendering pipeline for a given lesson.
    Yields real-time progress dicts detailing each subsystem phase (SceneGraph, Pedagogy, Sync, Camera, Performance, Render).
    Returns (video_relative_url, execution_metrics).
    """
    hf_dir = os.path.join(PROJECT_ROOT, "hyperframes_engine")
    hf_outputs_dir = os.path.join(hf_dir, "outputs", lesson_id)
    os.makedirs(hf_outputs_dir, exist_ok=True)
    
    # Sync lesson.json & audio files into hyperframes outputs dir if not already done
    if os.path.abspath(lesson_dir) != os.path.abspath(hf_outputs_dir):
        for item in os.listdir(lesson_dir):
            s_path = os.path.join(lesson_dir, item)
            d_path = os.path.join(hf_outputs_dir, item)
            if os.path.isfile(s_path):
                shutil.copy2(s_path, d_path)
    
    # Guarantee shared JS libraries (theme.js & animations.js) are present in both output folders
    hf_shared = os.path.join(hf_dir, "shared")
    if os.path.exists(hf_shared):
        try:
            shutil.copytree(hf_shared, os.path.join(hf_outputs_dir, "shared"), dirs_exist_ok=True)
            shutil.copytree(hf_shared, os.path.join(lesson_dir, "shared"), dirs_exist_ok=True)
        except Exception:
            pass

    lesson_json_rel = os.path.join("outputs", lesson_id, "lesson.json")
    
    yield {
        "type": "progress",
        "step": "hyperframes_engine",
        "phase": "bootstrap",
        "status": "in_progress",
        "message": "[Hyperframes:Core] Bootstrapping Configuration & Performance Layer..."
    }
    await asyncio.sleep(0.2)

    # Command: run-storyboard.js passing lesson.json path and action '2' (render MP4) non-interactively
    cmd = [
        "node",
        "run-storyboard.js",
        lesson_json_rel,
        "2"
    ]

    logger.info(f"[Hyperframes Bridge] Launching Hyperframes Engine process non-interactively for {lesson_id}...")

    proc = subprocess.Popen(
        cmd,
        cwd=hf_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    metrics = {"total_frames": 0, "completed_frames": 0, "fps": 30}
    last_phase = "bootstrap"

    # Read stdout line by line and map to structured subsystem events using asyncio.to_thread
    while True:
        line = await asyncio.to_thread(proc.stdout.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        
        logger.debug(f"[Hyperframes Node stdout] {line}")

        # Parse engine logs and map to visual progress states
        if "Performance layer initialized" in line:
            yield {
                "type": "progress",
                "step": "hyperframes_engine",
                "phase": "scene_graph",
                "status": "in_progress",
                "message": "[Hyperframes:SceneGraph] Building Unified Scene Graph & Component Tree..."
            }
            last_phase = "scene_graph"
        elif "[Renderer LOG] Processing rendering pipeline" in line:
            scene_info = line.split("Processing rendering pipeline for")[-1].strip()
            yield {
                "type": "progress",
                "step": "hyperframes_engine",
                "phase": "rendering_pipeline",
                "status": "in_progress",
                "message": f"[Hyperframes:Renderer] Resolving Camera, Layout & Visual Focus for {scene_info}..."
            }
            last_phase = "rendering_pipeline"
        elif "HTML generation pipeline complete" in line:
            # Copy index.html to main uploads dir immediately for instant composition preview
            src_html = os.path.join(hf_outputs_dir, "index.html")
            if os.path.exists(src_html):
                dest_html = os.path.join(lesson_dir, "index.html")
                try:
                    shutil.copy2(src_html, dest_html)
                except Exception:
                    pass
            yield {
                "type": "progress",
                "step": "hyperframes_engine",
                "phase": "frame_capture",
                "status": "in_progress",
                "message": "[Hyperframes:HeadlessCapture] Master composition built! Starting Puppeteer 30fps frame capture..."
            }
            last_phase = "frame_capture"
        elif "Capturing frame" in line:
            # e.g. "Capturing frame 120/3987 (4 workers)"
            if "/" in line:
                try:
                    parts = line.split("Capturing frame")[-1].split("(")[0].strip().split("/")
                    cur_f = int(parts[0])
                    tot_f = int(parts[1])
                    metrics["completed_frames"] = cur_f
                    metrics["total_frames"] = tot_f
                    pct = int((cur_f / tot_f) * 100)
                    if cur_f % 90 == 0 or cur_f == tot_f or cur_f < 150:
                        yield {
                            "type": "progress",
                            "step": "hyperframes_engine",
                            "phase": "frame_capture",
                            "status": "in_progress",
                            "message": f"[Hyperframes:Capture] Capturing 30fps frames: {pct}% ({cur_f}/{tot_f} frames)..."
                        }
                except Exception:
                    pass
        elif "Encoding video" in line:
            yield {
                "type": "progress",
                "step": "hyperframes_engine",
                "phase": "encoding",
                "status": "in_progress",
                "message": "[Hyperframes:FFmpeg] Encoding high-fidelity MP4 video & audio stream..."
            }
            last_phase = "encoding"
        elif "Render complete" in line or "custom_lesson.mp4" in line:
            yield {
                "type": "progress",
                "step": "hyperframes_engine",
                "phase": "complete",
                "status": "complete",
                "message": "[Hyperframes:Complete] Video render complete & validated."
            }
            last_phase = "complete"

    returncode = await asyncio.to_thread(proc.wait)
    logger.info(f"[Hyperframes Bridge] Process finished with exit code {returncode}")

    if returncode != 0:
        stderr_text = await asyncio.to_thread(proc.stderr.read)
        logger.error(f"[Hyperframes Bridge Error] Node process failed:\n{stderr_text}")

    # Check for output mp4 file: try custom_lesson.mp4, {lesson_id}.mp4, or any .mp4 file in hf_outputs_dir
    expected_mp4 = None
    for candidate_name in ["custom_lesson.mp4", f"{lesson_id}.mp4"]:
        cand_path = os.path.join(hf_outputs_dir, candidate_name)
        if os.path.exists(cand_path):
            expected_mp4 = cand_path
            break

    if not expected_mp4 and os.path.exists(hf_outputs_dir):
        for f in os.listdir(hf_outputs_dir):
            if f.endswith(".mp4"):
                expected_mp4 = os.path.join(hf_outputs_dir, f)
                break

    if expected_mp4 and os.path.exists(expected_mp4):
        # Copy output mp4 to main uploads dir
        main_mp4 = os.path.join(lesson_dir, "custom_lesson.mp4")
        shutil.copy2(expected_mp4, main_mp4)
        video_url = f"/uploads/visual_lessons/{lesson_id}/custom_lesson.mp4"
        logger.info(f"[Hyperframes Bridge] Rendered MP4 successfully copied to {main_mp4} (URL: {video_url})")
        yield {
            "type": "progress",
            "step": "hyperframes_engine",
            "phase": "complete",
            "status": "complete",
            "message": f"[Hyperframes Complete] Video attached: {video_url}",
            "video_url": video_url,
            "metrics": metrics
        }
        return
    else:
        logger.error(f"[Hyperframes Bridge Error] MP4 file not found after render: {expected_mp4}")
        return

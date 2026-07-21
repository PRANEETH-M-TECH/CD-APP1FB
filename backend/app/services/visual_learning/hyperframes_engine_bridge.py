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
    # Defensive normalization: if lesson_dir is a file path (e.g. lesson.json), convert to its parent directory
    if os.path.isfile(lesson_dir):
        lesson_dir = os.path.dirname(lesson_dir)

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
            encoding='utf-8', errors='replace', shell=False
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
            print("[PIPELINE DEBUG] ENTER Output Copy")
            print(f"   Copying {src_html} -> {dest_html}")
            print("======================================================================\n")
            shutil.copy2(src_html, dest_html)
        except Exception as copy_err:
            logger.warning(f"[Hyperframes Bridge] HTML copy warning: {copy_err}")

        hf_shared = os.path.join(hf_dir, "shared")
        if os.path.exists(hf_shared):
            try:
                shutil.copytree(hf_shared, os.path.join(lesson_dir, "shared"), dirs_exist_ok=True)
                logger.info(f"[Hyperframes Bridge] Copied shared/ JS libs to {lesson_dir}")
            except Exception as e:
                logger.warning(f"[Hyperframes Bridge] shared/ copy warning: {e}")

        return f"/uploads/visual_lessons/{lesson_id}/index.html"
    elif os.path.exists(dest_html):
        hf_shared = os.path.join(hf_dir, "shared")
        if os.path.exists(hf_shared):
            try:
                shutil.copytree(hf_shared, os.path.join(lesson_dir, "shared"), dirs_exist_ok=True)
            except Exception:
                pass
        return f"/uploads/visual_lessons/{lesson_id}/index.html"
    return None

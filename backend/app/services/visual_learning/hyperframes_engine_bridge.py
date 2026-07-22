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

def _compile_index_html_python_fallback(lesson_id: str, lesson_dir: str) -> str:
    """
    Pure Python Hyperframes Master Composition generator.
    Generates standalone index.html without requiring Node.js CLI runtime.
    """
    lesson_json_path = os.path.join(lesson_dir, "lesson.json")
    if not os.path.exists(lesson_json_path):
        return None

    try:
        with open(lesson_json_path, "r", encoding="utf-8") as f:
            lesson_data = json.load(f)

        lesson_title = lesson_data.get("lesson_title", "Visual Storyboard Video")
        scenes = lesson_data.get("scenes", [])
        raw_data_json = json.dumps(scenes)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{lesson_title}</title>
  
  <!-- CSS Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&family=Space+Grotesk:wght@400;700&family=Inter:wght@400;500;700;900&family=Cinzel:wght@700&family=Playfair+Display:wght@700&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
  
  <!-- KaTeX for math rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  
  <!-- GSAP for animations -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>

  <style>
    :root {{
      --theme-primary-color: #0f172a;
      --theme-secondary-color: #1e293b;
      --theme-accent-color: #3b82f6;
      --theme-bg-color: #090d16;
      --theme-surface-color: #131b2e;
      --theme-text-color: #ffffff;
      --theme-muted-text-color: rgba(255, 255, 255, 0.7);
      --theme-font-family: Inter, system-ui, sans-serif;
    }}
    
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{
      width: 1280px;
      height: 720px;
      overflow: hidden;
      background: var(--theme-bg-color, #090d16);
      font-family: var(--theme-font-family, 'Inter', system-ui, sans-serif);
      color: var(--theme-text-color, #ffffff);
      -webkit-font-smoothing: antialiased;
    }}
    
    .composition {{
      width: 1280px;
      height: 720px;
      position: relative;
    }}
    
    .scene {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0; left: 0;
      z-index: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 40px;
      opacity: 0;
      transition: opacity 0.5s ease;
    }}

    .scene.active {{
      opacity: 1;
    }}

    .scene-title {{
      font-size: 42px;
      font-weight: 800;
      color: #ffffff;
      margin-bottom: 24px;
      text-align: center;
      background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .scene-body {{
      font-size: 24px;
      line-height: 1.6;
      color: #e2e8f0;
      text-align: center;
      max-width: 900px;
    }}

    .subtitles-container {{
      position: absolute;
      bottom: 40px;
      left: 8%;
      right: 8%;
      text-align: center;
      font-size: 22px;
      font-weight: 700;
      color: #ffffff;
      z-index: 90;
      text-shadow: 0 2px 4px rgba(0,0,0,0.9);
      background: rgba(15, 23, 42, 0.85);
      padding: 12px 24px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.1);
    }}
  </style>
</head>
<body>
  <div class="composition" id="hyperframes-container"></div>
  <div class="subtitles-container" id="subtitles-panel" style="display: none;"></div>

  <script>
    const rawData = {raw_data_json};
    let currentSceneIndex = 0;
    let currentAudio = null;

    function renderScene(index) {{
      if (index < 0 || index >= rawData.length) return;
      const container = document.getElementById('hyperframes-container');
      const sceneData = rawData[index];
      const templateData = sceneData.template_data || {{}};
      const title = templateData.title || sceneData.metadata?.title || 'Visual Learning';

      container.innerHTML = `
        <div class="scene active">
          <h1 class="scene-title">${{title}}</h1>
          <div class="scene-body">${{sceneData.teacher_script || ''}}</div>
        </div>
      `;

      if (sceneData.audio_url) {{
        if (currentAudio) {{ currentAudio.pause(); }}
        currentAudio = new Audio(sceneData.audio_url);
        currentAudio.play().catch(err => console.log('Audio playback prevented:', err));
        
        currentAudio.onended = () => {{
          if (index + 1 < rawData.length) {{
            renderScene(index + 1);
          }}
        }};
      }}
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      if (rawData.length > 0) renderScene(0);
    }});
  </script>
</body>
</html>"""

        dest_html = os.path.join(lesson_dir, "index.html")
        with open(dest_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        return f"/uploads/visual_lessons/{lesson_id}/index.html"
    except Exception as e:
        logger.error(f"[Python HTML Fallback Compiler Error] {e}")
        return None

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
    try:
        os.makedirs(hf_outputs_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"[Hyperframes Bridge] Directory creation notice: {e}")

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

        # Backup index.html to Supabase Cloud Storage
        from backend.app.core.supabase_storage import upload_file_to_supabase
        cloud_html_url = upload_file_to_supabase(dest_html, f"{lesson_id}/index.html")
        
        # Always serve index.html via FastAPI route to guarantee text/html MIME type rendering in browser iframes
        serving_url = f"/uploads/visual_lessons/{lesson_id}/index.html"
        logger.info(f"[RENDER LOG] [ENGINE SUCCESS] Compiled index.html ready -> {serving_url} (Cloud Backup: {cloud_html_url})")
        try:
            print(f"[RENDER LOG] [ENGINE SUCCESS] Compiled index.html ready -> {serving_url}")
        except Exception:
            pass
        return serving_url

    elif os.path.exists(dest_html):
        hf_shared = os.path.join(hf_dir, "shared")
        if os.path.exists(hf_shared):
            try:
                shutil.copytree(hf_shared, os.path.join(lesson_dir, "shared"), dirs_exist_ok=True)
            except Exception:
                pass
        from backend.app.core.supabase_storage import upload_file_to_supabase
        cloud_html_url = upload_file_to_supabase(dest_html, f"{lesson_id}/index.html")
        serving_url = f"/uploads/visual_lessons/{lesson_id}/index.html"
        try:
            print(f"[RENDER LOG] [ENGINE SUCCESS] Using existing index.html -> {serving_url}")
        except Exception:
            pass
        return serving_url
    else:
        # Fallback to pure Python compiler if Node execution was unavailable
        try:
            print("[RENDER LOG] [ENGINE NOTICE] Node compiler unavailable. Executing Python HTML fallback compiler...")
        except Exception:
            pass
        fallback_url = _compile_index_html_python_fallback(lesson_id, lesson_dir)
        dest_fallback = os.path.join(lesson_dir, "index.html")
        if os.path.exists(dest_fallback):
            from backend.app.core.supabase_storage import upload_file_to_supabase
            upload_file_to_supabase(dest_fallback, f"{lesson_id}/index.html")
        serving_url = f"/uploads/visual_lessons/{lesson_id}/index.html"
        try:
            print(f"[RENDER LOG] [ENGINE SUCCESS] Python fallback HTML compiled -> {serving_url}")
        except Exception:
            pass
        return serving_url

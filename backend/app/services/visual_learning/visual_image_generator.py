import os
import logging
from google.genai import types
from backend.app.services.retrieval import qdrant_service as qdrant

logger = logging.getLogger(__name__)

def create_fallback_svg(title: str, slide_no: int, output_path: str):
    """
    Creates a styled SVG slide with modern typography, gradients, and subtle overlays.
    This serves as a visual fallback if Imagen 3 API is unavailable.
    """
    svg_content = f"""<svg width="800" height="450" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad_{slide_no}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0b0f19;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1e1b4b;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#grad_{slide_no})" />
  <!-- Inner styled glassmorphic outline -->
  <rect x="40" y="40" width="720" height="370" rx="16" fill="white" fill-opacity="0.08" stroke="white" stroke-opacity="0.15" stroke-width="2" />
  
  <!-- Slide Number Indicator -->
  <text x="80" y="95" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="white" opacity="0.5" letter-spacing="2">SLIDE {slide_no}</text>
  
  <!-- Core Slide Title -->
  <text x="400" y="225" dominant-baseline="middle" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="36" font-weight="800" fill="white">
    {title}
  </text>
  
  <!-- Brand Footer -->
  <text x="400" y="375" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="white" opacity="0.4" letter-spacing="1">CHADUVU-GURU • VISUAL LEARNING</text>
</svg>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

async def generate_slide_images(slides: list, lesson_id: str) -> list:
    """
    Generate educational slide illustrations. Returns list of image/SVG paths relative to server root.
    """
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "..", ".."))
    lesson_dir = os.path.join(PROJECT_ROOT, "uploads", "visual_lessons", lesson_id)
    os.makedirs(lesson_dir, exist_ok=True)
    
    image_urls = []
    client = qdrant.gemini_client
    
    for slide in slides:
        slide_no = slide.get("slide_no", 1)
        title = slide.get("title", f"Slide {slide_no}")
        prompt = slide.get("image_prompt", "")
        
        jpg_filename = f"slide_{slide_no}.jpg"
        jpg_path = os.path.join(lesson_dir, jpg_filename)
        svg_filename = f"slide_{slide_no}.svg"
        svg_path = os.path.join(lesson_dir, svg_filename)
        
        success = False
        if client and prompt:
            try:
                logger.info(f"[ImageGen] Requesting Imagen 4.0 for slide {slide_no}...")
                response = client.models.generate_images(
                    model='imagen-4.0-generate-001',
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/jpeg",
                        aspect_ratio="16:9",
                    )
                )
                if response.generated_images:
                    img_bytes = response.generated_images[0].image.image_bytes
                    with open(jpg_path, "wb") as f:
                        f.write(img_bytes)
                    image_urls.append(f"/uploads/visual_lessons/{lesson_id}/{jpg_filename}")
                    success = True
                    logger.info(f"[ImageGen] Saved Imagen JPG for slide {slide_no}")
                else:
                    logger.warning(f"[ImageGen] Empty response from Imagen for slide {slide_no}")
            except Exception as e:
                logger.error(f"[ImageGen] Imagen error for slide {slide_no}: {e}. Falling back to SVG.")
        
        if not success:
            # Try free Hugging Face Stable Diffusion if token is configured
            hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
            if hf_token:
                try:
                    import httpx
                    logger.info(f"[ImageGen] Requesting Hugging Face for slide {slide_no}...")
                    hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
                    headers = {"Authorization": f"Bearer {hf_token}"}
                    payload = {"inputs": prompt}
                    async with httpx.AsyncClient(timeout=40.0) as http_client:
                        img_resp = await http_client.post(hf_url, headers=headers, json=payload)
                        if img_resp.status_code == 200:
                            with open(jpg_path, "wb") as f:
                                f.write(img_resp.content)
                            image_urls.append(f"/uploads/visual_lessons/{lesson_id}/{jpg_filename}")
                            success = True
                            logger.info(f"[ImageGen] Saved Hugging Face image for slide {slide_no}")
                        else:
                            logger.warning(f"[ImageGen] Hugging Face returned status {img_resp.status_code}: {img_resp.text}")
                except Exception as hf_err:
                    logger.error(f"[ImageGen] Hugging Face generation failed: {hf_err}")

        if not success:
            # Try free Pollinations.ai generator fallback before SVG!
            try:
                import urllib.parse
                import httpx
                import random
                logger.info(f"[ImageGen] Falling back to Pollinations.ai for slide {slide_no}...")
                encoded_prompt = urllib.parse.quote(prompt)
                seed = random.randint(1000, 9999)
                pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=450&nologo=true&seed={seed}&private=true"
                
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    img_resp = await http_client.get(pollinations_url)
                    if img_resp.status_code == 200:
                        with open(jpg_path, "wb") as f:
                            f.write(img_resp.content)
                        image_urls.append(f"/uploads/visual_lessons/{lesson_id}/{jpg_filename}")
                        success = True
                        logger.info(f"[ImageGen] Saved Pollinations.ai image for slide {slide_no}")
                    else:
                        logger.warning(f"[ImageGen] Pollinations.ai returned status {img_resp.status_code} for slide {slide_no}")
            except Exception as p_err:
                logger.error(f"[ImageGen] Pollinations.ai failed: {p_err}")
        
        if not success:
            # Create custom SVG fallback diagram from Gemini if available
            try:
                svg_content = slide.get("svg_content", "").strip()
                # Simple validation that it starts with <svg and ends with </svg>
                if svg_content and "<svg" in svg_content and "</svg>" in svg_content:
                    # Strip any markdown wrappers if LLM accidentally wrapped it
                    if "```xml" in svg_content:
                        svg_content = svg_content.split("```xml")[1].split("```")[0].strip()
                    elif "```" in svg_content:
                        svg_content = svg_content.split("```")[1].split("```")[0].strip()
                    
                    with open(svg_path, "w", encoding="utf-8") as f:
                        f.write(svg_content)
                    image_urls.append(f"/uploads/visual_lessons/{lesson_id}/{svg_filename}")
                    success = True
                    logger.info(f"[ImageGen] Saved custom SVG illustration from Gemini blueprint for slide {slide_no}")
                else:
                    # Fallback to the default styled SVG template
                    create_fallback_svg(title, slide_no, svg_path)
                    image_urls.append(f"/uploads/visual_lessons/{lesson_id}/{svg_filename}")
                    logger.info(f"[ImageGen] Saved default fallback SVG for slide {slide_no}")
            except Exception as svg_err:
                logger.error(f"[ImageGen] SVG fallback failed: {svg_err}")
                raise RuntimeError(f"Could not generate image or SVG fallback for slide {slide_no}")
                
    return image_urls

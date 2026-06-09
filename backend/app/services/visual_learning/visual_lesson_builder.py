def build_lesson_package(blueprint: dict, image_urls: list, audio_urls: list, lesson_id: str) -> dict:
    """
    Assembles the final lesson package by mapping slide images and audio URLs 
    to their respective slides from the blueprint.
    """
    slides = blueprint.get("slides", [])
    assembled_slides = []
    
    # We map the lists of generated URLs to the slides sequentially
    for idx, slide in enumerate(slides):
        slide_no = slide.get("slide_no", idx + 1)
        title = slide.get("title", f"Slide {slide_no}")
        teacher_script = slide.get("teacher_script", "")
        
        # Get URLs matching this index (safeguard against size mismatch)
        image_url = image_urls[idx] if idx < len(image_urls) else ""
        audio_url = audio_urls[idx] if idx < len(audio_urls) else ""
        
        assembled_slides.append({
            "slide_no": slide_no,
            "title": title,
            "image_url": image_url,
            "audio_url": audio_url,
            "teacher_script": teacher_script
        })
        
    return {
        "lesson_title": blueprint.get("lesson_title", "Visual Lesson"),
        "lesson_type": blueprint.get("lesson_type", "conceptual"),
        "lesson_id": lesson_id,
        "slides": assembled_slides
    }

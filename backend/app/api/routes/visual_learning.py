from typing import Union
import logging
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.app.services.visual_learning.visual_learning_service import generate_visual_lesson_stream


logger = logging.getLogger(__name__)
router = APIRouter()

class VisualLearningRequest(BaseModel):
    query: str
    book_uuid: str
    class_name: Union[str, int]
    subject: str

@router.post("/api/visual_learning", tags=["Visual Learning"])
async def start_visual_learning(request: VisualLearningRequest):
    """
    POST endpoint for triggering Visual Learning Mode lesson generation.
    Returns a text/event-stream (SSE) representing real-time creation progress
    and final lesson blueprint with assets.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    class_name_str = str(request.class_name).strip()
    logger.info(f"[Route] Visual Learning initiated: query='{query}', book='{request.book_uuid}', class='{class_name_str}'")
    
    return StreamingResponse(
        generate_visual_lesson_stream(
            query=query,
            book_uuid=request.book_uuid,
            class_name=class_name_str,
            subject=request.subject
        ),
        media_type="text/event-stream"
    )

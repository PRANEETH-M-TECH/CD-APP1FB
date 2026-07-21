import logging
from fastapi import APIRouter, Query, HTTPException

from backend.app.services.analytics import achievements_service
from backend.app.services.analytics import profile_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/achievements/summary", tags=["Achievements"])
async def get_achievements_summary(uid: str = Query(...)):
    """
    Get comprehensive achievements data for a user.
    """
    try:
        logger.info(f"[ACHIEVEMENTS] Fetching achievements for uid: {uid}")
        achievements_data = achievements_service.get_user_achievements(uid)
        return achievements_data
    except Exception as e:
        logger.error(f"Failed to get achievements for {uid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/achievements/tiers", tags=["Achievements"])
async def get_achievement_tiers():
    """Get information about all achievement tiers"""
    try:
        tiers = ["newcomer", "rising_star", "scholar", "master", "legend"]
        tier_info = [achievements_service.get_tier_info(tier) for tier in tiers]
        return {"tiers": tier_info}
    except Exception as e:
        logger.error(f"Failed to get tier info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/profile/stats", tags=["Profile"])
async def get_profile_stats(uid: str = Query(...)):
    """
    Get comprehensive profile statistics for the enhanced profile page.
    """
    try:
        logger.info(f"[PROFILE] Fetching profile stats for uid: {uid}")
        profile_data = profile_service.get_profile_stats(uid)
        return profile_data
    except Exception as e:
        logger.error(f"Failed to get profile stats for {uid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

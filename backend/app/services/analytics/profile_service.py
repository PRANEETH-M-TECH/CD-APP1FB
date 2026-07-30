"""
Profile Service for Chaduvu Guru
Provides comprehensive profile statistics, levels, rankings, and analytics
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from google.cloud import firestore
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

from backend.app.core.firebase.firebase_init import db


def calculate_level_and_xp(total_questions: int) -> Dict:
    """
    Calculate user level and XP based on total questions.
    
    Level system:
    - Level 1-5: Newbie (0-500 questions)
    - Level 6-10: Rising Star (500-1000 questions)
    - Level 11-15: Pro Player (1000-1500 questions)
    - Level 16-20: Champion (1500-2000 questions)
    - Level 21+: Legend (2000+ questions)
    
    Each level requires 100 questions
    """
    level = (total_questions // 100) + 1
    current_level_xp = total_questions % 100
    next_level_xp = 100
    xp_progress = (current_level_xp / next_level_xp) * 100
    
    # Determine tier based on level
    if level <= 5:
        tier = "newbie"
        tier_name = "Newbie"
        tier_icon = "🌱"
    elif level <= 10:
        tier = "rising_star"
        tier_name = "Rising Star"
        tier_icon = "⭐"
    elif level <= 15:
        tier = "pro_player"
        tier_name = "Pro Player"
        tier_icon = "🎮"
    elif level <= 20:
        tier = "champion"
        tier_name = "Champion"
        tier_icon = "🏆"
    else:
        tier = "legend"
        tier_name = "Legend"
        tier_icon = "💎"
    
    return {
        "level": level,
        "current_xp": current_level_xp,
        "next_level_xp": next_level_xp,
        "xp_progress": round(xp_progress, 1),
        "tier": tier,
        "tier_name": tier_name,
        "tier_icon": tier_icon,
        "total_xp": total_questions
    }


def get_class_ranking(uid: str, user_class: str, total_points: int) -> Dict:
    """
    Get user's ranking in their class.
    Returns rank and total students in class.
    """
    try:
        # Get all students in the same class
        achievements_ref = db.collection("user_achievements")
        students = achievements_ref.stream()
        
        # Get their points and sort
        class_students = []
        for student in students:
            student_data = student.to_dict()
            # TODO: Filter by class (would need class info in user_achievements or user profile)
            if student_data.get("total_points"):
                class_students.append({
                    "uid": student.id,
                    "points": student_data.get("total_points", 0)
                })
        
        # Sort by points (descending)
        class_students.sort(key=lambda x: x["points"], reverse=True)
        
        # Find user's rank
        rank = 1
        for idx, student in enumerate(class_students):
            if student["uid"] == uid:
                rank = idx + 1
                break
        
        return {
            "rank": rank,
            "total_students": len(class_students),
            "percentile": round((1 - (rank / len(class_students))) * 100, 1) if class_students else 0
        }
    except Exception as e:
        logger.error(f"Error calculating class ranking: {e}")
        return {
            "rank": 0,
            "total_students": 0,
            "percentile": 0
        }


def get_subject_distribution(uid: str) -> Dict:
    """
    Get subject-wise question distribution.
    Returns count and percentage for each subject.
    """
    try:
        from .analytics_service import rebuild_user_analytics_from_queries
        user_stats = rebuild_user_analytics_from_queries(uid)
        
        subjects_count = user_stats.get("subjects_count", {})
        total = sum(subjects_count.values()) or 1  # Avoid division by zero
        
        distribution = {}
        for subject, count in subjects_count.items():
            distribution[subject] = {
                "count": count,
                "percentage": round((count / total) * 100, 1)
            }
        
        return distribution
    except Exception as e:
        logger.error(f"Error getting subject distribution: {e}")
        return {}


def get_activity_heatmap(uid: str, days: int = 90) -> List[Dict]:
    """
    Get activity heatmap data for last N days (GitHub-style).
    Returns list of {date, count} for each day.
    """
    try:
        # Get queries from last N days
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        queries_ref = db.collection("users").document(uid).collection("queries").where("timestamp", ">=", start_date)
        queries = queries_ref.stream()
        
        # Count queries per day
        daily_counts = defaultdict(int)
        for query in queries:
            query_data = query.to_dict()
            timestamp = query_data.get("timestamp")
            if timestamp:
                # Convert to date string (YYYY-MM-DD)
                date_str = timestamp.date().isoformat()
                daily_counts[date_str] += 1
        
        # Convert to list format
        heatmap_data = []
        for i in range(days):
            date = (datetime.now(timezone.utc) - timedelta(days=days - i - 1)).date()
            date_str = date.isoformat()
            heatmap_data.append({
                "date": date_str,
                "count": daily_counts.get(date_str, 0)
            })
        
        return heatmap_data
    except Exception as e:
        logger.error(f"Error generating activity heatmap: {e}")
        return []


def get_recent_activities(uid: str, limit: int = 5) -> List[Dict]:
    """
    Get recent activities: questions asked, badges unlocked.
    """
    try:
        activities = []
        
        # Get recent queries
        queries_ref = db.collection("users").document(uid).collection("queries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
        queries = queries_ref.stream()
        
        for query in queries:
            query_data = query.to_dict()
            activities.append({
                "type": "question",
                "icon": "🎯",
                "text": f"Asked: {query_data.get('query', 'Question')[:50]}...",
                "timestamp": query_data.get("timestamp").isoformat() if query_data.get("timestamp") else None,
                "subject": query_data.get("subject", "General")
            })
        
        # Get recent badge unlocks
        achievements_doc = db.collection("user_achievements").document(uid).get()
        if achievements_doc.exists:
            unlock_history = achievements_doc.to_dict().get("unlock_history", [])
            # Sort by timestamp (most recent first)
            sorted_history = sorted(unlock_history, key=lambda x: x.get("unlocked_at", datetime.min), reverse=True)[:3]
            
            for unlock in sorted_history:
                activities.append({
                    "type": "badge",
                    "icon": unlock.get("icon", "🏆"),
                    "text": f"Unlocked: {unlock.get('name')} badge",
                    "timestamp": unlock.get("unlocked_at").isoformat() if unlock.get("unlocked_at") else None,
                    "points": unlock.get("points", 0)
                })
        
        # Sort all activities by timestamp
        activities.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return activities[:limit]
    except Exception as e:
        logger.error(f"Error getting recent activities: {e}")
        return []


def get_profile_stats(uid: str) -> Dict:
    """
    Get comprehensive profile statistics.
    
    Returns:
        {
            "user_info": {...},
            "level_info": {...},
            "quick_stats": {...},
            "achievements": {...},
            "subject_distribution": {...},
            "activity_heatmap": [...],
            "recent_activities": [...],
            "class_ranking": {...}
        }
    """
    try:
        # Import analytics service
        from .analytics_service import rebuild_user_analytics_from_queries
        from .achievements_service import get_user_achievements
        
        # Get user analytics
        user_stats = rebuild_user_analytics_from_queries(uid)
        
        # Get achievements
        achievements_data = get_user_achievements(uid)
        
        # Calculate level and XP
        total_questions = user_stats.get("total_queries", 0)
        level_info = calculate_level_and_xp(total_questions)
        
        # Get top 6 achievements
        all_achievements = achievements_data.get("achievements", [])
        unlocked_achievements = [a for a in all_achievements if a.get("is_unlocked")]
        # Sort by tier priority and points
        tier_priority = {"legendary": 1, "diamond": 2, "gold": 3, "silver": 4, "bronze": 5}
        unlocked_achievements.sort(key=lambda x: (tier_priority.get(x.get("tier"), 10), -x.get("points", 0)))
        top_achievements = unlocked_achievements[:6]
        
        # Quick stats
        quick_stats = {
            "streak": user_stats.get("streak", 0),
            "total_questions": total_questions,
            "total_points": achievements_data.get("total_points", 0),
            "unlocked_badges": achievements_data.get("unlocked_count", 0),
            "total_badges": achievements_data.get("total_count", 0)
        }
        
        # Get class ranking (dummy for now)
        class_ranking = get_class_ranking(uid, "10", quick_stats["total_points"])
        quick_stats["class_rank"] = class_ranking.get("rank", 0)
        
        # Subject distribution
        subject_distribution = get_subject_distribution(uid)
        
        # Activity heatmap
        activity_heatmap = get_activity_heatmap(uid, days=90)
        
        # Recent activities
        recent_activities = get_recent_activities(uid, limit=5)
        
        return {
            "level_info": level_info,
            "quick_stats": quick_stats,
            "top_achievements": top_achievements,
            "subject_distribution": subject_distribution,
            "activity_heatmap": activity_heatmap,
            "recent_activities": recent_activities,
            "class_ranking": class_ranking
        }
        
    except Exception as e:
        logger.error(f"Error getting profile stats for {uid}: {e}", exc_info=True)
        return {
            "level_info": calculate_level_and_xp(0),
            "quick_stats": {
                "streak": 0,
                "total_questions": 0,
                "total_points": 0,
                "unlocked_badges": 0,
                "total_badges": 0,
                "class_rank": 0
            },
            "top_achievements": [],
            "subject_distribution": {},
            "activity_heatmap": [],
            "recent_activities": [],
            "class_ranking": {"rank": 0, "total_students": 0, "percentile": 0}
        }

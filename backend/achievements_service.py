"""
Achievements Service for Chaduvu Guru
Manages badges, points, and gamification features with cultural relevance for Indian students.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from google.cloud import firestore
import logging

logger = logging.getLogger(__name__)

# Initialize Firestore client (assumes already initialized in app.py)
db = None  # Will be set by app.py

def initialize_db(firestore_client):
    """Initialize the Firestore client"""
    global db
    db = firestore_client


class Achievement:
    """Represents a single achievement/badge"""
    def __init__(self, id: str, name: str, icon: str, description: str, 
                 category: str, points: int, tier: str, condition_type: str, condition_value: int):
        self.id = id
        self.name = name
        self.icon = icon
        self.description = description
        self.category = category  # core, streak, subject, special, time, milestone
        self.points = points
        self.tier = tier  # bronze, silver, gold, diamond, legendary
        self.condition_type = condition_type  # total_queries, streak, subject_queries, special_event
        self.condition_value = condition_value


# Define all achievements - Indian Student Edition (English names, Gaming/Comic vibes!)
ACHIEVEMENTS = [
    # Core Learning Badges (Gaming/Comic themed)
    Achievement("rookie", "Rookie", "🎮", "Asked your first 5 questions - Every gamer's start!", "core", 50, "bronze", "total_queries", 5),
    Achievement("curious_cat", "Curious Cat", "😺", "Asked 25 questions - Getting curious!", "core", 100, "silver", "total_queries", 25),
    Achievement("brain_gym", "Brain Gym", "💪🧠", "Asked 50 questions - Flexing that brain!", "core", 200, "gold", "total_queries", 50),
    Achievement("quiz_master", "Quiz Master", "🎯", "Asked 100 questions - Quiz champion!", "core", 400, "gold", "total_queries", 100),
    Achievement("sherlock", "Sherlock", "🔍", "Asked 250 questions - Detective level!", "core", 1000, "diamond", "total_queries", 250),
    Achievement("genius_mode", "Genius Mode", "🚀", "Asked 500+ questions - Unlocked genius!", "core", 2500, "legendary", "total_queries", 500),
    
    # Streak Badges (Universal English + Hyderabad Slang)
    Achievement("on_fire", "On Fire", "🔥", "3-day streak - You're heating up!", "streak", 60, "bronze", "streak", 3),
    Achievement("thunder_bolt", "Thunder Bolt", "⚡", "7-day streak - Electrifying!", "streak", 150, "silver", "streak", 7),
    Achievement("kiraak", "Kiraak", "😎", "10-day streak - किराक! Super cool!", "streak", 250, "gold", "streak", 10),
    Achievement("north_star", "North Star", "⭐", "15-day streak - Always shining!", "streak", 300, "gold", "streak", 15),
    Achievement("space_cadet", "Space Cadet", "🚀", "30-day streak - To infinity!", "streak", 600, "diamond", "streak", 30),
    Achievement("sun_never_sets", "Sun Never Sets", "☀️", "60-day streak - Unstoppable!", "streak", 1200, "legendary", "streak", 60),
    
    
    # Subject Mastery Badges (Student-friendly) - Updated for Physics & Biology
    Achievement("lab_rat", "Lab Rat", "🔬", "Mastered 30+ Science questions - Lab expert!", "subject", 300, "gold", "subject_queries_science", 30),
    Achievement("physics_wizard", "Physics Wizard", "⚛️", "Mastered 30+ Physics questions - Einstein level!", "subject", 300, "gold", "subject_queries_physics", 30),
    Achievement("bio_master", "Bio Master", "🧬", "Mastered 30+ Biology questions - Life science pro!", "subject", 300, "gold", "subject_queries_biology", 30),
    Achievement("maths_ninja", "Maths Ninja", "🥷🔢", "Mastered 30+ Maths questions - Ninja level!", "subject", 300, "gold", "subject_queries_maths", 30),
    Achievement("time_traveler", "Time Traveler", "⏰🌍", "Mastered 30+ Social Studies - History expert!", "subject", 300, "gold", "subject_queries_social", 30),
    Achievement("word_wizard", "Word Wizard", "📖✨", "Mastered 30+ Language questions - Magic with words!", "subject", 300, "gold", "subject_queries_english", 30),
    
    # Special Milestone Badges
    Achievement("first_blood", "First Blood", "🎯", "Asked your very first question - First Blood!", "milestone", 20, "bronze", "total_queries", 1),
    Achievement("weekly_warrior", "Weekly Warrior", "⚔️", "Learned every day for a week - Warrior mode!", "milestone", 150, "silver", "streak", 7),
    Achievement("all_rounder", "All-Rounder", "🌈", "Studied all subjects in one week - Cricket style!", "special", 400, "diamond", "all_subjects_week", 1),
    Achievement("bindaas", "Bindaas", "💯", "Fearlessly studied all subjects - बिंदास mode!", "special", 500, "diamond", "all_subjects_week", 1),
   
    # Time-based Badges  
    Achievement("early_bird", "Early Bird", "🐦🌅", "Early bird! Studied before 6 AM", "time", 100, "gold", "early_morning", 1),
]


def get_user_achievements(uid: str) -> Dict:
    """
    Get all achievements for a user with unlock status.
    
    Returns:
        {
            "total_points": int,
            "tier": str,
            "unlocked_count": int,
            "total_count": int,
            "achievements": List[Dict]
        }
    """
    try:
        # Get user stats from analytics
        from .analytics_service import rebuild_user_analytics_from_queries
        user_stats = rebuild_user_analytics_from_queries(uid)
        
        # Get user achievements document
        achievements_doc = db.collection("user_achievements").document(uid).get()
        user_achievements = achievements_doc.to_dict() if achievements_doc.exists else {
            "unlocked_badges": [],
            "total_points": 0,
            "unlock_history": []
        }
        
        unlocked_ids = set(user_achievements.get("unlocked_badges", []))
        total_points = user_achievements.get("total_points", 0)
        
        # Check all achievements
        achievement_list = []
        newly_unlocked = []
        
        for achievement in ACHIEVEMENTS:
            is_unlocked = achievement.id in unlocked_ids
            can_unlock = False
            progress = 0
            
            # Check unlock conditions
            if not is_unlocked:
                can_unlock, progress = check_achievement_condition(achievement, user_stats, uid)
                
                # Auto-unlock if conditions met
                if can_unlock:
                    unlocked_ids.add(achievement.id)
                    newly_unlocked.append(achievement)
                    total_points += achievement.points
            
            achievement_list.append({
                "id": achievement.id,
                "name": achievement.name,
                "icon": achievement.icon,
                "description": achievement.description,
                "category": achievement.category,
                "points": achievement.points,
                "tier": achievement.tier,
                "is_unlocked": is_unlocked or can_unlock,
                "progress": progress,
                "unlock_date": get_unlock_date(user_achievements, achievement.id)
            })
        
        # Save newly unlocked achievements
        if newly_unlocked:
            update_user_achievements(uid, list(unlocked_ids), total_points, newly_unlocked)
        
        # Determine user tier
        tier = get_user_tier(total_points)
        
        return {
            "total_points": total_points,
            "tier": tier,
            "unlocked_count": len(unlocked_ids),
            "total_count": len(ACHIEVEMENTS),
            "achievements": achievement_list,
            "newly_unlocked": [{"id": a.id, "name": a.name, "icon": a.icon, "points": a.points} for a in newly_unlocked]
        }
        
    except Exception as e:
        logger.error(f"Error getting achievements for {uid}: {e}", exc_info=True)
        return {
            "total_points": 0,
            "tier": "newcomer",
            "unlocked_count": 0,
            "total_count": len(ACHIEVEMENTS),
            "achievements": [],
            "newly_unlocked": []
        }


def check_achievement_condition(achievement: Achievement, user_stats: Dict, uid: str) -> tuple:
    """
    Check if achievement conditions are met.
    
    Returns:
        (can_unlock: bool, progress: float)  # progress is 0.0 to 1.0
    """
    try:
        condition_type = achievement.condition_type
        condition_value = achievement.condition_value
        
        if condition_type == "total_queries":
            current = user_stats.get("total_queries", 0)
            progress = min(current / condition_value, 1.0)
            return (current >= condition_value, progress)
        
        elif condition_type == "streak":
            current = user_stats.get("streak", 0)
            progress = min(current / condition_value, 1.0)
            return (current >= condition_value, progress)
        
        elif condition_type.startswith("subject_queries_"):
            subject = condition_type.replace("subject_queries_", "")
            subjects_count = user_stats.get("subjects_count", {})
            current = subjects_count.get(subject, 0)
            progress = min(current / condition_value, 1.0)
            return (current >= condition_value, progress)
        
        
        elif condition_type == "all_subjects_week":
            # Check if user queried all major subjects in last 7 days
            # Accept either: science OR (physics AND biology) for flexibility
            subjects_count = user_stats.get("subjects_count", {})
            
            # Core subjects everyone needs
            core_subjects = ["maths", "social", "english"]
            has_core = all(subjects_count.get(subj, 0) > 0 for subj in core_subjects)
            
            # Science subjects - accept either combined science OR both physics & biology
            has_science = (subjects_count.get("science", 0) > 0) or \
                         (subjects_count.get("physics", 0) > 0 and subjects_count.get("biology", 0) > 0)
            
            has_all = has_core and has_science
            
            # Calculate progress
            core_progress = sum(1 for subj in core_subjects if subjects_count.get(subj, 0) > 0) / 3.0
            science_progress = 1.0 if has_science else 0.0
            progress = (core_progress * 0.75) + (science_progress * 0.25)
            
            return (has_all, progress)
        
        elif condition_type == "early_morning":
            # Check if user has any queries before 6 AM (would need timestamp analysis)
            # For now, simplified check
            return (False, 0.0)  # Implement with actual time-based logic
        
        return (False, 0.0)
        
    except Exception as e:
        logger.error(f"Error checking condition for {achievement.id}: {e}")
        return (False, 0.0)


def update_user_achievements(uid: str, unlocked_badges: List[str], total_points: int, newly_unlocked: List[Achievement]):
    """Update user achievements in Firestore"""
    try:
        doc_ref = db.collection("user_achievements").document(uid)
        
        # Add to unlock history
        unlock_history = []
        for achievement in newly_unlocked:
            unlock_history.append({
                "achievement_id": achievement.id,
                "name": achievement.name,
                "icon": achievement.icon,
                "points": achievement.points,
                "unlocked_at": firestore.SERVER_TIMESTAMP
            })
        
        doc_ref.set({
            "uid": uid,
            "unlocked_badges": unlocked_badges,
            "total_points": total_points,
            "unlock_history": firestore.ArrayUnion(unlock_history),
            "last_updated": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        logger.info(f"✅ Updated achievements for {uid}: {len(newly_unlocked)} new badges")
        
    except Exception as e:
        logger.error(f"Error updating achievements for {uid}: {e}")


def get_unlock_date(user_achievements: Dict, achievement_id: str) -> Optional[str]:
    """Get unlock date for an achievement"""
    unlock_history = user_achievements.get("unlock_history", [])
    for entry in unlock_history:
        if entry.get("achievement_id") == achievement_id:
            timestamp = entry.get("unlocked_at")
            if timestamp:
                return timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
    return None


def get_user_tier(total_points: int) -> str:
    """Determine user tier based on total points"""
    if total_points >= 2500:
        return "legend"
    elif total_points >= 1000:
        return "champion"
    elif total_points >= 500:
        return "pro_player"
    elif total_points >= 100:
        return "rising_star"
    else:
        return "newbie"


def get_tier_info(tier: str) -> Dict:
    """Get tier display information"""
    tiers = {
        "newbie": {"name": "Newbie", "icon": "🌱", "color": "#10b981", "min_points": 0},
        "rising_star": {"name": "Rising Star", "icon": "⭐", "color": "#f59e0b", "min_points": 100},
        "pro_player": {"name": "Pro Player", "icon": "🎮", "color": "#3b82f6", "min_points": 500},
        "champion": {"name": "Champion", "icon": "🏆", "color": "#8b5cf6", "min_points": 1000},
        "legend": {"name": "Legend", "icon": "💎", "color": "#ec4899", "min_points": 2500}
    }
    return tiers.get(tier, tiers["newbie"])

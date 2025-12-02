"""
Enhanced Dashboard Service with AI-Powered Feedback
Analyzes student patterns and provides friendly insights.
"""

from google.cloud import firestore
from .firebase.firebase_init import db
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import defaultdict, Counter
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Import existing dashboard service functions
from .dashboard_service import (
    get_dashboard_summary,
    get_weekly_activity,
    get_strength_weakness,
    get_frequent_questions
)

# ==========================================
# AI-POWERED STUDENT FEEDBACK
# ==========================================

def generate_student_feedback(uid: str) -> Dict:
    """
    Generate AI-powered, student-friendly feedback and remarks.
    
    Analyzes student's query patterns and provides:
    - Friendly encouragement
    - Identified weak areas
    - Study suggestions
    - Streak motivation
    
    Returns:
        {
            "overall_feedback": str,
            "weak_topics": [str, ...],
            "strengths": [str, ...],
            "suggestions": [str, ...],
            "motivation_message": str
        }
    """
    try:
        # Get user stats and history
        stats_doc = db.collection("user_stats").document(uid).get()
        if not stats_doc.exists:
            return {
                "overall_feedback": "Hey! Start asking questions to get personalized feedback! 🌟",
                "weak_topics": [],
                "strengths": [],
                "suggestions": ["Start exploring different subjects!", "Build your learning streak!"],
                "motivation_message": "Every question is a step towards learning! 📚"
            }
        
        stats = stats_doc.to_dict()
        
        # Get recent queries
        queries_ref = db.collection("user_queries")\
            .where("uid", "==", uid)\
            .order_by("timestamp", direction=firestore.Query.DESCENDING)\
            .limit(50)
        
        queries = list(queries_ref.stream())
        
        # Analyze patterns
        subject_counts = stats.get("subjects_count", {})
        total_queries = stats.get("total_queries", 0)
        streak = stats.get("streak", 0)
        
        # Identify repeated topics (weakness indicators)
        chapter_frequency = Counter()
        recent_subjects = []
        
        for query_doc in queries:
            query_data = query_doc.to_dict()
            chapter = query_data.get("chapter_name", "Unknown")
            subject = query_data.get("subject", "unknown")
            chapter_frequency[f"{subject}:{chapter}"] += 1
            recent_subjects.append(subject)
        
        # Find topics asked about 3+ times (potential struggle areas)
        weak_topics = [topic for topic, count in chapter_frequency.most_common(10) if count >= 3]
        
        # Generate AI feedback
        prompt = f"""You are a friendly AI buddy helping a student understand their learning patterns.

STUDENT DATA:
- Total questions asked: {total_queries}
- Current streak: {streak} days
- Subjects studied: {', '.join(subject_counts.keys())}
- Most asked topics: {', '.join([t.split(':')[1] for t in chapter_frequency.most_common(5)])}
- Topics asked repeatedly: {', '.join([t.split(':')[1] for t in weak_topics]) if weak_topics else 'None'}

TASK:
Write a SHORT, friendly message (2-3 sentences) for this student that:
1. Celebrates their efforts (be specific about streak or query count)
2. Gently points out which topics they might need more help with
3. Encourages them to keep going

TONE: Like a supportive friend, NOT a teacher. Use emojis. Be warm and motivating.

Response format (JSON):
{{
  "overall_feedback": "your friendly 2-3 sentence message",
  "motivation_message": "one encouraging line about their streak or progress"
}}
"""
        
        # Call LLM
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            # Parse JSON from response
            import json
            response_text = response.text.strip()
            # Extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            ai_feedback = json.loads(response_text)
            
        except Exception as e:
            logger.error(f"AI feedback generation failed: {e}")
            ai_feedback = {
                "overall_feedback": f"You've asked {total_queries} questions! That's awesome! 🎉 Keep exploring and learning!",
                "motivation_message": f"Your {streak}-day streak is impressive! 🔥"
            }
        
        # Identify strengths (subjects with most engagement)
        strengths = [subj for subj, _ in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:2]]
        
        # Generate suggestions
        suggestions = []
        if weak_topics:
            suggestions.append(f"Try reviewing {weak_topics[0].split(':')[1]} with simpler questions first!")
        if streak < 3:
            suggestions.append("Build a daily learning streak - even 1 question a day helps!")
        if len(subject_counts) < 3:
            suggestions.append("Explore more subjects to broaden your knowledge!")
        
        return {
            "overall_feedback": ai_feedback.get("overall_feedback", "Great job learning!"),
            "weak_topics": [t.split(':')[1] for t in weak_topics[:3]],  # Top 3
            "strengths": strengths,
            "suggestions": suggestions if suggestions else ["Keep up the great work!"],
            "motivation_message": ai_feedback.get("motivation_message", f"Your {streak}-day streak is amazing! 🌟")
        }
        
    except Exception as e:
        logger.error(f"Failed to generate feedback: {e}", exc_info=True)
        return {
            "overall_feedback": "Keep asking questions and learning! 💪",
            "weak_topics": [],
            "strengths": [],
            "suggestions": ["Keep exploring!"],
            "motivation_message": "You're doing great! 🌟"
        }


def get_chapter_hotspots_for_student(uid: str, limit: int = 5) -> List[Dict]:
    """
    Get most queried chapters for this specific student.
    
    Returns:
        List of {chapter_name, subject, query_count, last_asked}
    """
    try:
        # Get all queries for this user
        queries_ref = db.collection("user_queries")\
            .where("uid", "==", uid)\
            .order_by("timestamp", direction=firestore.Query.DESCENDING)\
            .limit(100)
        
        queries = list(queries_ref.stream())
        
        # Count by chapter
        chapter_data = defaultdict(lambda: {"count": 0, "subject": "", "last_asked": None})
        
        for query_doc in queries:
            query = query_doc.to_dict()
            chapter = query.get("chapter_name", "Unknown")
            subject = query.get("subject", "unknown")
            timestamp = query.get("timestamp")
            
            chapter_data[chapter]["count"] += 1
            chapter_data[chapter]["subject"] = subject
            if chapter_data[chapter]["last_asked"] is None or (timestamp and chapter_data[chapter]["last_asked"] and timestamp > chapter_data[chapter]["last_asked"]):
                chapter_data[chapter]["last_asked"] = timestamp
        
        # Sort and format
        hotspots = []
        for chapter, data in sorted(chapter_data.items(), key=lambda x: x[1]["count"], reverse=True)[:limit]:
            last_asked = data["last_asked"]
            if last_asked and hasattr(last_asked, 'strftime'):
                last_asked_str = last_asked.strftime("%Y-%m-%d")
            else:
                last_asked_str = "Recently"
            
            hotspots.append({
                "chapter_name": chapter,
                "subject": data["subject"],
                "query_count": data["count"],
                "last_asked": last_asked_str,
                "is_struggle_area": data["count"] >= 5  # Flag if asked 5+ times
            })
        
        return hotspots
        
    except Exception as e:
        logger.error(f"Failed to get chapter hotspots for student {uid}: {e}", exc_info=True)
        return []


def get_enhanced_dashboard_data(uid: str) -> Dict:
    """
    Get complete enhanced dashboard data in one call.
    
    Returns all data needed for the student dashboard:
    - Basic stats
    - Weekly activity
    - AI feedback
    - Chapter hotspots
    - Recent questions
    """
    try:
        return {
            "summary": get_dashboard_summary(uid),
            "weekly_activity": get_weekly_activity(uid, weeks=2),
            "ai_feedback": generate_student_feedback(uid),
            "chapter_hotspots": get_chapter_hotspots_for_student(uid, limit=5),
            "recent_questions": get_frequent_questions(uid, limit=5),
            "strength_weakness": get_strength_weakness(uid)
        }
    except Exception as e:
        logger.error(f"Failed to get enhanced dashboard data for {uid}: {e}", exc_info=True)
        raise


logger.info("✅ Enhanced dashboard service loaded successfully")

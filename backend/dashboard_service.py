"""
Dashboard Service for CHADUVU-GURU
Aggregates analytics data for student and admin dashboards.
"""

from google.cloud import firestore
from .firebase.firebase_init import db
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==========================================
# STUDENT DASHBOARD DATA AGGREGATION
# ==========================================

def get_dashboard_summary(uid: str) -> Dict:
    """
    Get summary statistics for student dashboard.
    
    Returns:
        {
            "total_queries": int,
            "streak": int,
            "last_active": str,
            "top_subjects": [{subject, count}, ...],
            "total_subjects": int
        }
    """
    try:
        doc_ref = db.collection("user_stats").document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            return {
                "total_queries": 0,
                "streak": 0,
                "last_active": None,
                "top_subjects": [],
                "total_subjects": 0
            }
        
        data = doc.to_dict()
        
        # Get top subjects
        subjects_count = data.get("subjects_count", {})
        top_subjects = sorted(
            [{"subject": k, "count": v} for k, v in subjects_count.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:5]  # Top 5
        
        # Format last_active
        last_active = data.get("last_active")
        if last_active and hasattr(last_active, 'strftime'):
            last_active_str = last_active.strftime("%Y-%m-%d")
        else:
            last_active_str = None
        
        return {
            "total_queries": data.get("total_queries", 0),
            "streak": data.get("streak", 0),
            "last_active": last_active_str,
            "top_subjects": top_subjects,
            "total_subjects": len(subjects_count)
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard summary for {uid}: {e}", exc_info=True)
        raise


def get_weekly_activity(uid: str, weeks: int = 4) -> Dict:
    """
    Get weekly activity data for charts.
    
    Args:
        uid: User ID
        weeks: Number of weeks to include (default 4)
    
    Returns:
        {
            "dates": ["2025-12-01", "2025-12-02", ...],
            "counts": [5, 8, 3, ...]
        }
    """
    try:
        doc_ref = db.collection("user_stats").document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            logger.warning(f"No user_stats document found for UID: {uid} in get_weekly_activity.")
            return {"dates": [], "counts": []}
        
        data = doc.to_dict()
        weekly_activity = data.get("weekly_activity", {})
        
        # Generate last N weeks of dates
        today = datetime.now().date()
        dates = []
        for i in range(weeks * 7):
            date = today - timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))
        
        dates.reverse()  # Chronological order
        
        # Get counts for each date
        counts = [weekly_activity.get(date, 0) for date in dates]
        
        return {
            "dates": dates,
            "counts": counts
        }
        
    except Exception as e:
        logger.error(f"Failed to get weekly activity for {uid}: {e}", exc_info=True)
        raise


def get_strength_weakness(uid: str) -> Dict:
    """
    Analyze user's strengths and weaknesses by subject.
    
    Returns:
        {
            "strengths": [{subject, query_count, percentage}, ...],
            "weaknesses": [{subject, query_count, percentage}, ...]
        }
    """
    try:
        doc_ref = db.collection("user_stats").document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            logger.warning(f"No user_stats document found for UID: {uid} in get_strength_weakness.")
            return {"strengths": [], "weaknesses": []}
        
        data = doc.to_dict()
        subjects_count = data.get("subjects_count", {})
        total_queries = data.get("total_queries", 0)
        
        if total_queries == 0:
            return {"strengths": [], "weaknesses": []}
        
        # Calculate percentages
        subjects_with_pct = [
            {
                "subject": subject,
                "query_count": count,
                "percentage": round((count / total_queries) * 100, 1)
            }
            for subject, count in subjects_count.items()
        ]
        
        # Sort by count
        subjects_sorted = sorted(subjects_with_pct, key=lambda x: x["query_count"], reverse=True)
        
        # Top 3 are strengths, bottom 3 are weaknesses
        strengths = subjects_sorted[:3]
        weaknesses = subjects_sorted[-3:] if len(subjects_sorted) > 3 else []
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
    except Exception as e:
        logger.error(f"Failed to get strength/weakness for {uid}: {e}", exc_info=True)
        raise


def get_frequent_questions(uid: str, limit: int = 10) -> List[Dict]:
    """
    Get recent frequent questions.
    
    Args:
        uid: User ID
        limit: Number of queries to return
    
    Returns:
        List of query dictionaries with metadata
    """
    try:
        queries_ref = db.collection("user_queries")\
            .where("uid", "==", uid)\
            .limit(limit)
        
        docs = queries_ref.stream()
        
        queries = []
        for doc in docs:
            data = doc.to_dict()
            
            # Format timestamp
            timestamp = data.get("timestamp")
            if timestamp and hasattr(timestamp, 'strftime'):
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
            else:
                timestamp_str = None
            
            queries.append({
                "id": doc.id,
                "query": data.get("query", ""),
                "subject": data.get("subject", ""),
                "chapter_name": data.get("chapter_name", ""),
                "timestamp": timestamp_str,
                "mode": data.get("mode", "text")
            })
        
        return queries
        
    except Exception as e:
        logger.error(f"Failed to get frequent questions for {uid}: {e}", exc_info=True)
        raise


def get_common_mistakes(uid: str) -> Dict:
    """
    Get student's common mistakes and learning patterns.
    
    Returns:
        {
            "patterns": [str, ...],
            "confusion_topics": [str, ...],
            "recommended_tasks": [str, ...]
        }
    """
    try:
        doc_ref = db.collection("student_mistakes").document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            logger.warning(f"No student_mistakes document found for UID: {uid} in get_common_mistakes.")
            return {
                "patterns": [],
                "confusion_topics": [],
                "recommended_tasks": []
            }
        
        data = doc.to_dict()
        
        return {
            "patterns": data.get("patterns", []),
            "confusion_topics": data.get("confusion_topics", []),
            "recommended_tasks": data.get("recommended_tasks", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to get common mistakes for {uid}: {e}", exc_info=True)
        raise


# ==========================================
# ADMIN DASHBOARD DATA AGGREGATION
# ==========================================

def get_class_overview() -> Dict:
    """
    Get overview of query distribution across all classes.
    
    Returns:
        {
            "class_distribution": {
                "8": 450,
                "9": 320,
                ...
            },
            "total_queries": int
        }
    """
    try:
        # Aggregate from chapter_stats
        chapter_stats_ref = db.collection("chapter_stats").stream()
        
        class_counts = defaultdict(int)
        total = 0
        
        for doc in chapter_stats_ref:
            data = doc.to_dict()
            class_num = data.get("class", 0)
            query_count = data.get("total_queries", 0)
            
            class_counts[str(class_num)] += query_count
            total += query_count
        
        return {
            "class_distribution": dict(class_counts),
            "total_queries": total
        }
        
    except Exception as e:
        logger.error(f"Failed to get class overview: {e}", exc_info=True)
        raise


def get_chapter_hotspots(class_name: str, subject: str, limit: int = 10) -> List[Dict]:
    """
    Get most queried chapters for a class and subject.
    
    Args:
        class_name: Class
        subject: Subject name
        limit: Number of hotspots to return
    
    Returns:
        List of chapter stats sorted by query count
    """
    try:
        class_int = int(class_name.replace("Class", "").replace("class", "").strip())
        
        # Query chapter_stats filtered by class and subject
        chapter_stats_ref = db.collection("chapter_stats")\
            .where("class", "==", class_int)\
            .where("subject", "==", subject.lower())\
            .limit(limit)
        
        docs = chapter_stats_ref.stream()
        
        hotspots = []
        for doc in docs:
            data = doc.to_dict()
            hotspots.append({
                "chapter_id": data.get("chapter_id"),
                "chapter_name": data.get("chapter_name"),
                "total_queries": data.get("total_queries", 0),
                "unique_students": len(data.get("unique_students", [])),
                "avg_difficulty": data.get("avg_difficulty", 0.0)
            })
        
        return hotspots
        
    except Exception as e:
        logger.error(f"Failed to get chapter hotspots for {class_name} {subject}: {e}", exc_info=True)
        raise


def get_subject_distribution() -> Dict:
    """
    Get query distribution across all subjects.
    
    Returns:
        {
            "subjects": ["science", "maths", ...],
            "counts": [450, 320, ...]
        }
    """
    try:
        # Aggregate from chapter_stats
        chapter_stats_ref = db.collection("chapter_stats").stream()
        
        subject_counts = defaultdict(int)
        
        for doc in chapter_stats_ref:
            data = doc.to_dict()
            subject = data.get("subject", "unknown")
            query_count = data.get("total_queries", 0)
            
            subject_counts[subject] += query_count
        
        # Sort by count
        sorted_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "subjects": [subj for subj, _ in sorted_subjects],
            "counts": [count for _, count in sorted_subjects]
        }
        
    except Exception as e:
        logger.error(f"Failed to get subject distribution: {e}", exc_info=True)
        raise


def get_student_performance(uid: str) -> Dict:
    """
    Get detailed performance metrics for a specific student (admin view).
    
    Returns:
        Complete student stats with additional analytics
    """
    try:
        # Get user stats
        stats_doc = db.collection("user_stats").document(uid).get()
        
        if not stats_doc.exists:
            logger.warning(f"No user_stats document found for UID: {uid} in get_student_performance.")
            return {
                "error": "Student not found"
            }
        
        stats_data = stats_doc.to_dict()
        
        # Get recent queries
        queries_ref = db.collection("user_queries")\
            .where("uid", "==", uid)\
            .order_by("timestamp", direction=firestore.Query.DESCENDING)\
            .limit(20)
        
        recent_queries = []
        for doc in queries_ref.stream():
            data = doc.to_dict()
            timestamp = data.get("timestamp")
            if timestamp and hasattr(timestamp, 'strftime'):
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
            else:
                timestamp_str = None
            
            recent_queries.append({
                "query": data.get("query", ""),
                "subject": data.get("subject", ""),
                "timestamp": timestamp_str
            })
        
        # Get mistakes
        mistakes_doc = db.collection("student_mistakes").document(uid).get()
        mistakes_data = mistakes_doc.to_dict() if mistakes_doc.exists else {}
        
        return {
            "total_queries": stats_data.get("total_queries", 0),
            "streak": stats_data.get("streak", 0),
            "subjects_count": stats_data.get("subjects_count", {}),
            "chapters_count": stats_data.get("chapters_count", {}),
            "recent_queries": recent_queries,
            "patterns": mistakes_data.get("patterns", []),
            "confusion_topics": mistakes_data.get("confusion_topics", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to get student performance for {uid}: {e}", exc_info=True)
        raise


logger.info("✅ Dashboard service loaded successfully")

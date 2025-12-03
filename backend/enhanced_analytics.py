"""
Enhanced Analytics Module for Advanced Dashboard Features
Provides topic-level tracking, weak area analysis, and personalized suggestions
"""

from google.cloud import firestore
from datetime import datetime, timezone
from typing import Optional, Dict, List
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

# Import db from firebase_init
from .firebase.firebase_init import db


# ============================================
# TOPIC-LEVEL ANALYTICS
# ============================================

def track_topic_analytics(
    uid: str,
    subject: str,
    chapter_id: int,
    chapter_name: str,
    topics: List[str],
    difficulty_score: Optional[float] = None
) -> None:
    """
    Track analytics at topic level for detailed insights.
    
    Args:
        uid: User ID
        subject: Subject name
        chapter_id: Chapter ID
        chapter_name: Chapter name  
        topics: List of topics covered in this query
        difficulty_score: Difficulty rating (0-1)
    """
    try:
        for topic in topics:
            # Document ID: {uid}_{subject}_{chapter_id}_{topic}
            topic_slug = topic.lower().replace(" ", "_")[:50]
            doc_id = f"{uid}_{subject.lower()}_{chapter_id}_{topic_slug}"
            doc_ref = db.collection("topic_analytics").document(doc_id)
            
            doc = doc_ref.get()
            
            if not doc.exists:
                # Create new topic analytics
                doc_ref.set({
                    "uid": uid,
                    "subject": subject.lower(),
                    "chapter_id": chapter_id,
                    "chapter_name": chapter_name,
                    "topic": topic,
                    "query_count": 1,
                    "last_asked": firestore.SERVER_TIMESTAMP,
                    "difficulty_scores": [difficulty_score] if difficulty_score else [],
                    "avg_difficulty": difficulty_score or 0.0,
                    "mastery_level": 10  # Start at 10%
                })
                logger.info(f"✅ Created topic analytics: {doc_id}")
            else:
                # Update existing
                current_data = doc.to_dict()
                difficulty_scores = current_data.get("difficulty_scores", [])
                if difficulty_score:
                    difficulty_scores.append(difficulty_score)
                
                avg_diff = sum(difficulty_scores) / len(difficulty_scores) if difficulty_scores else 0.0
                
                # Calculate mastery: more queries + lower difficulty = higher mastery
                query_count = current_data.get("query_count", 0) + 1
                mastery = min(100, (query_count * 10) - (avg_diff * 50))
                
                doc_ref.update({
                    "query_count": firestore.Increment(1),
                    "last_asked": firestore.SERVER_TIMESTAMP,
                    "difficulty_scores": difficulty_scores,
                    "avg_difficulty": avg_diff,
                    "mastery_level": mastery
                })
                logger.info(f"✅ Updated topic analytics: {doc_id}")
                
    except Exception as e:
        logger.error(f"❌ Failed to track topic analytics: {e}")


# ============================================
# FREQUENT QUESTIONS TRACKING
# ============================================

def update_frequent_questions(
    uid: str,
    query: str,
    chapter_name: str,
    subject: str
) -> None:
    """
    Maintain a list of user's frequently asked questions.
    
    Args:
        uid: User ID
        query: The question asked
        chapter_name: Chapter name
        subject: Subject name
    """
    try:
        doc_ref = db.collection("frequent_questions").document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            # Create new
            doc_ref.set({
                "questions": [{
                    "query": query,
                    "count": 1,
                    "last_asked": datetime.now(timezone.utc),  # Can't use SERVER_TIMESTAMP in arrays
                    "chapter": chapter_name,
                    "subject": subject
                }],
                "last_updated": firestore.SERVER_TIMESTAMP
            })
        else:
            # Update existing
            data = doc.to_dict()
            questions = data.get("questions", [])
            
            # Check if question exists (fuzzy match on first 50 chars)
            query_key = query[:50].lower()
            found = False
            
            for q in questions:
                if q["query"][:50].lower() == query_key:
                    q["count"] += 1
                    q["last_asked"] = datetime.now(timezone.utc)
                    found = True
                    break
            
            if not found:
                questions.append({
                    "query": query,
                    "count": 1,
                    "last_asked": datetime.now(timezone.utc),
                    "chapter": chapter_name,
                    "subject": subject
                })
            
            # Keep only top 20, sorted by count
            questions.sort(key=lambda x: x["count"], reverse=True)
            questions = questions[:20]
            
            doc_ref.update({
                "questions": questions,
                "last_updated": firestore.SERVER_TIMESTAMP
            })
            
        logger.info(f"✅ Updated frequent questions for {uid}")
        
    except Exception as e:
        logger.error(f"❌ Failed to update frequent questions: {e}")


# ============================================
# WEAK AREA ANALYSIS
# ============================================

def analyze_weak_areas(uid: str) -> Dict:
    """
    AI-powered analysis of student's weak areas based on query patterns.
    
    Criteria:
    - Topics with high query count but low mastery
    - Chapters with repeated questions
    - Subjects needing more practice
    
    Returns:
        Dictionary of weak areas with suggestions
    """
    try:
        weak_areas = {
            "subjects": {},
            "chapters": [],
            "topics": [],
            "suggestions": [],
            "last_analysis": datetime.now(timezone.utc)
        }
        
        # Analyze topic analytics
        topic_docs = db.collection("topic_analytics").where("uid", "==", uid).stream()
        
        topic_issues = []
        for doc in topic_docs:
            data = doc.to_dict()
            query_count = data.get("query_count", 0)
            mastery = data.get("mastery_level", 0)
            
            # Weak topic: >3 queries but <40% mastery
            if query_count >= 3 and mastery < 40:
                topic_issues.append({
                    "subject": data.get("subject"),
                    "chapter": data.get("chapter_name"),
                    "topic": data.get("topic"),
                    "queries": query_count,
                    "mastery": mastery
                })
        
        # Sort by query count (more queries = bigger issue)
        topic_issues.sort(key=lambda x: x["queries"], reverse=True)
        weak_areas["topics"] = topic_issues[:5]  # Top 5
        
        # Group by subject
        subject_groups = defaultdict(list)
        for issue in topic_issues:
            subject_groups[issue["subject"]].append(issue)
        
        for subject, issues in subject_groups.items():
            weak_areas["subjects"][subject] = {
                "topic_count": len(issues),
                "chapters": list(set([i["chapter"] for i in issues])),
                "avg_mastery": sum([i["mastery"] for i in issues]) / len(issues),
                "reason": f"Multiple weak topics detected ({len(issues)} topics)"
            }
        
        # Generate suggestions
        if topic_issues:
            for issue in topic_issues[:3]:
                weak_areas["suggestions"].append({
                    "type": "practice",
                    "priority": "high",
                    "message": f"Review {issue['topic']} in {issue['chapter']} - asked {issue['queries']} times",
                    "action": f"Practice problems on {issue['topic']}"
                })
        
        # Save to Firestore
        db.collection("weak_areas").document(uid).set(weak_areas)
        logger.info(f"✅ Analyzed weak areas for {uid}: {len(topic_issues)} issues found")
        
        return weak_areas
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze weak areas: {e}")
        return {}


# ============================================
# PERSONALIZED SUGGESTIONS
# ============================================

def generate_suggestions(uid: str) -> List[Dict]:
    """
    Generate personalized study suggestions based on user behavior.
    
    Returns:
        List of actionable suggestions with priorities
    """
    try:
        suggestions = []
        
        # Get user stats
        user_stats = db.collection("user_stats").document(uid).get()
        if not user_stats.exists:
            return []
        
        stats = user_stats.to_dict()
        total_queries = stats.get("total_queries", 0)
        streak = stats.get("streak", 0)
        subjects_count = stats.get("subjects_count", {})
        
        # Streak encouragement
        if streak >= 3:
            suggestions.append({
                "type": "encouragement",
                "priority": "medium",
                "icon": "🔥",
                "message": f"Amazing! You're on a {streak}-day streak. Keep it up!",
                "action": "Continue your daily practice"
            })
        elif streak == 0 and total_queries > 5:
            suggestions.append({
                "type": "reminder",
                "priority": "medium",
                "icon": "📅",
                "message": "You haven't studied today. Build your streak!",
                "action": "Ask a question to continue learning"
            })
        
        # Subject balance
        if len(subjects_count) == 1:
            current_subject = list(subjects_count.keys())[0]
            suggestions.append({
                "type": "diversity",
                "priority": "low",
                "icon": "📚",
                "message": f"You're focusing on {current_subject}. Try exploring other subjects!",
                "action": "Practice Maths or other subjects"
            })
        
        # Milestone celebrations
        if total_queries in [10, 25, 50, 100]:
            suggestions.append({
                "type": "celebration",
                "priority": "high",
                "icon": "🎉",
                "message": f"Congratulations! You've asked {total_queries} questions!",
                "action": "Keep up the excellent work"
            })
        
        # Get frequent questions
        freq_doc = db.collection("frequent_questions").document(uid).get()
        if freq_doc.exists:
            freq_data = freq_doc.to_dict()
            questions = freq_data.get("questions", [])
            if questions and questions[0]["count"] >= 3:
                top_q = questions[0]
                suggestions.append({
                    "type": "review",
                    "priority": "high",
                    "icon": "🔍",
                    "message": f"You've asked about '{top_q['query'][:50]}...' {top_q['count']} times",
                    "action": f"Review {top_q['chapter']} notes carefully"
                })
        
        return suggestions
        
    except Exception as e:
        logger.error(f"❌ Failed to generate suggestions: {e}")
        return []


# ============================================
# ADMIN-SPECIFIC ANALYTICS
# ============================================

def get_student_detailed_report(uid: str) -> Dict:
    """
    Generate comprehensive student performance report for admin view.
    
    Returns complete analytics including:
    - Basic stats (queries, streak, subjects)
    - Chapter breakdown
    - Topic mastery levels
    - Weak areas
    - Recent activity
    - Improvement trends
    """
    try:
        report = {
            "uid": uid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "basic_stats": {},
            "chapter_breakdown": [],
            "topic_mastery": [],
            "weak_areas": [],
            "recent_queries": [],
            "suggestions": []
        }
        
        # Basic stats
        user_stats_doc = db.collection("user_stats").document(uid).get()
        if user_stats_doc.exists:
            report["basic_stats"] = user_stats_doc.to_dict()
        
        # Chapter breakdown
        queries = db.collection("user_queries").where("uid", "==", uid).stream()
        chapter_counts = Counter()
        subject_counts = Counter()
        
        for q in queries:
            data = q.to_dict()
            chapter = data.get("chapter_name", "Unknown")
            subject = data.get("subject", "unknown")
            chapter_counts[chapter] += 1
            subject_counts[subject] += 1
        
        report["chapter_breakdown"] = [
            {"chapter": k, "queries": v} 
            for k, v in chapter_counts.most_common(10)
        ]
        
        report["subject_distribution"] = dict(subject_counts)
        
        # Topic mastery
        topics = db.collection("topic_analytics").where("uid", "==", uid).stream()
        topic_list = []
        for t in topics:
            data = t.to_dict()
            topic_list.append({
                "topic": data.get("topic"),
                "subject": data.get("subject"),
                "chapter": data.get("chapter_name"),
                "mastery": data.get("mastery_level", 0),
                "queries": data.get("query_count", 0)
            })
        
        topic_list.sort(key=lambda x: x["mastery"])
        report["topic_mastery"] = topic_list
        
        # Weak areas
        weak_doc = db.collection("weak_areas").document(uid).get()
        if weak_doc.exists:
            report["weak_areas"] = weak_doc.to_dict().get("topics", [])
        
        # Recent queries
        recent = db.collection("user_queries")\
            .where("uid", "==", uid)\
            .order_by("timestamp", direction=firestore.Query.DESCENDING)\
            .limit(10)\
            .stream()
        
        for r in recent:
            data = r.to_dict()
            report["recent_queries"].append({
                "query": data.get("query"),
                "subject": data.get("subject"),
                "chapter": data.get("chapter_name"),
                "timestamp": data.get("timestamp").isoformat() if data.get("timestamp") else None
            })
        
        logger.info(f"✅ Generated detailed report for {uid}")
        return report
        
    except Exception as e:
        logger.error(f"❌ Failed to generate student report: {e}")
        return {}


logger.info("✅ Enhanced analytics module loaded successfully")

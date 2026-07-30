import re
import json
import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import firestore

from backend.app.services.analytics import analytics_service
from backend.app.services.analytics import dashboard_service
from backend.app.services.analytics import enhanced_dashboard_service
from backend.app.services.analytics import enhanced_analytics
from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.core.firebase.firebase_init import db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/dashboard/summary", tags=["Dashboard"])
async def get_dashboard_summary_endpoint(uid: str = Query(...)):
    """
    Get summary statistics for student dashboard.
    Uses user_queries as single source of truth for all analytics.
    """
    try:
        logger.info(f"[DASHBOARD SUMMARY] Rebuilding analytics from user_queries for uid: {uid}")
        summary = analytics_service.rebuild_user_analytics_from_queries(uid)
        
        return {
            "total_queries": summary["total_queries"],
            "subjects_explored": summary["subjects_explored"],
            "subjects_count": summary["subjects_count"],
            "weekly_activity": summary["weekly_activity"],
            "daily_activity": summary["daily_activity"],
            "last_active": summary["last_active"],
            "streak": summary["streak"],
            "longest_streak": summary["longest_streak"]
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/weekly", tags=["Dashboard"])
async def get_weekly_activity_endpoint(
    uid: str = Query(...),
    weeks: int = Query(4, ge=1, le=12)
):
    """
    Get weekly activity data for charts.
    """
    try:
        activity = dashboard_service.get_weekly_activity(uid, weeks)
        return activity
    except Exception as e:
        logger.error(f"Failed to get weekly activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/strength-weakness", tags=["Dashboard"])
async def get_strength_weakness_endpoint(uid: str = Query(...)):
    """
    Analyze student's strengths and weaknesses by subject.
    """
    try:
        analysis = dashboard_service.get_strength_weakness(uid)
        return analysis
    except Exception as e:
        logger.error(f"Failed to get strength/weakness analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/frequent-questions", tags=["Dashboard"])
async def get_frequent_questions_endpoint(
    uid: str = Query(...),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get recent frequent questions from the user.
    """
    try:
        questions = dashboard_service.get_frequent_questions(uid, limit)
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        logger.error(f"Failed to get frequent questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/mistakes", tags=["Dashboard"])
async def get_common_mistakes_endpoint(uid: str = Query(...)):
    """
    Get student's common mistakes and learning patterns.
    """
    try:
        mistakes = dashboard_service.get_common_mistakes(uid)
        return mistakes
    except Exception as e:
        logger.error(f"Failed to get common mistakes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/enhanced")
def enhanced_dashboard(uid: str, class_name: str = Query(None)):
    """
    Returns the full analytics dataset required by enhanced-dashboard.html.
    """
    try:
        return enhanced_dashboard_service.get_enhanced_dashboard_data(uid, class_name)
    except Exception as e:
        logger.error(f"Error in enhanced_dashboard endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/topic-breakdown")
def get_topic_breakdown(uid: str, subject: str = None, chapter_id: str = None):
    """Get topic-level analytics with optional filters"""
    try:
        query = db.collection("topic_analytics").where("uid", "==", uid)
        if subject and subject != "All Subjects":
            query = query.where("subject", "==", subject.lower())
            
        docs = query.stream()
        topics = []
        for doc in docs:
            topics.append(doc.to_dict())
            
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error fetching topic breakdown: {e}")
        return {"topics": []}


@router.get("/api/dashboard/weak-areas")
def get_weak_areas(uid: str):
    """Get AI-analyzed weak areas"""
    try:
        return enhanced_analytics.analyze_weak_areas(uid)
    except Exception as e:
        logger.error(f"Error fetching weak areas: {e}")
        return {}


@router.get("/api/dashboard/suggestions")
def get_suggestions(uid: str):
    """Get personalized study suggestions"""
    try:
        return {"suggestions": enhanced_analytics.generate_suggestions(uid)}
    except Exception as e:
        logger.error(f"Error fetching suggestions: {e}")
        return {"suggestions": []}


@router.get("/api/admin/student-report")
def get_student_report(uid: str):
    """Get comprehensive student report for admin"""
    try:
        return enhanced_analytics.get_student_detailed_report(uid)
    except Exception as e:
        logger.error(f"Error generating student report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/ai-feedback", tags=["Dashboard"])
async def get_ai_feedback_endpoint(uid: str = Query(...)):
    """
    Get AI-powered student feedback and insights.
    """
    try:
        feedback = enhanced_dashboard_service.generate_student_feedback(uid)
        return feedback
    except Exception as e:
        logger.error(f"Failed to generate AI feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/my-chapter-hotspots", tags=["Dashboard"])
async def get_my_chapter_hotspots_endpoint(
    uid: str = Query(...),
    limit: int = Query(5, ge=1, le=20)
):
    """
    Get chapters this student asks about most.
    """
    try:
        hotspots = enhanced_dashboard_service.get_chapter_hotspots_for_student(uid, limit)
        return {"hotspots": hotspots, "total": len(hotspots)}
    except Exception as e:
        logger.error(f"Failed to get chapter hotspots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dashboard/topic-clusters", tags=["Dashboard"])
async def get_topic_clusters(
    uid: str = Query(...),
    subject: str = Query(...),
    chapter_id: int = Query(...),
    chapter_name: str = Query(None)
):
    """
    Generate topic clusters from ALL user queries using LLM.
    """
    try:
        logger.info(f"[TOPIC CLUSTERS] ========== STARTING ==========")
        logger.info(f"[TOPIC CLUSTERS] Request: uid={uid}, subject={subject}, chapter_name={chapter_name}, chapter_id={chapter_id}")
        
        logger.info(f"[TOPIC CLUSTERS] Fetching all queries for subject={subject}")
        all_queries_ref = db.collection("user_queries")\
            .where("uid", "==", uid)\
            .where("subject", "==", subject.lower())\
            .stream()
        
        query_texts = []
        actual_chapter_name = chapter_name or "Unknown Chapter"
        
        for doc in all_queries_ref:
            data = doc.to_dict()
            doc_chapter_name = data.get("chapter_name", "")
            
            if chapter_name and doc_chapter_name.lower() == chapter_name.lower():
                query = data.get("query", "")
                if query:
                    query_texts.append(query)
                if not actual_chapter_name or actual_chapter_name == "Unknown Chapter":
                    actual_chapter_name = doc_chapter_name
        
        logger.info(f"[TOPIC CLUSTERS] Found {len(query_texts)} queries for chapter '{actual_chapter_name}'")
        
        if not query_texts:
            logger.warning(f"[TOPIC CLUSTERS] No queries found for chapter_name='{chapter_name}'")
            return {"chapter": actual_chapter_name, "topics": []}
        
        logger.info(f"[TOPIC CLUSTERS] Calling LLM with {len(query_texts)} queries")
        
        from backend.app.prompts.templates import GET_TOPIC_CLUSTERS_PROMPT
        prompt = GET_TOPIC_CLUSTERS_PROMPT.format(
            query_count=len(query_texts),
            queries_bullet_list=chr(10).join(f"- {q}" for q in query_texts[:50])
        )

        response = qdrant.openai_client.models.generate_content(
            model=qdrant.generation_model_name,
            contents=prompt
        )
        result_text = response.text.strip()
        
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        topics_data = json.loads(result_text)
        
        logger.info(f"[TOPIC CLUSTERS] Generated {len(topics_data.get('topics', []))} topics, saving to Firestore")
        
        for topic in topics_data.get('topics', []):
            topic_name = topic.get('topic_name', 'Unknown Topic')
            slug = re.sub(r'[^a-z0-9]+', '_', topic_name.lower()).strip('_')
            doc_id = f"{uid}_{subject}_{chapter_id}_{slug}"
            
            topic_doc_ref = db.collection('topic_analytics').document(doc_id)
            topic_doc_ref.set({
                'uid': uid,
                'subject': subject,
                'chapter_id': chapter_id,
                'chapter_name': chapter_name,
                'topic': topic_name,
                'query_count': topic.get('query_count', 0),
                'mastery_level': topic.get('mastery_level', 0.0),
                'difficulty_score': topic.get('difficulty_score', 0.0),
                'example_queries': topic.get('example_queries', []),
                'last_asked': firestore.SERVER_TIMESTAMP
            }, merge=True)
        
        logger.info(f"[TOPIC CLUSTERS] Saved {len(topics_data.get('topics', []))} topics to Firestore")
        
        formatted_topics = []
        for topic in topics_data.get('topics', []):
            formatted_topics.append({
                'topic': topic.get('topic_name', ''),
                'query_count': topic.get('query_count', 0),
                'mastery_level': topic.get('mastery_level', 0.0),
                'difficulty': topic.get('difficulty_score', 0.0),
                'example_queries': topic.get('example_queries', [])
            })
        
        return {
            "chapter": chapter_name,
            "topics": formatted_topics
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"[TOPIC CLUSTERS] Failed to parse LLM response: {e}")
        return {"chapter": "Unknown Chapter", "topics": []}
    except Exception as e:
        logger.error(f"[TOPIC CLUSTERS] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/analytics/initialize", tags=["Analytics"])
async def initialize_user_analytics(uid: str = Query(...)):
    """
    Initialize or repair user analytics document.
    """
    try:
        from datetime import datetime
        logger.info(f"[ANALYTICS INIT] Initializing analytics for uid: {uid}")
        
        doc_ref = db.collection('user_analytics').document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            logger.info(f"[ANALYTICS INIT] Document already exists for {uid}")
            return {"message": "Analytics document already exists", "data": doc.to_dict()}
        
        logger.info(f"[ANALYTICS INIT] Fetching query history for {uid}")
        queries_ref = db.collection('user_query_details')\
            .where('uid', '==', uid)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .stream()
        
        queries = list(queries_ref)
        query_count = len(queries)
        
        if query_count == 0:
            logger.info(f"[ANALYTICS INIT] No queries found, creating empty document")
            initial_data = {
                'total_queries_all_time': 0,
                'total_subjects_explored': 0,
                'current_streak': 0,
                'longest_streak': 0,
                'last_activity_date': None,
                'daily_stats': {},
                'weekly_stats': {},
                'subjects_set': [],
                'created_at': datetime.now().isoformat()
            }
            doc_ref.set(initial_data)
            return {"message": "Empty analytics document created", "queries_found": 0}
        
        logger.info(f"[ANALYTICS INIT] Backfilling analytics from {query_count} queries")
        
        daily_stats = {}
        weekly_stats = {}
        subjects_set = set()
        
        for query_doc in queries:
            query_data = query_doc.to_dict()
            timestamp = query_data.get('timestamp')
            subject = query_data.get('subject', '').lower()
            chapter = query_data.get('chapter_name', 'Unknown')
            
            if not timestamp:
                continue
                
            if hasattr(timestamp, 'strftime'):
                date_str = timestamp.strftime('%Y-%m-%d')
                week_str = timestamp.strftime('%Y-W%W')
            else:
                continue
            
            if subject:
                subjects_set.add(subject)
            
            if date_str not in daily_stats:
                daily_stats[date_str] = {
                    'queries_count': 0,
                    'subjects': [],
                    'chapters': []
                }
            
            daily_stats[date_str]['queries_count'] += 1
            if subject and subject not in daily_stats[date_str]['subjects']:
                daily_stats[date_str]['subjects'].append(subject)
            if chapter and chapter not in daily_stats[date_str]['chapters']:
                daily_stats[date_str]['chapters'].append(chapter)
            
            if week_str not in weekly_stats:
                weekly_stats[week_str] = {
                    'queries_count': 0,
                    'subjects': [],
                    'active_days': []
                }
            
            weekly_stats[week_str]['queries_count'] += 1
            if subject and subject not in weekly_stats[week_str]['subjects']:
                weekly_stats[week_str]['subjects'].append(subject)
            if date_str not in weekly_stats[week_str]['active_days']:
                weekly_stats[week_str]['active_days'].append(date_str)
        
        sorted_dates = sorted(daily_stats.keys(), reverse=True)
        current_streak = 0
        longest_streak = 0
        temp_streak = 0
        
        if sorted_dates:
            from datetime import datetime
            today = datetime.now().date()
            
            for i, date_str in enumerate(sorted_dates):
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                if i == 0:
                    diff = (today - date).days
                    if diff <= 1:
                        temp_streak = 1
                        if i + 1 < len(sorted_dates):
                            next_date = datetime.strptime(sorted_dates[i + 1], '%Y-%m-%d').date()
                            if (date - next_date).days == 1:
                                continue
                    else:
                        break
                else:
                    prev_date = datetime.strptime(sorted_dates[i - 1], '%Y-%m-%d').date()
                    if (prev_date - date).days == 1:
                        temp_streak += 1
                    else:
                        if temp_streak > longest_streak:
                            longest_streak = temp_streak
                        temp_streak = 0
                        break
            
            current_streak = temp_streak
            if temp_streak > longest_streak:
                longest_streak = temp_streak
        
        analytics_data = {
            'total_queries_all_time': query_count,
            'total_subjects_explored': len(subjects_set),
            'current_streak': current_streak,
            'longest_streak': longest_streak,
            'last_activity_date': sorted_dates[0] if sorted_dates else None,
            'daily_stats': daily_stats,
            'weekly_stats': weekly_stats,
            'subjects_set': list(subjects_set),
            'created_at': datetime.now().isoformat(),
            'backfilled': True,
            'backfill_date': datetime.now().isoformat()
        }
        
        doc_ref.set(analytics_data)
        return {
            "message": "Analytics successfully initialized from query history",
            "queries_backfilled": query_count,
            "subjects_found": len(subjects_set),
            "current_streak": current_streak,
            "longest_streak": longest_streak
        }
    except Exception as e:
        logger.error(f"[ANALYTICS INIT] Failed to initialize analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/summary", tags=["Analytics"])
async def get_analytics_summary(uid: str = Query(...)):
    """
    Get cumulative analytics summary for dashboard main stats.
    """
    try:
        doc_ref = db.collection('user_analytics').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            return {
                "total_queries_all_time": 0,
                "current_streak": 0,
                "total_subjects_explored": 0,
                "this_week_total": 0,
                "longest_streak": 0
            }
        
        data = doc.to_dict()
        
        from datetime import datetime
        current_week = datetime.now().strftime('%Y-W%W')
        weekly_stats = data.get('weekly_stats', {})
        this_week_total = weekly_stats.get(current_week, {}).get('queries_count', 0)
        
        return {
            "total_queries_all_time": data.get('total_queries_all_time', 0),
            "current_streak": data.get('current_streak', 0),
            "longest_streak": data.get('longest_streak', 0),
            "total_subjects_explored": data.get('total_subjects_explored', 0),
            "this_week_total": this_week_total
        }
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/details/{stat_type}", tags=["Analytics"])
async def get_stat_details(stat_type: str, uid: str = Query(...)):
    """
    Get detailed breakdown for a specific stat card.
    """
    try:
        logger.info(f"[ANALYTICS DETAILS] Rebuilding {stat_type} from user_queries for uid: {uid}")
        summary = analytics_service.rebuild_user_analytics_from_queries(uid)
        
        return {
            "total_queries": summary["total_queries"],
            "subjects_count": summary["subjects_count"],
            "subjects_explored": summary["subjects_explored"],
            "daily_activity": summary["daily_activity"],
            "weekly_activity": summary["weekly_activity"],
            "streak": summary["streak"],
            "longest_streak": summary["longest_streak"],
            "last_active": summary["last_active"]
        }
    except Exception as e:
        logger.error(f"Failed to get stat details for {stat_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/admin/class-overview", tags=["Admin Dashboard"])
async def get_class_overview_endpoint():
    """
    Get overview of query distribution across all classes.
    """
    try:
        overview = dashboard_service.get_class_overview()
        return overview
    except Exception as e:
        logger.error(f"Failed to get class overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/admin/chapter-hotspots", tags=["Admin Dashboard"])
async def get_chapter_hotspots_endpoint(
    class_name: str = Query(...),
    subject: str = Query(...),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get most queried chapters for a class and subject.
    """
    try:
        hotspots = dashboard_service.get_chapter_hotspots(class_name, subject, limit)
        return {"hotspots": hotspots, "total": len(hotspots)}
    except Exception as e:
        logger.error(f"Failed to get chapter hotspots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/admin/subject-distribution", tags=["Admin Dashboard"])
async def get_subject_distribution_endpoint():
    """
    Get query distribution across all subjects.
    """
    try:
        distribution = dashboard_service.get_subject_distribution()
        return distribution
    except Exception as e:
        logger.error(f"Failed to get subject distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/admin/student-performance", tags=["Admin Dashboard"])
async def get_student_performance_endpoint(uid: str = Query(...)):
    """
    Get detailed performance metrics for a specific student (admin view).
    """
    try:
        performance = dashboard_service.get_student_performance(uid)
        return performance
    except Exception as e:
        logger.error(f"Failed to get student performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


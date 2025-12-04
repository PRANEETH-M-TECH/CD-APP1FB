"""
Diagnostic Script: Check User Queries Structure
This script helps identify why topic clusters aren't showing for existing data
"""

from google.cloud import firestore
from backend.firebase.firebase_init import db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_user_queries(uid="Y0Ql0s2NzJVfZgfniPGi63Eomme2", subject="social"):
    """
    Check the structure of user_queries to see what fields exist
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DIAGNOSING USER QUERIES FOR UID: {uid}")
    logger.info(f"{'='*60}\n")
    
    # Fetch all queries for this user
    queries_ref = db.collection("user_queries")\
        .where("uid", "==", uid)\
        .limit(10)
    
    queries = list(queries_ref.stream())
    
    if not queries:
        logger.warning(f"❌ No queries found for uid={uid}")
        return
    
    logger.info(f"✅ Found {len(queries)} queries (showing first 10)\n")
    
    # Analyze structure
    for i, doc in enumerate(queries, 1):
        data = doc.to_dict()
        logger.info(f"Query {i}:")
        logger.info(f"  ID: {doc.id}")
        logger.info(f"  Query: {data.get('query', 'N/A')[:60]}...")
        logger.info(f"  Subject: {data.get('subject', 'MISSING')}")
        logger.info(f"  Chapter ID: {data.get('chapter_id', 'MISSING')}")
        logger.info(f"  Chapter Name: {data.get('chapter_name', 'MISSING')}")
        logger.info(f"  Timestamp: {data.get('timestamp', 'MISSING')}")
        logger.info(f"  All Fields: {list(data.keys())}")
        logger.info("")
    
    # Check chapter_id distribution
    logger.info(f"\n{'='*60}")
    logger.info("CHAPTER ID ANALYSIS")
    logger.info(f"{'='*60}\n")
    
    chapter_counts = {}
    for doc in queries:
        data = doc.to_dict()
        chapter_id = data.get('chapter_id', 'MISSING')
        chapter_name = data.get('chapter_name', 'Unknown')
        key = f"ID:{chapter_id} - {chapter_name}"
        chapter_counts[key] = chapter_counts.get(key, 0) + 1
    
    for chapter, count in sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {chapter}: {count} queries")
    
    # Test a specific query
    logger.info(f"\n{'='*60}")
    logger.info("TESTING TOPIC CLUSTERS QUERY")
    logger.info(f"{'='*60}\n")
    
    # Try to fetch with a specific chapter_id
    if queries:
        first_query = queries[0].to_dict()
        test_chapter_id = first_query.get('chapter_id')
        test_subject = first_query.get('subject', subject)
        
        if test_chapter_id:
            logger.info(f"Testing query with:")
            logger.info(f"  uid = {uid}")
            logger.info(f"  subject = {test_subject}")
            logger.info(f"  chapter_id = {test_chapter_id}")
            
            test_queries = db.collection("user_queries")\
                .where("uid", "==", uid)\
                .where("subject", "==", test_subject.lower())\
                .where("chapter_id", "==", test_chapter_id)\
                .stream()
            
            test_results = list(test_queries)
            logger.info(f"\n  ✅ Found {len(test_results)} queries matching these filters\n")
            
            if test_results:
                logger.info("Sample queries that would be clustered:")
                for doc in test_results[:5]:
                    q = doc.to_dict().get('query', '')[:80]
                    logger.info(f"    - {q}")
        else:
            logger.warning("  ❌ No chapter_id found in first query!")

if __name__ == "__main__":
    diagnose_user_queries()

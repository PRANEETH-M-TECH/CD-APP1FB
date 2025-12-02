
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from backend import analytics_service
    print("Successfully imported analytics_service")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_write():
    uid = "test_user_123"
    subject = "science"
    class_name = "8"
    chapter_id = 1
    chapter_name = "Test Chapter"
    
    print(f"Attempting to update stats for {uid}...")
    
    try:
        # 1. Log Query
        doc_id = analytics_service.log_query(
            uid=uid,
            class_name=class_name,
            subject=subject,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            query="Test query",
            reformulated_query="Test query",
            mode="text",
            llm_action="test",
            answer_length=100
        )
        print(f"Logged query: {doc_id}")
        
        # 2. Update User Stats
        analytics_service.update_user_stats(
            uid=uid,
            subject=subject,
            chapter_id=chapter_id,
            class_name=class_name
        )
        print("Updated user stats")
        
        # 3. Update Chapter Stats
        analytics_service.update_chapter_stats(
            class_name=class_name,
            subject=subject,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            uid=uid
        )
        print("Updated chapter stats")
        
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_write()

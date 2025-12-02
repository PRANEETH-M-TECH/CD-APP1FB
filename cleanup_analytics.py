
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def cleanup_session_based_data():
    """
    Removes data logged with session IDs instead of Firebase UIDs.
    This includes UIDs that contain underscores (session ID format).
    """
    print("=" * 60)
    print("ANALYTICS DATA CLEANUP SCRIPT")
    print("=" * 60)
    
    # 1. Clean user_stats
    print("\n1. Cleaning user_stats collection...")
    user_stats = db.collection("user_stats").stream()
    cleaned_stats = 0
    for doc in user_stats:
        if "_" in doc.id:  # Session IDs contain underscores
            doc.reference.delete()
            print(f"   Deleted user_stats: {doc.id}")
            cleaned_stats += 1
    print(f"   ✅ Removed {cleaned_stats} session-based user stats")
    
    # 2. Clean chapter_stats (keep these as they aggregate across users)
    # We don't delete chapter_stats because they're useful aggregate data
    print("\n2. Keeping chapter_stats (aggregate data is still useful)")
    
    # 3. Clean user_queries
    print("\n3. Cleaning user_queries collection...")
    queries = db.collection("user_queries").stream()
    cleaned_queries = 0
    for doc in queries:
        data = doc.to_dict()
        uid = data.get("uid", "")
        if "_" in uid:  # Session IDs contain underscores
            doc.reference.delete()
            print(f"   Deleted query: {doc.id} (UID: {uid[:30]}...)")
            cleaned_queries += 1
    print(f"   ✅ Removed {cleaned_queries} session-based queries")
    
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE!")
    print(f"Total removed: {cleaned_stats + cleaned_queries} documents")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Restart your server")
    print("2. Login as a student")
    print("3. Ask a NEW question")
    print("4. Check your dashboard - it should populate with real data")
    print("=" * 60)

if __name__ == "__main__":
    response = input("This will DELETE all session-based analytics data. Continue? (yes/no): ")
    if response.lower() == "yes":
        cleanup_session_based_data()
    else:
        print("Cleanup cancelled.")

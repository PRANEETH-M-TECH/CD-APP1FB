
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def check_analytics():
    print("--- Checking User Stats ---")
    users = db.collection("user_stats").stream()
    user_count = 0
    for doc in users:
        print(f"User: {doc.id} => {doc.to_dict()}")
        user_count += 1
    print(f"Total User Stats: {user_count}")

    print("\n--- Checking Chapter Stats ---")
    chapters = db.collection("chapter_stats").stream()
    chapter_count = 0
    for doc in chapters:
        print(f"Chapter: {doc.id} => {doc.to_dict()}")
        chapter_count += 1
    print(f"Total Chapter Stats: {chapter_count}")

    print("\n--- Checking User Queries (Last 5) ---")
    queries = db.collection("user_queries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
    for doc in queries:
        data = doc.to_dict()
        print(f"Query: {doc.id} | UID: {data.get('uid')} | Query: {data.get('query')}")

if __name__ == "__main__":
    check_analytics()

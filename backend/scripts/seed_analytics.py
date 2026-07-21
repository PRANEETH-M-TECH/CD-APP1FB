import os
import sys
import random
import datetime
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials

# Add backend directory to path so we can import modules if needed
# But for this standalone script, we'll try to keep it self-contained or use the existing init
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Firebase (Standalone)
# We need to initialize it here because the app might not be running
# or we want to run this independently.
try:
    # Try to get the service account path
    current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/scripts
    backend_dir = os.path.dirname(current_dir) # backend
    project_root = os.path.dirname(backend_dir) # root
    sa_path = os.path.join(project_root, "serviceAccountKey.json")
    
    if not os.path.exists(sa_path):
        print(f"❌ Service account key not found at {sa_path}")
        sys.exit(1)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    
    db = firestore.Client()
    print("✅ Connected to Firestore")

except Exception as e:
    print(f"❌ Failed to initialize Firebase: {e}")
    sys.exit(1)

# --- CONFIG ---
NUM_USERS = 1
DAYS_OF_HISTORY = 5
CLASSES = [8, 9, 10]
SUBJECTS = ["science", "social", "maths", "english"]
CHAPTERS = {
    "science": ["Crop Production", "Microorganisms", "Force and Pressure", "Light", "Sound"],
    "social": ["Resources", "Agriculture", "Industries", "Human Resources"],
    "maths": ["Rational Numbers", "Linear Equations", "Quadrilaterals", "Data Handling"],
    "english": ["The Best Christmas Present", "The Ant and the Cricket", "Geography Lesson"]
}

def seed_data():
    print(f"🌱 Starting data seed...")
    
    # 1. Create Users
    users = []
    for i in range(NUM_USERS):
        uid = f"test_student_{i+1}"
        users.append(uid)
        
        user_data = {
            "uid": uid,
            "name": f"Student {i+1}",
            "email": f"student{i+1}@example.com",
            "role": "student",
            "class": random.choice(CLASSES),
            "createdAt": datetime.datetime.now() - datetime.timedelta(days=random.randint(10, 60))
        }
        
        # Save to 'users' collection (optional, but good for admin)
        db.collection("users").document(uid).set(user_data)
        print(f"  Created user: {uid}")

    # 2. Generate History & Stats
    for uid in users:
        print(f"  Generating data for {uid}...")
        
        total_queries = 0
        subjects_count = {s: 0 for s in SUBJECTS}
        weekly_activity = {}
        
        # Generate queries for the last 30 days
        for day_offset in range(DAYS_OF_HISTORY):
            date = datetime.datetime.now() - datetime.timedelta(days=day_offset)
            date_str = date.strftime("%Y-%m-%d")
            
            # Randomly decide if user was active this day (70% chance)
            if random.random() > 0.3:
                num_queries = random.randint(1, 10)
                weekly_activity[date_str] = num_queries
                total_queries += num_queries
                
                for _ in range(num_queries):
                    subject = random.choice(SUBJECTS)
                    subjects_count[subject] += 1
                    chapter = random.choice(CHAPTERS[subject])
                    
                    # Log query
                    query_data = {
                        "uid": uid,
                        "class": 8, # Simplified
                        "subject": subject,
                        "chapter_name": chapter,
                        "query": f"Explain {chapter} concept {random.randint(1, 100)}",
                        "reformulated_query": f"Explain {chapter} concept {random.randint(1, 100)} detailed",
                        "mode": "text",
                        "timestamp": date + datetime.timedelta(hours=random.randint(9, 20)),
                        "answer_length": random.randint(100, 500)
                    }
                    db.collection("user_queries").add(query_data)
                    
                    # Update chapter stats (simplified)
                    chapter_id = f"8_{subject}_{chapter.replace(' ', '_')}"
                    db.collection("chapter_stats").document(chapter_id).set({
                        "class": 8,
                        "subject": subject,
                        "chapter_name": chapter,
                        "total_queries": firestore.Increment(1),
                        "unique_students": firestore.ArrayUnion([uid])
                    }, merge=True)

        # 3. Create User Stats
        streak = random.randint(1, 15)
        last_active = datetime.datetime.now()
        
        stats_data = {
            "total_queries": total_queries,
            "streak": streak,
            "last_active": last_active,
            "subjects_count": subjects_count,
            "weekly_activity": weekly_activity
        }
        db.collection("user_stats").document(uid).set(stats_data)
        
        # 4. Create Mistakes/Patterns
        mistakes_data = {
            "patterns": ["Confuses similar terms", "Forgets units in physics"],
            "confusion_topics": ["Force vs Pressure", "Photosynthesis"],
            "recommended_tasks": ["Review Chapter 3", "Practice numericals"]
        }
        db.collection("student_mistakes").document(uid).set(mistakes_data)
        
        # 5. Create Notes
        notes_data = {
            "notes": [
                {"title": "Physics Formula", "content": "F = ma", "createdAt": datetime.datetime.now()},
                {"title": "History Date", "content": "1947 - Independence", "createdAt": datetime.datetime.now()}
            ]
        }
        db.collection("saved_notes").document(uid).set(notes_data)

    print("\n✅ Seeding complete!")
    print(f"Created {NUM_USERS} users with query history.")
    print("You can now log in as 'test_student_1' (no password needed for dashboard view logic) or just use the uid 'test_student_1' in API calls.")

if __name__ == "__main__":
    seed_data()

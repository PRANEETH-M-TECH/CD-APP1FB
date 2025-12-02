import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Initialize Firebase
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(backend_dir)
    sa_path = os.path.join(project_root, "serviceAccountKey.json")
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    
    db = firestore.Client()
    print("✅ Connected to Firestore")
    
    # Reproduce the query
    uid = "Y0Ql0s2NzJVfZgfniPGi63Eomme2" # User from the error log
    print(f"Attempting query for uid: {uid}")
    
    queries_ref = db.collection("user_queries")\
        .where("uid", "==", uid)\
        .order_by("timestamp", direction=firestore.Query.DESCENDING)\
        .limit(5)
    
    docs = list(queries_ref.stream())
    print(f"✅ Success! Retrieved {len(docs)} docs")

except Exception as e:
    print(f"\n❌ ERROR CAUGHT:\n{e}")

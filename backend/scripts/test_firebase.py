import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(backend_dir)
    sa_path = os.path.join(project_root, "serviceAccountKey.json")
    
    print(f"Checking key at: {sa_path}")
    if not os.path.exists(sa_path):
        print("❌ Key not found")
        sys.exit(1)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    
    db = firestore.Client()
    print("✅ Connected. Writing test doc...")
    
    db.collection("test_collection").document("test_doc").set({"status": "ok"})
    print("✅ Write successful")

except Exception as e:
    print(f"❌ Error: {e}")

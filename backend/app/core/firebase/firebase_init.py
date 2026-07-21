import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage

# Path to service account
FIREBASE_DIR = os.path.dirname(os.path.abspath(__file__))
SA_PATH = None
curr_dir = FIREBASE_DIR
for _ in range(6):  # check up to 6 parent directories
    temp_path = os.path.join(curr_dir, "serviceAccountKey.json")
    if os.path.exists(temp_path):
        SA_PATH = temp_path
        break
    parent = os.path.dirname(curr_dir)
    if parent == curr_dir:
        break
    curr_dir = parent

if not SA_PATH:
    SA_PATH = os.path.abspath(os.path.join(FIREBASE_DIR, "..", "..", "..", "..", "serviceAccountKey.json"))

firebase_env_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or os.environ.get("FIREBASE_CREDENTIALS")

if not firebase_admin._apps:
    if firebase_env_json:
        try:
            cred_dict = json.loads(firebase_env_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("[Firebase Success] Initialized Firebase Admin from environment variable.")
        except Exception as e:
            print(f"[Firebase Warning] Failed to initialize Firebase from env: {e}")
    elif SA_PATH and os.path.exists(SA_PATH):
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH
            cred = credentials.Certificate(SA_PATH)
            firebase_admin.initialize_app(cred)
            print(f"[Firebase Success] Initialized Firebase Admin from file: {SA_PATH}")
        except Exception as e:
            print(f"[Firebase Warning] Failed to initialize Firebase from file: {e}")
    else:
        print("[Firebase Warning] serviceAccountKey.json not found on disk & FIREBASE_SERVICE_ACCOUNT_JSON not set. Firebase Admin SDK skipped.")

try:
    db = firestore.client() if firebase_admin._apps else None
except Exception as e:
    print(f"[Firebase Warning] Could not initialize Firestore client: {e}")
    db = None

# Google Cloud Storage
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
try:
    gcs_client = storage.Client() if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else None
    bucket = gcs_client.bucket(GCS_BUCKET) if (gcs_client and GCS_BUCKET) else None
except Exception as e:
    print(f"[GCS Warning] Cloud storage bucket initialization skipped: {e}")
    bucket = None

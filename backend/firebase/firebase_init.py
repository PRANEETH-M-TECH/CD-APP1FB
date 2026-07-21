import os
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage

# Path to service account
FIREBASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(FIREBASE_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
SA_PATH = os.path.join(PROJECT_ROOT, "serviceAccountKey.json")

if not os.path.exists(SA_PATH):
    raise FileNotFoundError(
        f"❌ Firebase service account key NOT found at: {SA_PATH}\n"
        "1. Download your service account key from Firebase Console.\n"
        "2. Rename it to 'serviceAccountKey.json'.\n"
        "3. Place it in the project root directory.\n"
        "(See 'serviceAccountKey.example.json' for the expected structure)"
    )

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH

# Initialize Firebase Admin
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SA_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to initialize Firebase: {e}")

db = firestore.client()

# Google Cloud Storage
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
if not GCS_BUCKET:
    raise ValueError("❌ GCS_BUCKET_NAME not found in .env file.")

try:
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(GCS_BUCKET)
except Exception as e:
    print(f"⚠️ Warning: Could not connect to GCS bucket '{GCS_BUCKET}': {e}")
    bucket = None

print("[FIREBASE] Firebase Admin Initialized")
if bucket:
    print(f"[FIREBASE] Connected to Bucket: {GCS_BUCKET}")

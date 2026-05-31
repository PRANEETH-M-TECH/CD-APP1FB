import os
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage

# Path to service account
FIREBASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Search upwards for serviceAccountKey.json
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
    # Fallback to default expected location in workspace root (4 levels up)
    SA_PATH = os.path.abspath(os.path.join(FIREBASE_DIR, "..", "..", "..", "..", "serviceAccountKey.json"))

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
        raise RuntimeError(f"[Firebase ERROR] Failed to initialize Firebase: {e}")

db = firestore.client()

# Google Cloud Storage
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
if not GCS_BUCKET:
    raise ValueError("[Firebase ERROR] GCS_BUCKET_NAME not found in .env file.")

try:
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(GCS_BUCKET)
except Exception as e:
    print(f"[Firebase Warning] Could not connect to GCS bucket '{GCS_BUCKET}': {e}")
    bucket = None

print("[FIREBASE] Firebase Admin Initialized")
if bucket:
    print(f"[FIREBASE] Connected to Bucket: {GCS_BUCKET}")

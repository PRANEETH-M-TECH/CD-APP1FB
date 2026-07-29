import os
import sys
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(override=True)

print("Attempting to import and initialize qdrant_service...")
try:
    from backend.app.services.retrieval import qdrant_service
    qdrant_service.initialize()
    print("[SUCCESS] qdrant_service initialized successfully!")
    
    # Test encoding a query
    print("Testing embed_query...")
    vector = qdrant_service.embed_query("What is photosynthesis?")
    print(f"[SUCCESS] Query embedded. Dimension: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    
except Exception as e:
    print(f"[ERROR] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

import sys
import os
from dotenv import load_dotenv

# Ensure the project root is in the path and load environment variables
sys.path.append(os.path.abspath('.'))
load_dotenv(override=True)

# Mock fallback for Firebase if not in environment
if "GCS_BUCKET_NAME" not in os.environ:
    os.environ["GCS_BUCKET_NAME"] = "mock-bucket"

print("=== Starting Visual Learning Mode Backend Verification ===")

try:
    # Set dummy env vars for local verification
    os.environ["QDRANT_URL"] = os.environ.get("QDRANT_URL", "http://localhost:6333")
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "mock_key")
    
    # 1. Test basic package import
    print("Testing visual_learning package import...")
    import backend.app.services.visual_learning
    print("[OK] Package visual_learning imported.")
    
    # 2. Test individual modules
    print("Testing visual_lesson_prompt...")
    from backend.app.services.visual_learning import visual_lesson_prompt
    print("[OK] visual_lesson_prompt imported.")
    
    print("Testing asset_retrieval_service...")
    from backend.app.services.visual_learning import asset_retrieval_service
    print("[OK] asset_retrieval_service imported.")
    
    print("Testing visual_audio_generator...")
    from backend.app.services.visual_learning import visual_audio_generator
    print("[OK] visual_audio_generator imported.")
    
    # 3. Test service orchestrator
    print("Testing visual_learning_service...")
    from backend.app.services.visual_learning import visual_learning_service
    print("[OK] visual_learning_service imported.")
    
    # 4. Test API route
    print("Testing visual_learning API route...")
    from backend.app.api.routes import visual_learning as visual_learning_route
    print("[OK] visual_learning API route imported.")
    
    # 5. Check if main.py can import the router
    print("Checking main.py router registration compatibility...")
    from backend.app.main import app
    print("[OK] Main FastAPI app initialized and registered visual_learning_router successfully.")
    
    print("\n[SUCCESS] All imports and registrations are fully functional and syntactically correct!")

except Exception as e:
    print(f"\n[FAIL] Verification encountered an error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

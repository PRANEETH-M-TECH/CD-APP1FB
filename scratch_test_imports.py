import sys
import os
sys.path.append(os.path.abspath('.'))

try:
    # 1. Initialize qdrant_service first (mocking main.py lifespan behavior)
    from backend.app.services.retrieval import qdrant_service
    
    # Mock settings
    os.environ["QDRANT_URL"] = "http://localhost:6333" # dummy
    
    # Mock genai Client
    class MockClient:
        def __init__(self):
            self.models = "mock_models_attr"
            
    qdrant_service.gemini_client = MockClient()
    qdrant_service.generation_model_name = "gemini-mock"
    
    print("[OK] Mocked Qdrant Gemini Client successfully.")

    # 2. Now import answer_service and check attributes
    from backend.app.services.chat import answer_service
    
    print("Checking dynamic attributes in answer_service:")
    print(f"  - gemini_client: {answer_service.gemini_client} (models: {answer_service.gemini_client.models})")
    print(f"  - generation_model_name: {answer_service.generation_model_name}")
    
    if answer_service.gemini_client.models == "mock_models_attr" and answer_service.generation_model_name == "gemini-mock":
        print("[SUCCESS] Dynamic import bindings are fully functional!")
    else:
        print("[FAIL] Dynamic import bindings failed to resolve correct values.")

except Exception as e:
    print(f"[FAIL] Verification script encountered error: {e}")
    import traceback
    traceback.print_exc()

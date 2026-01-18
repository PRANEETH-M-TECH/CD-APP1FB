import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env from the script's directory
# override=True ensures .env file takes precedence over system environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("⚠️  No GOOGLE_API_KEY found in .env file")
    print(f"Looking for .env at: {env_path}")
    print(f".env exists: {env_path.exists()}")
    exit(1)

print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-8:]}")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Explain quantum computing in one sentence."
)

print(f"\n📝 Response: {response.text}")

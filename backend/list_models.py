import os
from dotenv import load_dotenv
from google import genai  # NEW SDK!

# Load environment variables
load_dotenv()

# Configure API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("❌ No API key found in environment!")
    exit(1)

# Create client
client = genai.Client(api_key=api_key)

# List all available models
print("🔍 Listing all available Gemini models:\n")
print("=" * 70)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Description: {model.description[:100]}..." if len(model.description) > 100 else f"   Description: {model.description}")
        print("-" * 70)

print("\n💡 Use these model names in your code (without 'models/' prefix)")

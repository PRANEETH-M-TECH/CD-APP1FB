import os
from dotenv import load_dotenv
from google import genai

# Load .env variables and override pre-existing env variables
load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
print(f"Loaded API key from env: {api_key[:15] if api_key else 'None'}... (len: {len(api_key) if api_key else 0})")

# Let's also check GEMINI_API_KEY explicitly
print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')[:15] if os.getenv('GOOGLE_API_KEY') else 'None'}...")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')[:15] if os.getenv('GEMINI_API_KEY') else 'None'}...")

if not api_key:
    print("Error: No API key found in environment.")
    exit(1)

client = genai.Client(api_key=api_key)

print("\n--- Testing available models ---")
try:
    models = list(client.models.list())
    print("Successfully retrieved models list. First 15 models:")
    for model in models[:15]:
        print(f" - {model.name} (Supported actions: {model.supported_generation_methods})")
except Exception as e:
    print(f"Error listing models: {e}")

# Check if a model starting with gemini-3.6 or containing '3.6' exists
print("\n--- Checking for 3.6 or 3.5 models ---")
try:
    all_models = list(client.models.list())
    matched = [m.name for m in all_models if "3.6" in m.name or "3.5" in m.name]
    print(f"Matched models: {matched}")
except Exception as e:
    print(f"Error filtering models: {e}")

# Test generating with gemini-3.6-flash (if the user requested it)
test_model = "gemini-3.6-flash"
print(f"\n--- Testing generation with '{test_model}' ---")
try:
    response = client.models.generate_content(
        model=test_model,
        contents="Hello, this is a test. Reply with 'OK' if you receive this."
    )
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Failed to generate content with '{test_model}': {e}")

# Test generating with gemini-2.5-flash just in case
print("\n--- Testing generation with 'gemini-2.5-flash' ---")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello, this is a test. Reply with 'OK' if you receive this."
    )
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Failed to generate content with 'gemini-2.5-flash': {e}")

#!/usr/bin/env python3
"""
Quick test script to verify the Gemini API key is working.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: No Gemini API key found in environment (.env).")
    print("   Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
    exit(1)

print(f"✓ API Key found (starts with: {api_key[:10]}...)")

try:
    # Configure Gemini
    genai.configure(api_key=api_key)
    print("✓ Gemini configured successfully")
    
    # List available models
    print("\n🔍 Available models:")
    try:
        models = genai.list_models()
        flash_models = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"   - {m.name}")
                if 'flash' in m.name.lower():
                    flash_models.append(m.name)
        
        if not flash_models:
            print("\n⚠️ No flash models found. Using first available model...")
            model_name = models[0].name if models else "gemini-1.5-flash"
        else:
            model_name = flash_models[0]
        
        print(f"\n✓ Using model: {model_name}")
    except Exception as e:
        print(f"   Could not list models: {e}")
        print("   Trying with default model name...")
        model_name = "gemini-1.5-flash"
    
    # Try to initialize a model
    model = genai.GenerativeModel(model_name)
    print("✓ Model initialized successfully")
    
    # Test with a simple prompt
    print("\n🔄 Testing API with a simple prompt...")
    response = model.generate_content("Say 'Hello, the API key is working!' in exactly 7 words.")
    
    # Check if we got a response
    if response and response.text:
        print("\n✅ SUCCESS! Gemini API key is working!")
        print(f"   Response: {response.text}")
        exit(0)
    else:
        print("\n⚠️ WARNING: Got a response but no text content.")
        print(f"   Response: {response}")
        exit(1)
        
        
except Exception as e:
    print(f"\n❌ ERROR: Failed to use Gemini API")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")
    
    # Check for common error types
    if "API_KEY_INVALID" in str(e) or "invalid API key" in str(e).lower():
        print("\n💡 The API key appears to be invalid or expired.")
        print("   Please generate a new key at: https://aistudio.google.com/app/apikey")
    elif "quota" in str(e).lower() or "limit" in str(e).lower():
        print("\n💡 The API key may have exceeded its quota or rate limit.")
    
    exit(1)

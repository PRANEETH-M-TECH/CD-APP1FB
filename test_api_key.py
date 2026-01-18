#!/usr/bin/env python3
"""
Simple script to test if your Google Gemini API key is valid and working.
Just replace YOUR_API_KEY_HERE with your actual key and run this script.
"""

from google import genai

# ========================================
# PASTE YOUR API KEY HERE:
# ========================================
API_KEY = "AIzaSyCKIKiSqPGDN8RkIRA12L-VNhBWGJNqCds"  # Replace with your actual key

def test_api_key():
    """Test if the API key is valid and working."""
    
    print("=" * 60)
    print("🔑 TESTING GOOGLE GEMINI API KEY")
    print("=" * 60)
    
    # Check if key is provided
    if not API_KEY or API_KEY == "AIzaSyCKIKiSqPGDN8RkIRA12L-VNhBWGJNqCds":
        print("❌ ERROR: Please paste your API key in the script first!")
        return False
    
    print(f"\n📋 API Key: {API_KEY[:10]}...{API_KEY[-8:]}")
    print(f"📏 Length: {len(API_KEY)} characters")
    
    # Try to create a client and test generation
    try:
        print("\n🤖 Creating Gemini Client...")
        client = genai.Client(api_key=API_KEY)
        print("✅ Client created successfully!")
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return False
    
    # Try to generate content
    try:
        print("\n💬 Testing content generation...")
        print("   Sending test prompt: 'Say hello in one sentence'")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello in one sentence"
        )
        
        print("\n✅ SUCCESS! API key is working!")
        print(f"\n📝 Response from Gemini:")
        print(f"   {response.text}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Content generation failed!")
        print(f"   Error: {e}")
        print(f"\n💡 This usually means:")
        print(f"   - API key is invalid or expired")
        print(f"   - API key doesn't have v1beta access")
        print(f"   - API key needs to be created in Google AI Studio")
        return False

if __name__ == "__main__":
    success = test_api_key()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 RESULT: Your API key is VALID and WORKING!")
        print("✅ You can use this key in your .env file")
    else:
        print("❌ RESULT: Your API key is NOT working")
        print("🔧 Solution: Create a new key at https://aistudio.google.com/app/apikey")
        print("   Make sure to select 'Create API key in new project'")
    print("=" * 60)

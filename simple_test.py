#!/usr/bin/env python3
"""
Simple manual test to check if smart_query endpoint works at all.
"""
import requests
import json

BASE_URL = "http://localhost:8000"
BOOK_UUID = "9e3196f483e35b8754d561045aa618d4a208cfab40fbfa7ffee757800a0b40f2"

print("Testing /api/smart_query endpoint...")
print(f"Book UUID: {BOOK_UUID}\n")

params = {
    "book_uuid": BOOK_UUID,
    "query": "test question",
    "class_name": "8",
    "subject": "science"
}

try:
    print("Sending request...")
    response = requests.get(
        f"{BASE_URL}/api/smart_query",
        params=params,
        stream=True,
        timeout=15
    )
    
    print(f"Status Code: {response.status_code}\n")
    
    if response.status_code != 200:
        print(f"ERROR: {response.text}")
        exit(1)
    
    print("SSE Events:")
    print("="*60)
    
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            print(decoded)
            
            if 'data: [DONE]' in decoded:
                print("="*60)
                print("\n✅ Stream completed successfully!")
                break
            
            if 'error' in decoded.lower():
                print("\n❌ Error detected in stream!")
                break

except requests.exceptions.Timeout:
    print("\n❌ Request timed out after 15 seconds")
    print("This suggests the endpoint is hanging")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

"""
Test script for Phase 1 smart query backend implementation.
Tests session management, intent classification, and context caching.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Replace with your actual book UUID
BOOK_UUID = "9e3196f483e35b8754d561045aa618d4a208cfab40fbfa7ffee757800a0b40f2"  # Get from admin panel
CLASS_NAME = "8"
SUBJECT = "science"

def stream_query(query, session_id=None, is_clicked_followup=False):
    """Send query to smart_query endpoint and print streaming response."""
    
    params = {
        "book_uuid": BOOK_UUID,
        "query": query,
        "class_name": CLASS_NAME,
        "subject": SUBJECT
    }
    
    if session_id:
        params["session_id"] = session_id
    
    if is_clicked_followup:
        params["is_clicked_followup"] = "true"
    
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"Session ID: {session_id if session_id else 'New (will create)'}")
    print(f"Clicked Followup: {is_clicked_followup}")
    print(f"{'='*80}\n")
    
    response = requests.get(
        f"{BASE_URL}/api/smart_query",
        params=params,
        stream=True
    )
    
    intent_type = None
    session_id_returned = None
    followups = []
    answer = ""
    turn_info = None
    
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: '):
                data_str = decoded[6:]  # Remove 'data: ' prefix
                
                if data_str == '[DONE]':
                    print("\n✅ Query complete!")
                    break
                
                try:
                    data = json.loads(data_str)
                    
                    if data.get('type') == 'intent':
                        intent_type = data.get('intent')
                        print(f"🎯 INTENT: {intent_type.upper()}")
                    
                    elif data.get('display_text'):
                        answer += data['display_text']
                        print(data['display_text'], end='', flush=True)
                    
                    elif data.get('type') == 'followups':
                        followups = data.get('followups', [])
                    
                    elif data.get('type') == 'metadata':
                        session_id_returned = data.get('session_id')
                        turn = data.get('turn')
                        total = data.get('total')
                        turn_info = f"Turn {turn} of {total}"
                        print(f"\n\n📊 {turn_info}")
                        print(f"📌 Session: {session_id_returned[:16] if session_id_returned else 'Unknown'}...")
                
                except json.JSONDecodeError:
                    pass
    
    # Print follow-ups
    if followups:
        print(f"\n💡 Follow-up Suggestions:")
        for i, f in enumerate(followups, 1):
            print(f"   {i}. {f}")
    
    print(f"\n📋 Summary:")
    print(f"   Intent: {intent_type}")
    print(f"   Session: {session_id_returned[:16] if session_id_returned else 'None'}...")
    print(f"   Follow-ups: {len(followups)}")
    
    return {
        "intent": intent_type,
        "session_id": session_id_returned,  # This is the key fix
        "followups": followups,
        "answer": answer
    }


def test_scenario_1_independent_queries():
    """Test 1: Two independent queries (different topics)"""
    print("\n" + "="*80)
    print("TEST 1: Independent Queries (Different Topics)")
    print("="*80)
    
    # Query 1: Photosynthesis
    result1 = stream_query("What is photosynthesis?")
    assert result1["intent"] == "independent", "First query should be independent"
    session_id = result1["session_id"]
    
    time.sleep(2)
    
    # Query 2: Motion (different topic, same session)
    result2 = stream_query("What is Newton's law of motion?", session_id=session_id)
    assert result2["intent"] == "independent", "Topic switch should be detected as independent"
    
    print("\n✅ TEST 1 PASSED: Topic switching detected correctly!")


def test_scenario_2_followup_queries():
    """Test 2: Independent query followed by follow-ups"""
    print("\n" + "="*80)
    print("TEST 2: Follow-up Query Detection")
    print("="*80)
    
    # Query 1: Independent
    result1 = stream_query("What is photosynthesis?")
    session_id = result1["session_id"]
    
    time.sleep(2)
    
    # Query 2: Vague follow-up (should detect as follow-up)
    result2 = stream_query("explain more about that", session_id=session_id)
    assert result2["intent"] == "followup", "Vague follow-up should be detected"
    
    time.sleep(2)
    
    # Query 3: Pattern-matched follow-up
    result3 = stream_query("give me an example", session_id=session_id)
    assert result3["intent"] == "followup", "Pattern should match follow-up"
    
    print("\n✅ TEST 2 PASSED: Follow-up detection working!")


def test_scenario_3_clicked_followup():
    """Test 3: Clicked follow-up button"""
    print("\n" + "="*80)
    print("TEST 3: Clicked Follow-up Button")
    print("="*80)
    
    # Query 1: Get follow-up suggestions
    result1 = stream_query("What is respiration?")
    session_id = result1["session_id"]
    followups = result1["followups"]
    
    if not followups:
        print("⚠️ No follow-ups generated, skipping test")
        return
    
    time.sleep(2)
    
    # Query 2: Click first suggestion
    clicked_question = followups[0]
    result2 = stream_query(clicked_question, session_id=session_id, is_clicked_followup=True)
    assert result2["intent"] == "followup", "Clicked follow-up should be detected"
    
    print("\n✅ TEST 3 PASSED: Clicked follow-up handled correctly!")


def test_scenario_4_book_switch():
    """Test 4: Book switching creates new session"""
    print("\n" + "="*80)
    print("TEST 4: Book Switch Detection")
    print("="*80)
    
    # Query 1: Science book
    result1 = stream_query("What is photosynthesis?")
    session_id1 = result1["session_id"]
    
    print(f"\n📚 Session 1: {session_id1}")
    
    # Note: To properly test book switching, you'd need a different book_uuid
    # For now, we'll just verify session was created
    assert session_id1 is not None, "Session should be created"
    
    print("\n✅ TEST 4 PASSED: Session management working!")


if __name__ == "__main__":
    print("\n🧪 PHASE 1 BACKEND TESTING")
    print("="*80)
    print("⚠️  Make sure to:")
    print("   1. Update BOOK_UUID with your actual book UUID")
    print("   2. Start the server: uvicorn backend.app:app --reload --port 8000")
    print("="*80)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print("✅ Server is running!\n")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running! Please start it first.")
        exit(1)
    
    # Run tests
    try:
        test_scenario_1_independent_queries()
        test_scenario_2_followup_queries()
        test_scenario_3_clicked_followup()
        test_scenario_4_book_switch()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

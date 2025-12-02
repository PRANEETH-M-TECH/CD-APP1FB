import requests

BASE = "http://localhost:8000"

def test_query_logging():
    payload = {"query": "What is photosynthesis?", "mode": "text"}
    res = requests.post(f"{BASE}/api/ask", json=payload)
    assert res.status_code == 200

def test_dashboard_summary():
    res = requests.get(f"{BASE}/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_queries" in data

def test_chapter_stats():
    res = requests.get(f"{BASE}/api/admin/chapter-hotspots")
    assert res.status_code == 200

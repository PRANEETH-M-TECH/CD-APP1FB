# Analytics Integration Testing Guide

## What Was Fixed

### Critical Bugs Fixed in `analytics_service.py`:
1. ✅ Undefined `last_active` variable - causing crash when updating existing user stats
2. ✅ Undefined `weekly_key` variable - preventing weekly activity tracking
3. ✅ Wrong dictionary access (`doc.get()` instead of `current_data.get()`)
4. ✅ Reference to undefined `new_streak` variable in logging

### Flow Improvements:
1. ✅ Removed session_id fallback - now uses real Firebase UIDs only
2. ✅ Added comprehensive logging for debugging
3. ✅ Token now properly passed via query params for EventSource
4. ✅ Auth middleware extracts token from query params

## How to Test End-to-End

### Step 1: Restart the Server
```bash
# Stop the current server (Ctrl+C)
# Start fresh
cd /Users/mac/Desktop/CG-FOLDER/CD-APP1FB
./venv/bin/python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8080
```

### Step 2: Clear Browser Data (Important!)
1. Open your browser's Developer Tools (F12)
2. Go to Application tab
3. Clear all cookies, local storage, and session storage for localhost:8080
4. Refresh the page

### Step 3: Login as Student
1. Go to http://localhost:8080
2. Login with: student8@cg.com (or your student account)
3. Select a book (Class 8, Social Studies, etc.)

### Step 4: Ask a NEW Question
**IMPORTANT**: Old questions were logged with session IDs. You must ask a NEW question!

1. Type a question in the chat
2. Submit and wait for response
3. Check the terminal logs - you should see:
   ```
   INFO:     Token extracted from query params for /api/smart_query
   INFO:     Authenticated user: Y0Ql0s2NzJVfZgfniPGi63Eomme2 (student8@cg.com)
   INFO:     [ANALYTICS] Using UID for analytics: Y0Ql0s2NzJVfZgfniPGi63Eomme2
   ```

### Step 5: Verify Student Dashboard
1. Click "My Dashboard" in the sidebar
2. You should now see:
   - Total Questions: 1 (or more)
   - Learning Streak: 1 🔥
   - Subjects Explored: 1
   - This Week: 1
   - Charts with data
   - Chapter-wise table populated

### Step 6: Verify Admin Dashboard
1. Logout from student account
2. Login to admin dashboard at http://localhost:8080/admin-login.html
3. Use admin@cg.com (or your admin credentials)
4. Go to Admin Analytics Dashboard
5. You should see:
   - Total Queries: updated count
   - Class distribution chart
   - Subject distribution pie chart
   - Chapter hotspots table

### Step 7: Verify Database
Run this command to check Firestore:
```bash
./venv/bin/python check_analytics_data.py
```

You should see:
- User Stats with YOUR actual Firebase UID
- Chapter Stats with query counts
- User Queries with YOUR actual Firebase UID (not session IDs)

## Troubleshooting

### If Dashboard is Still Empty:
1. **Check browser console** for JavaScript errors
2. **Check terminal logs** - look for:
   - "Token extracted from query params"
   - "Authenticated user: [YOUR_UID]"  
   - "✅ Query logged: ..."
   - "✅ User stats updated..."
   - "✅ Chapter stats updated..."

### If No Token in Logs:
- Frontend might not be sending it
- Clear browser cache and reload
- Check that you're logged in (firebase.auth().currentUser should exist)

### If Token Present but UID is "anonymous":
- Token verification might be failing
- Check serviceAccountKey.json is valid
- Check Firebase project ID matches

### If Errors in Analytics Functions:
- Check the terminal for Python tracebacks
- The bugs I fixed should have resolved this
- If new errors appear, share them with me

## Expected Data Flow

```
User asks question
  ↓
Frontend (script.js):
  - Gets Firebase token via firebase.auth().currentUser.getIdToken()
  - Appends token to EventSource URL
  ↓
Backend (auth_middleware.py):
  - Extracts token from query params
  - Verifies with Firebase
  - Sets request.state.uid = <actual Firebase UID>
  ↓
Backend (app.py smart_query_engine):
  - Gets uid from request.state
  - Logs to analytics with real UID
  ↓
Analytics Functions:
  - log_query() → user_queries collection
  - update_user_stats() → user_stats collection
  - update_chapter_stats() → chapter_stats collection
  ↓
Dashboards query these collections and display data
```

## Success Criteria

✅ Student Dashboard shows non-zero values after asking questions
✅ Admin Dashboard shows aggregated statistics
✅ Database check shows real Firebase UIDs (not session IDs)
✅ Terminal logs show proper authentication flow
✅ No Python errors in terminal
✅ No JavaScript errors in browser console

## Next Steps If This Works

Once you confirm it's working:
1. I can clean up old session-based data from Firestore
2. I can add more features to dashboards
3. I can improve the UI with better visualizations
4. I can add export/download features for analytics

Let me know the results!

# Firestore Security Rules Deployment

This file contains the Firebase security rules for the Chaduvu Guru application.

## How to Deploy

### Option 1: Firebase CLI (Recommended)
```bash
cd /Users/mac/Desktop/CG-FOLDER/CD-APP1FB
firebase deploy --only firestore:rules
```

### Option 2: Firebase Console
1. Go to https://console.firebase.google.com
2. Select your project
3. Navigate to **Firestore Database** → **Rules**
4. Copy and paste the contents of `firestore.rules`
5. Click **Publish**

## Rules Overview

The security rules ensure:
- ✅ Users can only create/read/write notebooks in their own user document
- ✅ All authenticated users can read books and chapters
- ✅ Only admins can write/modify books
- ✅ Users can write their own analytics data

## Testing After Deployment

After deploying, test notebook creation:
1. Open My Bag
2. Create a new notebook
3. It should work without permission errors

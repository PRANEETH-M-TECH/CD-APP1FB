import os
import firebase_admin
from firebase_admin import credentials, firestore

def clear_collection(db_client, collection_name, batch_size=50):
    """
    Deletes all documents in a Firestore collection in batches.
    """
    coll_ref = db_client.collection(collection_name)
    docs = coll_ref.limit(batch_size).stream()
    deleted = 0

    while True:
        doc_list = list(docs)
        if not doc_list:
            break

        print(f"  - Deleting {len(doc_list)} documents from '{collection_name}'...")
        batch = db_client.batch()
        for doc in doc_list:
            batch.delete(doc.reference)
        
        batch.commit()
        deleted += len(doc_list)
        
        # Get the next batch
        docs = coll_ref.limit(batch_size).stream()

    if deleted > 0:
        print(f"  ✓ Successfully deleted {deleted} documents from '{collection_name}'.")
    else:
        print(f"  - Collection '{collection_name}' is already empty.")

def main():
    """
    Main function to initialize Firebase and clear analytics collections.
    """
    print("🔥 Initializing Firestore data cleaner...")

    # --- Initialize Firebase Admin ---
    try:
        sa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serviceAccountKey.json")
        if not os.path.exists(sa_path):
            raise FileNotFoundError(
                f"Firebase service account key not found at: {sa_path}\n"
                "Please ensure 'serviceAccountKey.json' is in the project root."
            )
        
        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        print("✅ Firebase initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Firebase: {e}")
        return

    # --- Collections to Clear ---
    # WARNING: This will permanently delete all data in these collections.
    analytics_collections = [
        "user_queries",
        "user_stats",
        "chapter_stats",
        "student_mistakes"
    ]
    
    print("\n⚠️ This script will PERMANENTLY DELETE all documents from the following collections:")
    for coll in analytics_collections:
        print(f"  - {coll}")
    
    # --- User Confirmation ---
    confirm = input("\nAre you sure you want to continue? (yes/no): ").lower().strip()
    if confirm != 'yes':
        print("\n🚫 Operation cancelled.")
        return
        
    # --- Execute Deletion ---
    print("\n🚀 Starting deletion process...\n")
    for collection_name in analytics_collections:
        print(f"Processing collection: '{collection_name}'")
        try:
            clear_collection(db, collection_name)
        except Exception as e:
            print(f"  ✗ An error occurred while clearing '{collection_name}': {e}")
    
    print("\n✨ All specified analytics collections have been processed.")
    print("You now have a clean slate for analytics data.")

if __name__ == "__main__":
    main()

"""
My Bag Service for CHADUVU-GURU
Manages student notebooks and saved content.
"""

from google.cloud import firestore
from .firebase.firebase_init import db
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================
# NOTEBOOK MANAGEMENT
# ============================================

def create_notebook(uid: str, notebook_name: str, subject: str = "General", color: str = "#4F46E5") -> str:
    """
    Create a new notebook for the student.
    
    Args:
        uid: User ID
        notebook_name: Name of the notebook
        subject: Subject category
        color: Hex color code for the notebook
    
    Returns:
        Notebook ID
    """
    try:
        doc_ref = db.collection("notebooks").document()
        notebook_id = doc_ref.id
        
        notebook_data = {
            "notebook_id": notebook_id,
            "uid": uid,
            "name": notebook_name,
            "subject": subject,
            "color": color,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "item_count": 0
        }
        
        doc_ref.set(notebook_data)
        logger.info(f"✅ Created notebook '{notebook_name}' for user {uid}")
        return notebook_id
        
    except Exception as e:
        logger.error(f"❌ Failed to create notebook: {e}")
        raise


def get_notebooks(uid: str) -> List[Dict]:
    """
    Get all notebooks for a user.
    
    Args:
        uid: User ID
    
    Returns:
        List of notebook dictionaries
    """
    try:
        notebooks_ref = db.collection("notebooks")\
            .where("uid", "==", uid)\
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
        
        notebooks = []
        for doc in notebooks_ref.stream():
            notebook_data = doc.to_dict()
            # Format timestamps
            if notebook_data.get("created_at"):
                notebook_data["created_at"] = notebook_data["created_at"].isoformat()
            if notebook_data.get("updated_at"):
                notebook_data["updated_at"] = notebook_data["updated_at"].isoformat()
            notebooks.append(notebook_data)
        
        logger.info(f"✅ Retrieved {len(notebooks)} notebooks for user {uid}")
        return notebooks
        
    except Exception as e:
        logger.error(f"❌ Failed to get notebooks: {e}")
        raise


def delete_notebook(uid: str, notebook_id: str) -> None:
    """
    Delete a notebook and all its contents.
    
    Args:
        uid: User ID (for security check)
        notebook_id: Notebook ID to delete
    """
    try:
        # Verify ownership
        notebook_ref = db.collection("notebooks").document(notebook_id)
        notebook = notebook_ref.get()
        
        if not notebook.exists:
            raise ValueError("Notebook not found")
        
        if notebook.to_dict().get("uid") != uid:
            raise ValueError("Unauthorized: You don't own this notebook")
        
        # Delete all items in the notebook
        items_ref = db.collection("bag_items")\
            .where("notebook_id", "==", notebook_id)
        
        for item_doc in items_ref.stream():
            item_doc.reference.delete()
        
        # Delete the notebook
        notebook_ref.delete()
        logger.info(f"✅ Deleted notebook {notebook_id} for user {uid}")
        
    except Exception as e:
        logger.error(f"❌ Failed to delete notebook: {e}")
        raise


# ============================================
# CONTENT MANAGEMENT
# ============================================

def save_to_bag(
    uid: str,
    notebook_id: str,
    content: str,
    title: str = None,
    source_query: str = None,
    chapter_name: str = None,
    subject: str = None
) -> str:
    """
    Save content to a notebook.
    
    Args:
        uid: User ID
        notebook_id: Target notebook ID
        content: The content to save
        title: Optional title (auto-generated if not provided)
        source_query: Original query that generated this content
        chapter_name: Chapter the content is from
        subject: Subject category
    
    Returns:
        Item ID
    """
    try:
        # Verify notebook ownership
        notebook_ref = db.collection("notebooks").document(notebook_id)
        notebook = notebook_ref.get()
        
        if not notebook.exists:
            raise ValueError("Notebook not found")
        
        if notebook.to_dict().get("uid") != uid:
            raise ValueError("Unauthorized: You don't own this notebook")
        
        # Create the item
        item_ref = db.collection("bag_items").document()
        item_id = item_ref.id
        
        # Auto-generate title if not provided
        if not title:
            title = content[:50] + "..." if len(content) > 50 else content
        
        item_data = {
            "item_id": item_id,
            "notebook_id": notebook_id,
            "uid": uid,
            "title": title,
            "content": content,
            "source_query": source_query,
            "chapter_name": chapter_name,
            "subject": subject,
            "created_at": firestore.SERVER_TIMESTAMP,
            "is_favorite": False
        }
        
        item_ref.set(item_data)
        
        # Update notebook item count and timestamp
        notebook_ref.update({
            "item_count": firestore.Increment(1),
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        
        logger.info(f"✅ Saved content to notebook {notebook_id} for user {uid}")
        return item_id
        
    except Exception as e:
        logger.error(f"❌ Failed to save to bag: {e}")
        raise


def get_bag_items(uid: str, notebook_id: str = None) -> List[Dict]:
    """
    Get items from bag, optionally filtered by notebook.
    
    Args:
        uid: User ID
        notebook_id: Optional notebook filter
    
    Returns:
        List of bag item dictionaries
    """
    try:
        items_ref = db.collection("bag_items").where("uid", "==", uid)
        
        if notebook_id:
            items_ref = items_ref.where("notebook_id", "==", notebook_id)
        
        items_ref = items_ref.order_by("created_at", direction=firestore.Query.DESCENDING)
        
        items = []
        for doc in items_ref.stream():
            item_data = doc.to_dict()
            # Format timestamp
            if item_data.get("created_at"):
                item_data["created_at"] = item_data["created_at"].isoformat()
            items.append(item_data)
        
        logger.info(f"✅ Retrieved {len(items)} items for user {uid}")
        return items
        
    except Exception as e:
        logger.error(f"❌ Failed to get bag items: {e}")
        raise


def delete_bag_item(uid: str, item_id: str) -> None:
    """
    Delete an item from bag.
    
    Args:
        uid: User ID (for security check)
        item_id: Item ID to delete
    """
    try:
        item_ref = db.collection("bag_items").document(item_id)
        item = item_ref.get()
        
        if not item.exists:
            raise ValueError("Item not found")
        
        item_data = item.to_dict()
        if item_data.get("uid") != uid:
            raise ValueError("Unauthorized: You don't own this item")
        
        notebook_id = item_data.get("notebook_id")
        
        # Delete the item
        item_ref.delete()
        
        # Update notebook item count
        if notebook_id:
            notebook_ref = db.collection("notebooks").document(notebook_id)
            notebook_ref.update({
                "item_count": firestore.Increment(-1),
                "updated_at": firestore.SERVER_TIMESTAMP
            })
        
        logger.info(f"✅ Deleted item {item_id} for user {uid}")
        
    except Exception as e:
        logger.error(f"❌ Failed to delete bag item: {e}")
        raise


def toggle_favorite(uid: str, item_id: str) -> bool:
    """
    Toggle favorite status of an item.
    
    Args:
        uid: User ID
        item_id: Item ID
    
    Returns:
        New favorite status
    """
    try:
        item_ref = db.collection("bag_items").document(item_id)
        item = item_ref.get()
        
        if not item.exists:
            raise ValueError("Item not found")
        
        if item.to_dict().get("uid") != uid:
            raise ValueError("Unauthorized")
        
        current_status = item.to_dict().get("is_favorite", False)
        new_status = not current_status
        
        item_ref.update({"is_favorite": new_status})
        logger.info(f"✅ Toggled favorite for item {item_id}: {new_status}")
        
        return new_status
        
    except Exception as e:
        logger.error(f"❌ Failed to toggle favorite: {e}")
        raise


logger.info("✅ Bag service loaded successfully")

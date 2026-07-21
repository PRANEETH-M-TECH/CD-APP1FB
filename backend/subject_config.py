"""
Subject Configuration for CHADUVU-GURU
Defines subject structure based on class levels
"""

# Subject structure by class range
SUBJECT_CONFIG = {
    # Classes 1-6: Traditional subjects including combined Science
    "1-6": {
        "subjects": ["english", "maths", "science", "social"],
        "icons": {
            "english": "📖",
            "maths": "🔢",
            "science": "🔬",
            "social": "🌍"
        },
        "colors": {
            "english": "#8b5cf6",
            "maths": "#f59e0b",
            "science": "#3b82f6",
            "social": "#10b981"
        }
    },
    
    # Classes 7-10: Science split into Physics and Biology
    "7-10": {
        "subjects": ["english", "maths", "physics", "biology", "social"],
        "icons": {
            "english": "📖",
            "maths": "🔢",
            "physics": "⚛️",
            "biology": "🧬",
            "social": "🌍"
        },
        "colors": {
            "english": "#8b5cf6",
            "maths": "#f59e0b",
            "physics": "#3b82f6",
            "biology": "#10b981",
            "social": "#ec4899"
        }
    }
}

# All possible subjects (for database queries and analytics)
ALL_SUBJECTS = ["english", "maths", "science", "physics", "biology", "social"]

# Mapping for backward compatibility (science -> physics/biology)
SUBJECT_MIGRATION_MAP = {
    "science": ["physics", "biology"]  # Old science queries can be categorized under both
}

def get_subjects_for_class(class_num: int) -> list:
    """
    Get the list of subjects for a given class number.
    
    Args:
        class_num: Class number (1-10)
    
    Returns:
        List of subject names
    """
    try:
        class_num = int(class_num)
        if 1 <= class_num <= 6:
            return SUBJECT_CONFIG["1-6"]["subjects"]
        elif 7 <= class_num <= 10:
            return SUBJECT_CONFIG["7-10"]["subjects"]
        else:
            # Default to all subjects for edge cases
            return ALL_SUBJECTS
    except (ValueError, TypeError):
        return ALL_SUBJECTS

def get_subject_icon(subject: str, class_num: int = None) -> str:
    """
    Get the emoji icon for a subject.
    
    Args:
        subject: Subject name
        class_num: Optional class number for context
    
    Returns:
        Emoji icon string
    """
    if class_num:
        class_num = int(class_num)
        config_key = "1-6" if class_num <= 6 else "7-10"
        return SUBJECT_CONFIG[config_key]["icons"].get(subject.lower(), "📚")
    
    # Default icons
    icons = {
        "english": "📖",
        "maths": "🔢",
        "science": "🔬",
        "physics": "⚛️",
        "biology": "🧬",
        "social": "🌍"
    }
    return icons.get(subject.lower(), "📚")

def get_subject_color(subject: str, class_num: int = None) -> str:
    """
    Get the color code for a subject.
    
    Args:
        subject: Subject name
        class_num: Optional class number for context
    
    Returns:
        Hex color code
    """
    if class_num:
        class_num = int(class_num)
        config_key = "1-6" if class_num <= 6 else "7-10"
        return SUBJECT_CONFIG[config_key]["colors"].get(subject.lower(), "#6b7280")
    
    # Default colors
    colors = {
        "english": "#8b5cf6",
        "maths": "#f59e0b",
        "science": "#3b82f6",
        "physics": "#3b82f6",
        "biology": "#10b981",
        "social": "#ec4899"
    }
    return colors.get(subject.lower(), "#6b7280")

def normalize_subject(subject: str) -> str:
    """
    Normalize subject name to lowercase.
    
    Args:
        subject: Subject name in any case
    
    Returns:
        Lowercase subject name
    """
    return subject.lower().strip()

def is_valid_subject(subject: str, class_num: int = None) -> bool:
    """
    Check if a subject is valid for the given class.
    
    Args:
        subject: Subject name
        class_num: Optional class number
    
    Returns:
        True if valid, False otherwise
    """
    subject = normalize_subject(subject)
    
    if class_num:
        valid_subjects = get_subjects_for_class(class_num)
        return subject in valid_subjects
    
    return subject in ALL_SUBJECTS

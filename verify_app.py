import sys
import os
sys.path.append(os.getcwd())

try:
    from backend.app import app
    print("Import backend.app successful")
except Exception as e:
    print(f"Import backend.app failed: {e}")
    sys.exit(1)

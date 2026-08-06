#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess

def main():
    print("==================================================")
    print("      CHADUVU-GURU AI ASSISTANT BOOTSTRAPPER     ")
    print("==================================================")

    # 1. Check/Set up Virtual Environment
    in_venv = (sys.prefix != sys.base_prefix) or hasattr(sys, 'real_prefix')
    
    if not in_venv:
        print("\n[!] Running outside a virtual environment.")
        venv_dir = os.path.join(os.getcwd(), ".venv")
        
        if not os.path.exists(venv_dir):
            print(f"[*] Creating Python virtual environment in '{venv_dir}'...")
            try:
                subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
                print("[OK] Virtual environment created successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to create virtual environment: {e}")
                sys.exit(1)
        else:
            print(f"[*] Found existing virtual environment folder: '{venv_dir}'")

        # Locate virtualenv's Python executable
        if os.name == "nt":
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(venv_dir, "bin", "python")

        if os.path.exists(venv_python):
            print("[*] Re-executing bootstrapper inside the virtual environment...")
            # Re-execute this script using the virtualenv's python
            cmd = [venv_python] + sys.argv
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
        else:
            print(f"[ERROR] Could not find python executable at: {venv_python}")
            sys.exit(1)

    print(f"\n[OK] Running inside virtual environment: {sys.prefix}")

    # 2. Install/Update Python Dependencies
    print("\n[*] Checking/Installing Python dependencies from requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("[OK] Python dependencies installed successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to install Python dependencies: {e}")
        sys.exit(1)

    # 3. Check / Install Node.js
    # Ensure path includes virtualenv's bin/Scripts
    venv_bin_dir = os.path.join(sys.prefix, "Scripts" if os.name == "nt" else "bin")
    os.environ["PATH"] = venv_bin_dir + os.pathsep + os.environ.get("PATH", "")

    # Look for node and npm
    node_path = shutil.which("node")
    npm_path = shutil.which("npm")
    
    if node_path and npm_path:
        print(f"[OK] Found Node.js: {node_path}")
        print(f"[OK] Found npm: {npm_path}")
    else:
        print("\n[*] Node.js or npm not found in system PATH.")
        print("[*] Automatically installing local Node.js environment via nodeenv...")
        try:
            # We call nodeenv to integrate node inside python's active virtualenv
            subprocess.run([sys.executable, "-m", "nodeenv", "--python-virtualenv"], check=True)
            print("[OK] Local Node.js and npm installed inside virtual environment.")
            
            # Re-verify they are found now
            node_path = shutil.which("node")
            npm_path = shutil.which("npm")
            if node_path and npm_path:
                print(f"[OK] Verified Node.js: {node_path}")
                print(f"[OK] Verified npm: {npm_path}")
            else:
                # Add check specifically for Windows path updates
                if os.name == "nt" and os.path.exists(os.path.join(sys.prefix, "Scripts", "node.exe")):
                    print("[OK] Verified local Node.js exists on Windows.")
                else:
                    print("[WARN] Local Node.js files exist, but could not resolve path in current shell.")
        except Exception as e:
            print(f"[ERROR] Failed to install local Node.js/npm using nodeenv: {e}")
            print("[WARN] Remotion-based storyboard video generation might not work.")

    # 4. Sync Node dependencies (run npm install)
    print("\n[*] Checking Node.js dependencies...")
    # Check root npm package
    if os.path.exists("package.json"):
        print("[*] Running 'npm install' in root folder...")
        try:
            # Use shell=True for windows to resolve npm command if globally installed
            subprocess.run("npm install", shell=True, check=True)
            print("[OK] Root npm packages updated.")
        except Exception as e:
            print(f"[WARN] Failed to install root npm packages: {e}")

    # 5. Check configuration files (.env)
    print("\n[*] Checking environment configuration...")
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("[*] Creating '.env' from '.env.example'...")
            shutil.copy(".env.example", ".env")
            print("[WARNING] Created default '.env'. PLEASE edit '.env' to specify your real API keys.")
        else:
            print("[WARN] No '.env' or '.env.example' file found.")
    else:
        print("[OK] '.env' configuration file is present.")

    # 5b. Check Firebase service account credentials
    # Firestore backs almost everything (auth, curriculum data, query
    # caching) - without this the server still starts (firebase_init.py
    # degrades gracefully, logging a warning instead of crashing), but
    # nearly every route will fail at runtime. Not an .env variable by
    # default - it's a JSON key file at the repo root (or the
    # FIREBASE_SERVICE_ACCOUNT_JSON / FIREBASE_CREDENTIALS env vars, for
    # deployments where a file isn't convenient).
    print("\n[*] Checking Firebase service account credentials...")
    has_env_creds = bool(os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or os.environ.get("FIREBASE_CREDENTIALS"))
    if os.path.exists("serviceAccountKey.json"):
        print("[OK] 'serviceAccountKey.json' is present.")
    elif has_env_creds:
        print("[OK] Firebase credentials supplied via environment variable.")
    else:
        print("[WARN] No 'serviceAccountKey.json' found and no FIREBASE_SERVICE_ACCOUNT_JSON/")
        print("       FIREBASE_CREDENTIALS environment variable set. The server will still")
        print("       start, but every Firestore-backed route (auth, curriculum, caching) will")
        print("       fail. Download a real service account key from your Firebase project")
        print("       (Project Settings -> Service Accounts -> Generate New Private Key),")
        print("       save it as 'serviceAccountKey.json' in the repo root, and re-run this")
        print("       script. 'serviceAccountKey.example.json' shows the expected shape.")

    # 6. Run the Application
    print("\n==================================================")
    print("      SETUP COMPLETE - RUNNING THE APPLICATION    ")
    print("==================================================")
    print("Starting FastAPI backend server...")
    print("Access the application at:")
    print("  - Main Interface: http://localhost:8000/user")
    print("  - Admin Dashboard: http://localhost:8000/admin")
    print("  - Enhanced Dashboard: http://localhost:8000/enhanced-dashboard")
    print("==================================================\n")
    
    try:
        # Run uvicorn server
        subprocess.run([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--reload"])
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")

if __name__ == "__main__":
    main()

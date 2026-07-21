import os
import sys
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Ensure Windows uses Proactor event loop for subprocess support
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# Load environment variables FIRST with override to prioritize .env file over system env vars
# Resolve root .env path relative to main.py
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MAIN_DIR, "..", ".."))
env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=env_path, override=True)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.retrieval import qdrant_service as qdrant
from backend.app.core.auth_middleware import auth_middleware
from backend.app.api.routes import (
    books_router,
    chat_router,
    dashboard_router,
    bag_router,
    profile_router,
    tts_router,
    visual_learning_router
)

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, initialize all models and database connections
    try:
        qdrant.initialize()
        print("[OK] Qdrant initialized successfully")
    except Exception as e:
        print(f"[WARN] Qdrant initialization failed: {e}")
        print("[WARN] Server will continue without Qdrant (some features may be limited)")
    yield
    # On shutdown

# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan, title="CHADUVU-GURU API Backend", version="1.0.0")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware
app.middleware("http")(auth_middleware)

# Register routes
app.include_router(books_router)
app.include_router(chat_router)
app.include_router(dashboard_router)
app.include_router(bag_router)
app.include_router(profile_router)
app.include_router(tts_router)
app.include_router(visual_learning_router)

# --- STATIC FILE SERVING ---
PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static files directories
app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# --- HTML TEMPLATE ROUTING ---
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(PUBLIC_DIR, 'index.html'))

@app.get("/enhanced-dashboard")
async def enhanced_dashboard_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'enhanced-dashboard.html'))

@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'admin.html'))

@app.get("/user")
async def user_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'user.html'))

@app.get("/chapters")
async def chapters_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'chapters.html'))

@app.get("/profile")
async def profile_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'profile.html'))

@app.get("/achievements")
async def achievements_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'achievements.html'))

@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'dashboard.html'))

@app.get("/mode-selection")
async def mode_selection_page():
    return FileResponse(os.path.join(PUBLIC_DIR, 'mode-selection.html'))

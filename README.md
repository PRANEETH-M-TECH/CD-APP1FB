# CHADUVU-GURU — AI Study Assistant

Chaduvu Guru is an AI-powered study assistant for CBSE students (Classes 6–10).
It answers questions grounded in the student's actual ingested textbook content
(retrieval-augmented generation over Firestore + Qdrant), and can generate a
full narrated, animated video lesson (the "HyperFrames" engine) instead of a
plain text answer when the topic calls for it.

For the full request pipeline — every stage, mapped to the exact file and
function responsible — see [`docs/APPLICATION_WORKFLOW.md`](docs/APPLICATION_WORKFLOW.md)
(a styled version is also at `docs/workflow.html`).

## Features

- Conversational chat interface, grounded in each student's real ingested textbooks (not generic web knowledge).
- Automatically generates a narrated, animated video lesson for concepts that need one, instead of just text.
- Class- and subject-aware retrieval (hybrid vector + keyword search over Qdrant).
- Per-scene text-to-speech narration (Sarvam), streamed to the student as it's generated.
- Student dashboards, streaks, and progress analytics.

## Project Structure

```
.
├── backend/
│   └── app/                    # The actual FastAPI application (backend/app/main.py is the entry point)
│       ├── api/routes/         # HTTP routes (chat, books, tts, dashboard, ...)
│       ├── core/                # Firebase/Firestore init, auth, redis, subject config
│       ├── services/            # Retrieval (Qdrant), visual_learning (HyperFrames), analytics, chat, llm
│       └── orchestrator_test/   # Classification/RAG prompt + test harnesses (dev tooling, not user-facing)
├── hyperframes_engine/          # Node.js video/animation engine (templates, renderer, GSAP timelines)
├── public/                      # Frontend static files (HTML, CSS, JS)
├── docs/                        # Architecture reference (kept in sync with the code — see above)
├── chapterdata/                 # Cached chapter metadata
├── requirements.txt             # Python dependencies
├── bootstrap.py                 # One-command setup + run (see below)
├── .env.example                 # Template for required environment variables
├── serviceAccountKey.example.json  # Template for the Firebase credentials file (see below)
└── README.md                    # This file
```

## Setup and Installation

### Credentials you'll need before running anything

This app needs two separate sets of credentials — both are excluded from git
via `.gitignore`, so you must supply your own copies locally:

1. **`.env`** — API keys and config (OpenAI, Qdrant, Sarvam TTS, Supabase, etc.).
   Copy `.env.example` to `.env` and fill in real values.
2. **`serviceAccountKey.json`** — a Firebase service account key, used for
   Firestore (auth, curriculum data, query caching — almost everything).
   Get one from your Firebase project: **Project Settings → Service Accounts
   → Generate New Private Key**, then save it as `serviceAccountKey.json` in
   the repo root. `serviceAccountKey.example.json` shows the expected shape.
   > If this file is missing, the server still starts (it degrades
   > gracefully rather than crashing), but nearly every route will fail at
   > runtime since almost everything reads from Firestore. For a deployment
   > where a file isn't convenient, you can instead set the
   > `FIREBASE_SERVICE_ACCOUNT_JSON` (or `FIREBASE_CREDENTIALS`) environment
   > variable to the full JSON contents.

> [!IMPORTANT]
> Never commit `.env` or `serviceAccountKey.json` — both contain real secrets.

---

### Method A: Automated Setup (Recommended)

`bootstrap.py` automates the whole process. It will:
- Create (or reuse) a Python virtual environment (`.venv`) and re-run itself inside it.
- Upgrade pip and install all Python dependencies from `requirements.txt`.
- Check for Node.js/npm; if missing, install a local copy inside `.venv` via `nodeenv` (needed for the HyperFrames video engine).
- Run `npm install` for the frontend's JS dependencies.
- Copy `.env.example` to `.env` if `.env` doesn't exist yet.
- Check for `serviceAccountKey.json` (or the equivalent environment variable) and warn clearly if it's missing.
- Start the FastAPI server.

**Steps:**
```bash
git clone <repository-url>
cd <repository-name>
python bootstrap.py
```

Then fill in real values in `.env` and add `serviceAccountKey.json` if
`bootstrap.py` warned you about either, and run `python bootstrap.py` again
(or just `uvicorn backend.app.main:app --reload`, since dependencies are
already installed at that point).

---

### Method B: Manual Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
2. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Node.js (v18+) and JS dependencies:**
   ```bash
   npm install
   ```
4. **Configure credentials** — see "Credentials you'll need" above:
   ```bash
   cp .env.example .env   # macOS/Linux
   copy .env.example .env # Windows CMD/PowerShell
   ```
   Then edit `.env` with real values, and place a real `serviceAccountKey.json`
   in the repo root.
5. **Run the server:**
   ```bash
   uvicorn backend.app.main:app --reload
   ```

---

## Application Entry Points

Once the server is running:
- **Main student interface**: [http://localhost:8000/user](http://localhost:8000/user)
- **Admin dashboard**: [http://localhost:8000/admin](http://localhost:8000/admin)
- **Enhanced dashboard**: [http://localhost:8000/enhanced-dashboard](http://localhost:8000/enhanced-dashboard)

## Dev tooling

`backend/app/orchestrator_test/` has two CLI test harnesses useful when
debugging classification/RAG or video-generation quality without needing a
full browser session:
- `test_orchestrator_cli.py` — exercises classification + RAG retrieval only (no video, no TTS).
- `test_runner.py` — the underlying pipeline both harnesses and the real app share.

Both need `.env` and `serviceAccountKey.json` configured the same way as the
main app, and hit the real Firestore/Qdrant/OpenAI backends (no mocking).

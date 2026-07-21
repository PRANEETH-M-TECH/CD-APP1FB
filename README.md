# CHADUVU-GURU - AI Study Assistant

CHADUVU-GURU is an AI-powered study assistant that helps students understand their textbooks better. It provides a conversational interface to ask questions about the content of uploaded PDF textbooks.

## Features

-   Upload PDF textbooks.
-   Conversational chat interface to ask questions.
-   Support for different classes and subjects.
-   Summarization of chapters.
-   Conversational mode for a more natural interaction.

## Project Structure

```
.
├── backend/         # Contains the FastAPI backend application
│   ├── app.py       # Main FastAPI application
│   └── qdrant.py    # Qdrant client and helper functions
├── public/          # Frontend static files (HTML, CSS, JS)
├── uploads/         # Directory for uploaded PDF files
├── chapterdata/     # Cached chapter data
├── chpchunks/       # Cached chapter chunks
├── summary/         # Cached chapter summaries
├── bm25_indices/    # Cached BM25 indices
├── requirements.txt # Python dependencies
├── .gitignore       # Files and directories to be ignored by Git
└── README.md        # This file
```

## Setup and Installation

### Git and Environment Credentials
> [!IMPORTANT]
> **Do not commit your real `.env` file to GitHub.** It contains sensitive API keys. The `.env` file is excluded from git in `.gitignore`. Instead:
> 1. You will find `.env.example` in the root folder, which contains blank configuration variables.
> 2. Copy `.env.example` to `.env` and fill in your actual credentials locally.

You can set up and run the application using either the **Automated Bootstrapper** (recommended for quick setup) or **Manually**.

---

### Method A: Automated Setup (Recommended)

A unified bootstrapper `bootstrap.py` is provided to fully automate the process. It will:
- Check for or create a Python virtual environment (`.venv`).
- Upgrade pip and install all Python dependencies from `requirements.txt`.
- Check if Node.js/npm is installed on the machine. If not, it automatically installs a local version of Node.js inside the `.venv` using `nodeenv`.
- Run `npm install` to sync packages in the root and in the `remotion_test_app/` folder.
- Copy `.env.example` to `.env` if `.env` does not exist.
- Start the FastAPI application.

**Commands to run:**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Run the bootstrapper:
   ```bash
   python bootstrap.py
   ```

---

### Method B: Manual Setup

If you prefer to configure your environment manually without using the bootstrap script:

1. **Activate Python Virtual Environment:**
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```
2. **Install Python Packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Node.js & npm dependencies:**
   Make sure you have Node.js (v18+) installed globally, then run:
   ```bash
   # Sync root packages
   npm install
   
   # Sync Remotion app packages
   cd remotion_test_app
   npm install
   cd ..
   ```
4. **Configure Credentials:**
   Copy the example environment file and edit the keys:
   ```bash
   cp .env.example .env   # On macOS/Linux
   copy .env.example .env # On Windows CMD/PowerShell
   ```
   Open `.env` and insert your `GOOGLE_API_KEY`, `SARVAM_API_KEY`, etc.
5. **Run the Server:**
   ```bash
   uvicorn backend.app.main:app --reload
   ```

---

## Running Standalone Remotion Storyboard Tests

If you want to test storyboard generation, audio synthesis, and video rendering/previewing **without starting the full application server**, you can run the standalone test runner:

```bash
python run_visual_learning_test.py
```

### What this test script does:
1. Prompts you to enter a query (e.g. *"explain structure of neuron"*).
2. Contacts Gemini to generate a video storyboard sequence.
3. Automatically retrieves and downloads relevant educational images and icons.
4. Synthesizes voice narration using Sarvam AI and saves the audio files.
5. Launches a prompt selection where you can choose:
   - **Option [1] (Preview)**: Launches the local web player preview.
   - **Option [2] (Render)**: Compiles the storyboard directly into a standalone MP4 video in `remotion_test_app/outputs/output_videos/`.

---

## Application Entry Points

When the main server is running:
- **Main User Interface**: [http://localhost:8000/user](http://localhost:8000/user)
- **Admin Dashboard**: [http://localhost:8000/admin](http://localhost:8000/admin)
- **Enhanced Dashboard**: [http://localhost:8000/enhanced-dashboard](http://localhost:8000/enhanced-dashboard)


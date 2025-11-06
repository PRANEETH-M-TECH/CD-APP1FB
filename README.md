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

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add the following variables. See `.env.example` for a template.

    ```
    # Qdrant configuration
    QDRANT_URL="http://localhost:6333"
    QDRANT_API_KEY=""

    # Google Gemini API Key
    GOOGLE_API_KEY="your-google-api-key"
    ```

    You will also need to have a Qdrant instance running. You can use the official Docker image to run it locally:
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

## Running the Application

To run the backend server, use the following command:

```bash
uvicorn backend.app:app --reload
```

The application will be available at `http://localhost:8000`.

You can access the main user interface at `http://localhost:8000/user`.
The admin interface is at `http://localhost:8000/admin`.


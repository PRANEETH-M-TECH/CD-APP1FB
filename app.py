
import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict

from qrdant import (
    process_and_embed_book,
    get_books,
    semantic_search,
    generate_answer,
)

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for books (replace with a database in a real application)
BOOKS_DB = []

# Create uploads directory if it doesn't exist
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

class Chapter(BaseModel):
    name: str
    start_page: int
    end_page: int

class Book(BaseModel):
    title: str
    class_name: str
    subject: str
    chapters: List[Chapter]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles PDF file uploads from the admin panel.
    """
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

@app.post("/api/books")
async def create_book(
    title: str = Form(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    chapters: str = Form(...) # JSON string of chapters
):
    """
    Processes and stores book and chapter metadata.
    """
    import json
    chapters_list = json.loads(chapters)
    
    # For now, we assume one PDF per book, and the latest upload is the one to use.
    # In a real app, you'd have a more robust way of linking uploads to books.
    uploaded_files = sorted(os.listdir(UPLOADS_DIR), key=lambda f: os.path.getmtime(os.path.join(UPLOADS_DIR, f)))
    if not uploaded_files:
        return {"error": "No file uploaded"}, 400
    
    pdf_path = os.path.join(UPLOADS_DIR, uploaded_files[-1])
    
    # Process the book and embed its content
    process_and_embed_book(pdf_path, title, class_name, subject, chapters_list)
    
    book_data = {
        "title": title,
        "class_name": class_name,
        "subject": subject,
        "chapters": chapters_list
    }
    BOOKS_DB.append(book_data)
    
    return {"message": "Book added successfully!", "book": book_data}


@app.get("/api/books")
async def list_books():
    """
    Returns a list of available books.
    """
    return get_books()

@app.post("/api/query")
async def query_book(query: str, book_title: str):
    """
    Performs a semantic search within a specific book and returns a generated answer.
    """
    search_results = semantic_search(book_title, query)
    
    # For simplicity, we'll take the top few results to generate an answer
    context = " ".join([result.payload['text'] for result in search_results])
    
    answer = generate_answer(query, context)
    
    return {"answer": answer, "sources": search_results}

# Mount the 'public' directory to serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
async def read_root():
    """
    Serves the main landing page.
    """
    return FileResponse('public/index.html')

@app.get("/admin")
async def admin_page():
    return FileResponse('public/admin.html')

@app.get("/user")
async def user_page():
    return FileResponse('public/user.html')


import os
import json
import hashlib
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
from dotenv import load_dotenv
from typing import List, Dict

# --- CONFIGURATION ---
COLLECTION_NAME = "test_data"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def get_client():
    """Initializes and returns the Qdrant client."""
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise ValueError("QDRANT_URL must be set in the environment.")
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return client

def get_book_uuid(file_path):
    """Generates a unique UUID for a book using SHA256 hash of its content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_if_book_exists(client, book_uuid):
    """Checks if a book with the given UUID already exists in the collection."""
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="textbook_uuid",
                    match=models.MatchValue(value=book_uuid),
                ),
            ]
        ),
        limit=1,
    )
    return len(response[0]) > 0

def get_pdf_text(pdf_path, start_page, end_page):
    """Extracts text from a specified page range in a PDF."""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page_num in range(start_page - 1, end_page):
        if 0 <= page_num < len(pdf_reader.pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_user_input():
    """Gets metadata from the user interactively."""
    class_name = input("Enter the class name: ")
    subject = input("Enter the subject: ")
    
    while True:
        try:
            num_chapters = int(input("Enter the number of chapters: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    chapters = []
    for i in range(num_chapters):
        print(f"\n--- Chapter {i+1} ---")
        chapter_name = input(f"Enter the name for chapter {i+1}: ")
        while True:
            try:
                start_page = int(input(f"Enter the starting page for chapter '{chapter_name}': "))
                end_page = int(input(f"Enter the ending page for chapter '{chapter_name}': "))
                if start_page > 0 and end_page >= start_page:
                    break
                else:
                    print("Invalid page numbers. Ensure start page is > 0 and end page is >= start page.")
            except ValueError:
                print("Invalid input. Please enter numbers for pages.")
        
        chapters.append({"name": chapter_name, "start_page": start_page, "end_page": end_page})
        
    return class_name, subject, chapters

def upload_book(client, model, pdf_path, class_name, subject, chapters):
    """Processes, chunks, and uploads a book to Qdrant."""
    book_uuid = get_book_uuid(pdf_path)

    if check_if_book_exists(client, book_uuid):
        print(f"This book (UUID: {book_uuid}) already exists in the collection. Skipping upload.")
        return

    print(f"Processing new book with UUID: {book_uuid}")
    print(f"Class: {class_name}, Subject: {subject}")

    for chapter in chapters:
        print(f"\nProcessing Chapter: '{chapter['name']}' (Pages: {chapter['start_page']}-{chapter['end_page']})...")
        chapter_text = get_pdf_text(pdf_path, chapter['start_page'], chapter['end_page'])
        text_chunks = get_text_chunks(chapter_text)
        print(f"  - Created {len(text_chunks)} text chunks.")

        points_to_upload = []
        for i, chunk in enumerate(text_chunks):
            embedding = model.encode(chunk, convert_to_tensor=False)
            points_to_upload.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "class": class_name,
                        "subject": subject,
                        "textbook_uuid": book_uuid,
                        "chapter": chapter['name'],
                        "start_page": chapter['start_page'],
                        "end_page": chapter['end_page'],
                        "text": chunk,
                    }
                )
            )

        if points_to_upload:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upload,
                wait=True
            )
            print(f"  - Uploaded {len(points_to_upload)} points to Qdrant.")

    print("\nProcessing complete!")

def get_chapters(client: QdrantClient, textbook_uuid: str) -> List[str]:
    """Returns a list of distinct chapters for a given textbook UUID."""
    seen_chapters = set()
    next_offset = None
    
    while True:
        response, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=textbook_uuid))
                ]
            ),
            limit=100,
            offset=next_offset,
            with_payload=["chapter"]
        )
        
        if not response:
            break
            
        for point in response:
            seen_chapters.add(point.payload['chapter'])
            
        if next_offset is None:
            break
            
    return sorted(list(seen_chapters))

def get_chunks_by_chapter(client: QdrantClient, textbook_uuid: str, chapter_name: str) -> List[Dict]:
    """Returns all text chunks for a given chapter in a textbook."""
    all_chunks = []
    next_offset = None

    while True:
        response, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=textbook_uuid)),
                    models.FieldCondition(key="chapter", match=models.MatchValue(value=chapter_name))
                ]
            ),
            limit=250,
            offset=next_offset,
            with_payload=True
        )
        
        if not response:
            break
        
        for point in response:
            all_chunks.append(point.payload)
            
        if next_offset is None:
            break
            
    return all_chunks

def get_available_books(client: QdrantClient) -> Dict[str, str]:
    """Fetches a dictionary of available books (UUID -> textbook_name)."""
    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=["textbook_uuid"]
    )
    
    return {p.payload['textbook_uuid']: f"Book-{p.payload['textbook_uuid'][:8]}" for p in response}

def semantic_search(client: QdrantClient, model: SentenceTransformer, query: str, book_uuid: str, top_k: int = 5, metadata_filters: dict = None) -> List[str]:
    """Performs semantic search within a specific book."""
    query_embedding = model.encode(query, convert_to_tensor=False).tolist()
    
    filter_conditions = [models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
    if metadata_filters:
        for key, value in metadata_filters.items():
            filter_conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=models.Filter(must=filter_conditions),
        limit=top_k,
        with_payload=["text"]
    )
    
    return [hit.payload['text'] for hit in search_result]

# --- FIXED COLLECTION & INDEX HANDLING ---
def ensure_payload_indexes(client: QdrantClient, fields: list):
    """Ensure the payload indexes exist for all specified fields."""
    for field in fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            if "already exists" not in str(e):
                print(f"Error creating index for {field}: {e}")

def create_collection_if_not_exists(client: QdrantClient, model: SentenceTransformer):
    """
    Checks if the Qdrant collection exists and creates it if not.
    It also verifies vector dimensions if the collection already exists.
    """
    model_embedding_dimension = model.get_sentence_embedding_dimension()
    
    # Use the recommended way to check for collection existence
    if client.collection_exists(collection_name=COLLECTION_NAME):
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        existing_dimension = collection_info.config.params.vectors.size
        
        if existing_dimension == model_embedding_dimension:
            print(f"Collection '{COLLECTION_NAME}' already exists with correct dimensions.")
        else:
            print(
                f"Warning: Collection '{COLLECTION_NAME}' exists with dimension "
                f"{existing_dimension}, but model has dimension {model_embedding_dimension}."
            )
            print(f"Please delete the collection and restart the script.")
            # Exit or raise an error to prevent uploading mismatched data
            raise ValueError("Vector dimension mismatch. Please resolve the collection.")
    else:
        # If it does not exist, create it
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=model_embedding_dimension,
                distance="Cosine"
            ),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")

    # Ensure payload indexes are created
    print("Verifying payload indexes...")
    for field in ["class", "subject", "chapter", "textbook_uuid"]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            # This can happen if the index already exists, which is fine.
            # A more robust check might be needed for production, but this is okay for now.
            if "already exists" not in str(e).lower():
                 print(f"Could not create index for field '{field}'. Reason: {e}")
    print(f"Collection '{COLLECTION_NAME}' is ready.")

def main():
    """Interactive upload script."""
    try:
        client = get_client()
        model = SentenceTransformer(EMBEDDING_MODEL)
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        return

    try:
        create_collection_if_not_exists(client, model)
    except Exception as e:
        print(f"Could not create or check collection: {e}")
        return

    while True:
        pdf_path = input("\nEnter the path to the PDF file to upload (or 'quit' to exit): ")
        if pdf_path.lower() == 'quit':
            break
        
        if not os.path.exists(pdf_path) or not pdf_path.lower().endswith('.pdf'):
            print("Error: The file does not exist or is not a PDF. Please provide a valid path.")
            continue

        class_name, subject, chapters = get_user_input()
        upload_book(client, model, pdf_path, class_name, subject, chapters)

if __name__ == "__main__":
    main()

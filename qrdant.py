import os
import uuid
import hashlib
import json
from qdrant_client import QdrantClient as QC, models
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

# --- CONFIGURATION ---
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "my_documents")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- GLOBAL VARIABLES (INITIALIZED ON STARTUP) ---
# These are initialized in the `initialize` function, not on import
client: QC | None = None
local_embedder: SentenceTransformer | None = None
generation_model: genai.GenerativeModel | None = None

def initialize():
    """
    Initializes all models and the Qdrant client.
    This function is called once when the FastAPI application starts.
    """
    global client, local_embedder, generation_model

    local_embedder = SentenceTransformer(EMBEDDING_MODEL)
    GENERATION_MODEL_NAME = "models/gemini-flash-latest"
    generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
    client = QC(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )

    # --- Collection Management ---
    model_embedding_dimension = local_embedder.get_sentence_embedding_dimension()
    
    # Use the robust `collection_exists` check
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=model_embedding_dimension,
                distance=models.Distance.COSINE
            ),
        )
        # Create payload indexes only when the collection is first created
        for field in ["class_name", "subject", "chapter", "textbook_uuid"]:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass # Avoid crashing if index creation fails for some reason


# --- HELPER FUNCTIONS ---
def get_book_uuid(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_if_book_exists(book_uuid: str) -> bool:
    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
        ),
        limit=1,
    )
    return len(response) > 0

# --- CORE LOGIC ---
def process_and_embed_book(pdf_path: str, class_name: str, subject: str, chapters: list):
    print(f"\n--- Starting Book Processing ---")
    print(f"File: {os.path.basename(pdf_path)}")
    print(f"Class: {class_name}, Subject: {subject}")

    book_uuid = get_book_uuid(pdf_path)
    
    # If book exists, delete all its old entries before re-embedding to ensure an overwrite
    if check_if_book_exists(book_uuid):
        print(f"Book with UUID {book_uuid} already exists. Deleting old entries...")
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="textbook_uuid",
                            match=models.MatchValue(value=book_uuid),
                        )
                    ]
                )
            ),
        )
        print("Old entries deleted. Proceeding with re-embedding...")

    print("Reading PDF and splitting text...")
    reader = PdfReader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    total_chapters = len(chapters)
    print(f"Found {total_chapters} chapters to process.")

    for i, chapter in enumerate(chapters):
        chapter_name = chapter['name']
        print(f"\nProcessing Chapter {i+1}/{total_chapters}: '{chapter_name}'...")
        
        start_page, end_page = chapter['start_page'], chapter['end_page']
        chapter_text = ""
        for page_num in range(start_page - 1, end_page):
            if 0 <= page_num < len(reader.pages):
                chapter_text += reader.pages[page_num].extract_text() or ""
        
        text_chunks = text_splitter.split_text(chapter_text)
        print(f"Split chapter into {len(text_chunks)} text chunks.")
        
        points_to_upload = []
        for chunk in text_chunks:
            embedding = local_embedder.encode(chunk).tolist()
            points_to_upload.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "class_name": class_name, "subject": subject, "textbook_uuid": book_uuid,
                        "filename": os.path.basename(pdf_path),
                        "chapter": chapter_name, "start_page": start_page, "end_page": end_page, "text": chunk,
                    }
                )
            )
        
        if points_to_upload:
            print(f"Embedding complete. Uploading {len(points_to_upload)} points to Qdrant...")
            client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)
            print(f"Chapter '{chapter_name}' processed and saved successfully.")

    print("\n--- Book Processing Complete ---")
    # This function runs in the background, so it does not return a value to the client.
    return

def get_books(class_name: str = None, subject: str = None) -> List[Dict[str, str]]:

    filter_conditions = []

    if class_name:

        filter_conditions.append(models.FieldCondition(key="class_name", match=models.MatchValue(value=class_name)))



    # Always fetch all points for the given class_name first, then filter by subject in Python

    # because Qdrant's 'match' filter is case-sensitive.

    scroll_filter = models.Filter(must=filter_conditions) if filter_conditions else None



    response, _ = client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter=scroll_filter,

        limit=1000, # Limit to a reasonable number

        with_payload=["textbook_uuid", "subject", "class_name", "filename"]

    )

    

    unique_books = {}

    for p in response:

        book_uuid = p.payload.get('textbook_uuid')

        payload_subject = p.payload.get('subject')



        # Case-insensitive filtering for subject in Python

        if subject and payload_subject and subject.lower() != payload_subject.lower():

            continue # Skip if subject doesn't match case-insensitively



        if book_uuid and book_uuid not in unique_books:

            unique_books[book_uuid] = {

                "id": book_uuid, 

                "subject": payload_subject, # Use the original casing from payload

                "class_name": p.payload.get('class_name', 'N/A'),

                "filename": p.payload.get('filename')

            }

    

    return list(unique_books.values())



def get_chapter_names(book_uuid: str) -> List[str]:



    """Retrieves a list of unique chapter names for a book."""



    if not client:



        raise RuntimeError("Qdrant client not initialized.")



    



    response, _ = client.scroll(



        collection_name=COLLECTION_NAME,



        scroll_filter=models.Filter(



            must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]



        ),



        limit=1000, # Assuming a book won't have more than 1000 chunks with unique chapter names



        with_payload=["chapter"]



    )



    



    unique_names = set()



    for point in response:



        if name := point.payload.get("chapter"):



            unique_names.add(name)



            



    return sorted(list(unique_names))







def get_chapters_for_book(book_uuid: str) -> List[Dict]:



    """



    Retrieves a sorted list of unique chapters with their page ranges for a given book UUID.



    This implementation follows the user's suggested logic.



    """



    if not client:



        raise RuntimeError("Qdrant client not initialized.")







    # 1. Get all unique chapter names first.



    chapter_names = get_chapter_names(book_uuid)



    



    if not chapter_names:



        return []







    chapter_info = []



    # 2. For each chapter, find its page range.



    for name in chapter_names:



        # We only need one chunk to get the page range since it's the same for all chunks in a chapter.



        response, _ = client.scroll(



            collection_name=COLLECTION_NAME,



            scroll_filter=models.Filter(



                must=[



                    models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid)),



                    models.FieldCondition(key="chapter", match=models.MatchValue(value=name))



                ]



            ),



            limit=1,



            with_payload=["start_page", "end_page"]



        )



        



        start_page, end_page = None, None



        if response:



            payload = response[0].payload



            start_page = payload.get("start_page")



            end_page = payload.get("end_page")



            



        chapter_info.append({



            "name": name,



            "start_page": start_page,



            "end_page": end_page



        })







    # The chapter_names were already sorted, so the final list should be too.



    return chapter_info





def semantic_search(book_uuid: str, query: str, metadata_filters: dict = None) -> List[dict]:
    query_embedding = local_embedder.encode(query).tolist()
    filter_conditions = [models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
    if metadata_filters:
        for key, value in metadata_filters.items():
            if value:
                filter_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    search_result = client.search(
        collection_name=COLLECTION_NAME, query_vector=query_embedding,
        query_filter=models.Filter(must=filter_conditions), limit=5, with_payload=True
    )
    return [hit.payload for hit in search_result]

def generate_answer(query: str, context: str) -> str:
    prompt = f"""Answer the user's question based on the context provided.

Context:
{context}

Question:
{query}

Answer:
"""
    response = generation_model.generate_content(prompt)
    return response.text

def generate_chapters_from_text(text_sample: str) -> str:
    """
    Uses the generative model to extract a complete chapter list from the book's text.
    """
    prompt = f'''
    You are an expert assistant tasked with analyzing the text of a textbook to identify its chapter structure.
    Analyze the following text, which includes the table of contents and opening pages of a book.
    Your goal is to extract all the chapters, along with their starting and ending page numbers.

    Please return your response as a single, valid JSON object that strictly follows this schema:
    {{
      "chapters": [
        {{
          "chapter_name": "The name of the chapter",
          "start_page": <integer>,
          "end_page": <integer>
        }}
      ]
    }}

    - "chapter_name" should be the full name of the chapter.
    - "start_page" is the page number where the chapter begins.
    - "end_page" is the page number where the chapter ends. The end page of one chapter is typically the page before the start page of the next chapter. The final chapter should end on the last page of the book.

    Do not include any text or explanation outside of the JSON object.

    Here is the text from the book:
    ---
    {text_sample}
    ---
    '''
    try:
        response = generation_model.generate_content(prompt)
        text = response.text.strip()

        # Find the start of the JSON object and its corresponding end
        json_start = text.find('{')
        if json_start == -1:
            print("LLM response did not contain a JSON object.")
            return '{{ "chapters": [] }}'

        # Find the matching closing brace for the opening brace
        open_braces = 0
        json_end = -1
        for i, char in enumerate(text[json_start:]):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            
            if open_braces == 0:
                json_end = json_start + i + 1
                break
        
        if json_end == -1:
            print("Could not find matching closing brace in LLM response.")
            return '{{ "chapters": [] }}'

        clean_json = text[json_start:json_end]
        return clean_json

    except Exception as e:
        print(f"Error during LLM chapter generation: {e}")
        return '{{ "chapters": [] }}'

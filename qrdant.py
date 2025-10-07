
import os
from qdrant_client import QdrantClient as QC, models
from google.generativeai import GenerativeModel
from pypdf import PdfReader

# Qdrant configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "chaduvu_guru_collection"

# Gemini model for embeddings and generation
EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATION_MODEL_NAME = "gemini-1.0-pro"

embedding_model = GenerativeModel(EMBEDDING_MODEL_NAME)
generation_model = GenerativeModel(GENERATION_MODEL_NAME)

# Initialize Qdrant client
client = QC(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Create collection if it doesn't exist
try:
    client.get_collection(collection_name=COLLECTION_NAME)
except Exception:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )


def process_and_embed_book(pdf_path: str, title: str, class_name: str, subject: str, chapters: list):
    """
    Extracts text from a PDF, processes it in chapters, and embeds it in Qdrant.
    """
    reader = PdfReader(pdf_path)
    points_to_upsert = []

    for i, chapter in enumerate(chapters):
        chapter_text = ""
        # Note: PyPDF pages are 0-indexed, so we subtract 1
        for page_num in range(chapter['start_page'] - 1, chapter['end_page']):
            if page_num < len(reader.pages):
                chapter_text += reader.pages[page_num].extract_text()

        if chapter_text:
            # Generate embedding for the chapter text
            embedding_result = embedding_model.embed_content([chapter_text])
            text_embedding = embedding_result["embedding"]
            
            # Create a point to upsert into Qdrant
            points_to_upsert.append(
                models.PointStruct(
                    id=f"{title}_{i}",  # Unique ID for each chapter
                    vector=text_embedding,
                    payload={
                        "text": chapter_text,
                        "book_title": title,
                        "class_name": class_name,
                        "subject": subject,
                        "chapter_name": chapter['name']
                    },
                )
            )

    if points_to_upsert:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True,
        )

def get_books():
    """
    Retrieves a list of unique book titles from the collection.
    """
    # This is a simplified way to get unique books. 
    # For larger datasets, you might manage this list separately.
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000, # Adjust as needed
        with_payload=["book_title"],
        with_vectors=False,
    )[0]
    
    unique_books = set()
    for record in response:
        unique_books.add(record.payload['book_title'])
        
    return list(unique_books)

def semantic_search(book_title: str, query: str):
    """
    Performs a semantic search within a specific book.
    """
    # Generate embedding for the query
    query_embedding_result = embedding_model.embed_content([query])
    query_embedding = query_embedding_result["embedding"]
    
    # Search for similar vectors in Qdrant, filtered by book_title
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="book_title",
                    match=models.MatchValue(value=book_title),
                )
            ]
        ),
        limit=3,  # Return top 3 most relevant chapters
    )
    
    return search_result

def generate_answer(query: str, context: str) -> str:
    """
    Generates a natural language answer based on the query and context.
    """
    prompt = f"""Answer the user's question based on the context provided.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = generation_model.generate_content(prompt)
    return response.text


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

from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "my_documents")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- GLOBAL VARIABLES (INITIALIZED ON STARTUP) ---
# These are initialized in the `initialize` function, not on import
client: QC | None = None
local_embedder: SentenceTransformer | None = None
generation_model: genai.GenerativeModel | None = None
bm25_indices: dict = {}
book_corpus: dict = {}

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

def _get_all_chunks_for_book(book_uuid: str) -> List[Dict]:
    """Scrolls through all points in a collection for a given book_uuid."""
    if not client:
        raise RuntimeError("Qdrant client not initialized.")
    
    all_points = []
    next_offset = None
    
    while True:
        response, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
            ),
            limit=250,
            with_payload=True,
            offset=next_offset
        )
        all_points.extend(response)
        if not next_offset:
            break
            
    return [point.payload for point in all_points]

def get_or_build_bm25_index(book_uuid: str):
    """
    Builds or retrieves a cached BM25 index for a given book.
    """
    if book_uuid in bm25_indices:
        return bm25_indices[book_uuid]

    print(f"Building BM25 index for book: {book_uuid}")
    corpus_docs = _get_all_chunks_for_book(book_uuid)
    
    if not corpus_docs:
        return None

    # Store the full payload for later retrieval
    book_corpus[book_uuid] = corpus_docs
    
    # Tokenize the text content for BM25
    tokenized_corpus = [doc["text"].split(" ") for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Cache the index
    bm25_indices[book_uuid] = bm25
    return bm25


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

def get_book_metadata(book_uuid: str) -> dict:
    """Retrieves metadata for a specific book UUID."""
    if not client:
        raise RuntimeError("Qdrant client not initialized.")
    
    # Scroll for one point to get the metadata
    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
        ),
        limit=1,
        with_payload=["class_name", "subject"]
    )
    
    if response:
        payload = response[0].payload
        return {
            "class_name": payload.get("class_name"),
            "subject": payload.get("subject")
        }
    return {}


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





    # Sort the final list by the starting page number to ensure correct book order.



    chapter_info.sort(key=lambda x: (x.get('start_page') or 0))



    return chapter_info


def hybrid_search(book_uuid: str, query: str, classification: str, metadata_filters: dict = None) -> List[dict]:
    """
    Performs a hybrid search using both dense and sparse retrieval methods,
    normalizing their scores and weighting them based on the query classification.
    """
    # 1. Get BM25 index and corpus
    bm25 = get_or_build_bm25_index(book_uuid)
    corpus = book_corpus.get(book_uuid, [])

    if not bm25 or not corpus:
        print("Warning: BM25 index not found. Falling back to pure semantic search.")
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
        return [(res.score, res.payload) for res in search_result]

    # 2. Perform Dense (Vector) Search
    query_embedding = local_embedder.encode(query).tolist()
    filter_conditions = [models.FieldCondition(key="textbook_uuid", match=models.MatchValue(value=book_uuid))]
    if metadata_filters:
        for key, value in metadata_filters.items():
            if value:
                filter_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    dense_search_result = client.search(
        collection_name=COLLECTION_NAME, 
        query_vector=query_embedding,
        query_filter=models.Filter(must=filter_conditions), 
        limit=10, 
        with_payload=True
    )
    dense_results = {res.payload['text']: res.score for res in dense_search_result}
    print("\n--- Dense (Semantic) Search Results ---")
    print(f"Retrieved {len(dense_results)} results.")
    for i, res in enumerate(dense_search_result):
        print(f"  {i+1}. Score: {res.score:.4f} | Text: {res.payload['text'][:100]}...")

    # 3. Perform Sparse (BM25) Search
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    sparse_results_with_scores = []
    for i, doc in enumerate(corpus):
        sparse_results_with_scores.append((bm25_scores[i], doc))
    
    sparse_results_with_scores.sort(key=lambda x: x[0], reverse=True)
    top_10_sparse = sparse_results_with_scores[:10]
    sparse_results = {doc['text']: score for score, doc in top_10_sparse}
    print("\n--- Sparse (BM25) Search Results ---")
    print(f"Retrieved {len(sparse_results)} results.")
    for i, (score, doc) in enumerate(top_10_sparse):
        print(f"  {i+1}. Score: {score:.4f} | Text: {doc['text'][:100]}...")

    # 4. Normalize, Combine, and Re-rank
    alpha = 0.7 if classification == 'conceptual' else 0.3

    all_doc_texts = list(set(dense_results.keys()) | set(sparse_results.keys()))
    
    # Min-Max Normalization
    max_dense = max(dense_results.values()) if dense_results else 0
    min_dense = min(dense_results.values()) if dense_results else 0
    max_sparse = max(sparse_results.values()) if sparse_results else 0
    min_sparse = min(sparse_results.values()) if sparse_results else 0

    fused_results = []
    for text in all_doc_texts:
        # Normalize dense score to 0-1 range
        dense_score = dense_results.get(text, 0)
        norm_dense = (dense_score - min_dense) / (max_dense - min_dense) if (max_dense - min_dense) > 0 else 0
        
        # Normalize sparse score to 0-1 range
        sparse_score = sparse_results.get(text, 0)
        norm_sparse = (sparse_score - min_sparse) / (max_sparse - min_sparse) if (max_sparse - min_sparse) > 0 else 0

        fused_score = (alpha * norm_dense) + ((1 - alpha) * norm_sparse)
        
        original_payload = next((doc for doc in corpus if doc['text'] == text), None)
        if original_payload:
            fused_results.append((fused_score, original_payload))

    fused_results.sort(key=lambda x: x[0], reverse=True)
    
    final_results = fused_results[:5]
    print("\n--- Final Fused & Ranked Results ---")
    print(f"Returning {len(final_results)} results after Hybrid Fusion.")
    for i, (score, payload) in enumerate(final_results):
        print(f"  {i+1}. Fused Score: {score:.4f} | Text: {payload['text'][:100]}...")
    print("-" * 30)

    return final_results



def reformulate_and_classify_query(query: str, class_name: str = None, subject: str = None, chapter_list: list = None) -> dict:
    """
    Uses the generative model to reformulate the user's query and classify it, using book context.
    """
    context_prompt = ""
    if class_name and subject:
        context_prompt += f"The user is asking a question about a textbook for Class '{class_name}', Subject '{subject}'.\n"
    if chapter_list:
        chapters_formatted = "\n".join([f"- {name}" for name in chapter_list])
        context_prompt += f"The book contains the following chapters:\n{chapters_formatted}\n"

    prompt = f'''
    You are an expert in query analysis. Your task is to reformulate a user's query to be more effective for a semantic search system and to classify the query's intent, using the provided context about the textbook.

    **Textbook Context:**
    {context_prompt}

    Based on the context and the user's query, perform the following two tasks:

    1.  **Reformulate the Query**: Rephrase the query to be clearer and more specific for a vector database search. Use the context to make the query less ambiguous. For example, if the query is "what is the first law?" and the context mentions "Newton's Laws", reformulate it to "What is Newton's First Law of Motion?".
    2.  **Classify the Query**: Determine if the query is 'factual' or 'conceptual'.
        -   'factual': The query asks for specific, concrete pieces of information (e.g., "What is the formula for momentum?", "Who discovered penicillin?").
        -   'conceptual': The query asks for explanations, summaries, or comparisons (e.g., "Explain the difference between heat and temperature.", "Summarize the causes of World War I.").

    Return your response as a single, valid JSON object with two keys: "reformulated_query" and "classification".

    **User Query:**
    "{query}"

    **JSON Response:**
    '''
    try:
        response = generation_model.generate_content(prompt)
        # Basic parsing to find the JSON object
        json_text = response.text.strip()
        json_start = json_text.find('{')
        json_end = json_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            clean_json = json_text[json_start:json_end]
            return json.loads(clean_json)
        else:
            # Fallback if JSON is not found
            return {
                "reformulated_query": query,
                "classification": "conceptual" # Default to conceptual
            }
    except Exception as e:
        print(f"Error during query reformulation/classification: {e}")
        # Fallback in case of any error
        return {
            "reformulated_query": query,
            "classification": "conceptual"
        }

def generate_answer(query: str, context: str, class_name: str) -> str:
    # Determine the class_level string
    class_level = f"Class {class_name}" if class_name else "student"

    prompt = f"""
    **Your Role:** You are CHADUVU-GURU, an intelligent and friendly AI tutor for a {class_level}. Your goal is to explain topics clearly and interactively.

    **Core Instructions:**
    1.  Your primary source of information is the provided "Textbook Context". You MUST base your answer on this content.
    2.  If the context is insufficient, you may use your general knowledge to enrich the explanation, but do not contradict the textbook.
    3.  Explain concepts in a simple, engaging way. Use short, real-world examples or analogies.
    4.  Do not copy the textbook verbatim. Rephrase everything in your own conversational words.
    5.  If the question cannot be answered from the context, reply with only this exact sentence: "I'm sorry, but this topic does not seem to be covered in your textbook."

    **Formatting and Writing Style:**
    1.  **Direct Answer First:** Start with a direct, 1-2 sentence summary answer to the core question.
    2.  **Structure with Headings:** Organize your answer using Markdown H2 (`##`) for main sections and H3 (`###`) for subsections.
    3.  **Short Paragraphs:** Keep paragraphs short and focused (no more than 3 sentences).
    4.  **Use Lists:** Use bullet points (`*`) or numbered lists for clarity.
    5.  **Use Tables:** For comparisons or structured data, use a well-formatted Markdown table. For example: `| Feature | Detail A | Detail B |\n|---|---|---|\n| Point 1 | Yes | No |`
    6.  **Limit Bolding:** Use bolding (`**word**`) only for short, important highlights (no more than 3 consecutive words).

    **Textbook Context:**
    ---
    {context}
    ---

    **User's Question:**
    {query}

    Now, as CHADUVU-GURU, provide a step-by-step explanation following all the rules above.
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
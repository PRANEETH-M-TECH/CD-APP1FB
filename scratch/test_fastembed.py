import sys
import time

try:
    from fastembed import TextEmbedding
    print("fastembed imported successfully!")
except ImportError as e:
    print(f"Failed to import fastembed: {e}")
    sys.exit(1)

start_time = time.time()
print("Loading model 'sentence-transformers/all-MiniLM-L6-v2'...")
try:
    # fastembed downloads and loads the ONNX weights automatically
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Test embedding generation
texts = ["Explain what is a balanced diet and why we should follow it?", "Photosynthesis process"]
print("Generating embeddings...")
try:
    embeddings_generator = model.embed(texts)
    embeddings = list(embeddings_generator)
    print(f"Generated {len(embeddings)} embeddings.")
    for i, emb in enumerate(embeddings):
        print(f" - Text {i+1} embedding dimension: {len(emb)}")
        print(f"   First 5 values: {list(emb[:5])}")
except Exception as e:
    print(f"Error generating embeddings: {e}")
    sys.exit(1)

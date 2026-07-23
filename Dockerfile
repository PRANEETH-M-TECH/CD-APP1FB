FROM python:3.11-slim

# Force unbuffered Python stdout/stderr for real-time log streaming on Render
ENV PYTHONUNBUFFERED=1

# Install system dependencies & Node.js for Hyperframes compilation
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch (160MB instead of 2.5GB GPU bloat)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirement manifests and install Python & Node dependencies
COPY requirements.txt package.json ./
RUN pip install --no-cache-dir -r requirements.txt
RUN npm install

# Pre-download SentenceTransformer model weights during build so server boots instantly (0s startup delay)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code
COPY . .

# Expose Render standard port 10000
EXPOSE 10000

# Launch FastAPI using dynamic $PORT (defaulting to 10000 for Render)
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

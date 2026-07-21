FROM python:3.11-slim

# Install system dependencies & Node.js for Hyperframes compilation
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Copy requirement manifests and install Python & Node dependencies
COPY requirements.txt package.json ./
RUN pip install --no-cache-dir -r requirements.txt
RUN npm install

# Copy application source code
COPY . .

# Expose default port
EXPOSE 7860

# Launch FastAPI using shell expansion for dynamic $PORT binding
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

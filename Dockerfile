# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for document parsers & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose default port (FastAPI usually runs on 8000, Render injects PORT env variable)
ENV PORT=8000
EXPOSE 8000

# Start command using uvicorn binding to PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

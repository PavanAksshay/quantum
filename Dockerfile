FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Hugging Face Spaces runs containers on port 7860
EXPOSE 7860

# Start FastAPI backend
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]

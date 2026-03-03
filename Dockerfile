FROM python:3.10-slim

# HuggingFace Spaces runs on port 7860
ENV PORT=7860
WORKDIR /app

# System deps for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/        ./api/
COPY pipeline/   ./pipeline/
COPY data/       ./data/

# Expose HF Spaces port
EXPOSE 7860

# Start FastAPI on port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

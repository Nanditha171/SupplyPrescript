FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including coinor-cbc solver and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt httpx

# Copy project source code
COPY . .

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Default command: start FastAPI backend
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


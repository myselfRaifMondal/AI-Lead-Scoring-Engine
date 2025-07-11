FROM python:3.9-slim

# Set pip environment variables to avoid timeouts
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_RETRIES=5
ENV PIP_TIMEOUT=1000

# Set working directory
WORKDIR /app

# Install system dependencies with retry logic and hash mismatch mitigation
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    (apt-get install -y --fix-missing \
        gcc \
        g++ \
        libpq-dev \
        curl \
    || apt-get install -y --fix-missing --allow-unauthenticated \
        gcc \
        g++ \
        libpq-dev \
        curl) && \
    apt-get --fix-broken install -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Install additional ML dependencies
RUN pip install --no-cache-dir --timeout=1000 --retries=5 \
    xgboost>=1.6.0 \
    lightgbm>=3.3.0 \
    torch>=1.12.0 \
    transformers>=4.20.0 \
    sentence-transformers>=2.2.0

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY monitoring/ ./monitoring/

# Create models directory
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "api/main.py"]

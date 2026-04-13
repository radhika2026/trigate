FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec first (layer caching)
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir -e "."

# Copy application code
COPY gateway/ ./gateway/

# Expose port
EXPOSE 8000

# Start gateway
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

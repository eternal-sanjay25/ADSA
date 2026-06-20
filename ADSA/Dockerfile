FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p outputs/charts outputs/reports outputs/models outputs/uploads sample_data

# Expose port
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]

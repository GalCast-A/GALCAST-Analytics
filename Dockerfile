# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies for scikit-learn and cvxpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Upgrade pip and install dependencies with verbose output for debugging
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -v || { echo "pip install failed"; exit 1; }

# Copy application code
COPY . .

# Expose port (informational, Cloud Run uses PORT env var)
EXPOSE 8080

# Define environment variables
ENV PYTHONUNBUFFERED=1
# Cloud Run sets PORT, default to 8080 if not set
ENV PORT=8080

# Command to run the application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "--bind", "0.0.0.0:${PORT}", "main:app"]

# ============================================================================
# Production Dockerfile - Reviewer Recommendation Backend
# ============================================================================
# Base: python:3.11-slim
# Features: CPU-only inference, minimal image size, production-ready
# Deployment: Render, Railway, Fly.io, etc.
# ============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="Applied AI Assignment"
LABEL description="Reviewer Recommendation System - Backend API"
LABEL version="1.0"

# ============================================================================
# System Dependencies
# ============================================================================

# Install system packages needed for Python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools for compiling Python packages
    build-essential \
    gcc \
    g++ \
    # OpenMP support for LightGBM (libgomp1)
    libgomp1 \
    # PDF processing with pdfplumber/poppler
    poppler-utils \
    # Optional OCR support (tesseract-ocr)
    # tesseract-ocr \
    # Clean up apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Working Directory
# ============================================================================

WORKDIR /app

# ============================================================================
# Python Dependencies
# ============================================================================

# Copy requirements first for Docker layer caching
# (Only rebuild if requirements.txt changes)
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Application Code
# ============================================================================

# Copy only needed files (exclude unnecessary files)
# This keeps the image size small
COPY app.py .
COPY db_utils.py .
COPY coauthor_graph.py .
COPY ranker.py .
COPY parser.py .
COPY embedding.py .
COPY tfidf_engine.py .
COPY topic_model.py .
COPY utils.py .

# Copy data and models (prebuilt embeddings)
# NOTE: Embeddings must be built before Docker image creation
COPY data/ ./data/
COPY models/ ./models/

# ============================================================================
# Runtime Configuration
# ============================================================================

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default environment variables (can be overridden at runtime)
ENV BACKEND_DB=data/papers.db
ENV FRONTEND_ORIGIN=http://localhost:5173
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')" || exit 1

# ============================================================================
# User and Permissions
# ============================================================================

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# ============================================================================
# Startup Command
# ============================================================================

# Expose port (configurable via $PORT)
EXPOSE ${PORT}

# Start uvicorn with dynamic port binding
# Use 0.0.0.0 to allow external connections
# --host 0.0.0.0: Listen on all network interfaces
# --port $PORT: Use PORT environment variable (defaults to 8000)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

# ============================================================================
# Build and Run Instructions
# ============================================================================
# 
# Build:
#   docker build -t reviewer-recommender-backend .
#
# Run locally:
#   docker run -p 8000:8000 reviewer-recommender-backend
#
# Run with custom port:
#   docker run -p 5000:5000 -e PORT=5000 reviewer-recommender-backend
#
# Run with custom database:
#   docker run -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     -e BACKEND_DB=/app/data/papers.db \
#     reviewer-recommender-backend
#
# Deploy to Render:
#   1. Push to GitHub
#   2. Connect to Render
#   3. Set environment variables:
#      - BACKEND_DB=data/papers.db
#      - FRONTEND_ORIGIN=https://your-frontend.onrender.com
#      - PORT=10000 (Render assigns this automatically)
#   4. Deploy
#
# ============================================================================
# Image Size Optimization
# ============================================================================
# - Uses python:3.11-slim (not full python:3.11)
# - Removes apt cache after installing system packages
# - Uses --no-cache-dir with pip to avoid storing downloaded packages
# - Only copies necessary files (not entire repo)
# - Runs as non-root user (appuser)
#
# Expected image size: ~2-3 GB
# (Most size comes from torch, transformers, sentence-transformers)
#
# To reduce further:
# - Use multi-stage build to separate build-time and runtime dependencies
# - Pre-download model weights and include in image
# - Consider using distroless base image
#
# ============================================================================
# CPU-Only Inference Notes
# ============================================================================
# - FAISS uses CPU IndexFlatIP (no GPU needed)
# - LightGBM uses CPU for prediction (very fast)
# - TF-IDF is CPU-based (scikit-learn)
# - torch/transformers will use CPU by default (no CUDA in container)
# - Embeddings are prebuilt (data/faiss_index.faiss) so no GPU needed at runtime
#
# Performance:
# - FAISS search: ~10-50ms for 1000s of vectors
# - TF-IDF search: ~5-20ms
# - LightGBM predict: ~1-5ms per candidate
# - Total latency: ~100-200ms per request
#
# ============================================================================

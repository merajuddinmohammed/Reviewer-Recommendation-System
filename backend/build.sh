#!/usr/bin/env bash
# Render Build Script for Backend

set -e  # Exit on error

echo "=========================================="
echo "Starting Render Build Process"
echo "=========================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if models exist
echo "Checking for pre-built models..."

if [ -f "models/tfidf_vectorizer.pkl" ] && [ -f "data/faiss_index.faiss" ]; then
    echo "✓ Models found!"
    echo "  - TF-IDF model: models/tfidf_vectorizer.pkl"
    echo "  - FAISS index: data/faiss_index.faiss"
    echo "  - ID map: data/id_map.npy"
else
    echo "⚠ Models not found in repository"
    echo "  Models must be built locally and committed to Git"
    echo "  Or uploaded to cloud storage (S3/GCS) and downloaded here"
fi

echo "=========================================="
echo "Build Complete!"
echo "=========================================="

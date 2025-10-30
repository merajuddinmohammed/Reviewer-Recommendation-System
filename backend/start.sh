#!/usr/bin/env bash
# Render Start Script - Optimized for low memory

# Set environment variables to reduce memory usage
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Start the server
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1

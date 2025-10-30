# Render Deployment Configuration
# This file is used by Render to optimize memory usage

# Memory optimization flags
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Reduce memory for tokenizers
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

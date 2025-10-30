"""
Configuration Module - Centralized Settings for Reviewer Recommendation System

This module provides centralized configuration management for all tunable parameters
in the reviewer recommendation system. All constants are defined here with sensible
defaults and can be overridden via environment variables for deployment flexibility.

Configuration Categories:
1. Ranking Parameters: Control ranking behavior (topk, neighbors, weights)
2. Performance Parameters: Control speed/memory tradeoffs (batch sizes, neighbors)
3. Feature Engineering: Control feature computation (recency, weighting)

Environment Variable Support:
- All parameters can be overridden via environment variables
- Numeric values are parsed with type checking
- Invalid values fall back to defaults with warnings

Author: Applied AI Assignment
Date: December 2024
"""

import os
import logging
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions for Environment Variable Parsing
# ============================================================================

def get_env_int(key: str, default: int) -> int:
    """
    Get integer from environment variable with fallback to default.
    
    Args:
        key: Environment variable name
        default: Default value if env var not set or invalid
        
    Returns:
        Integer value from env var or default
    """
    value = os.environ.get(key)
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid integer for {key}='{value}', using default: {default}")
        return default


def get_env_float(key: str, default: float) -> float:
    """
    Get float from environment variable with fallback to default.
    
    Args:
        key: Environment variable name
        default: Default value if env var not set or invalid
        
    Returns:
        Float value from env var or default
    """
    value = os.environ.get(key)
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid float for {key}='{value}', using default: {default}")
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """
    Get boolean from environment variable with fallback to default.
    
    Args:
        key: Environment variable name
        default: Default value if env var not set or invalid
        
    Returns:
        Boolean value from env var or default
    """
    value = os.environ.get(key)
    if value is None:
        return default
    
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif value_lower in ('false', '0', 'no', 'off'):
            return False
    
    logger.warning(f"Invalid boolean for {key}='{value}', using default: {default}")
    return default


# ============================================================================
# RANKING PARAMETERS
# ============================================================================

# TOPK_RETURN: Number of top reviewers to return in final ranking
# - Higher values give more options but slower response
# - Lower values are faster but may miss good candidates
# - Typical range: 5-50
# - Default: 10 (good balance for most use cases)
TOPK_RETURN = get_env_int('TOPK_RETURN', default=10)

# N1_FAISS: Number of candidates retrieved from FAISS (embedding similarity)
# - First-stage retrieval using semantic similarity
# - Higher values: More recall, slower, more memory
# - Lower values: Faster, less memory, may miss relevant candidates
# - Must be >= TOPK_RETURN
# - Typical range: 50-500
# - Default: 200 (good recall/speed tradeoff)
N1_FAISS = get_env_int('N1_FAISS', default=200)

# N2_TFIDF: Number of candidates retrieved from TF-IDF (lexical similarity)
# - First-stage retrieval using keyword matching
# - Higher values: Better lexical coverage, slower
# - Lower values: Faster, may miss keyword-specific matches
# - Must be >= TOPK_RETURN
# - Typical range: 50-500
# - Default: 200 (matches N1_FAISS for balanced retrieval)
N2_TFIDF = get_env_int('N2_TFIDF', default=200)


# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# RECENCY_TAU: Time decay constant (years) for recency score
# - Controls how quickly older papers lose relevance
# - Formula: recency_score = exp(-age_in_years / TAU)
# - Higher tau: Slower decay, older papers still valuable
# - Lower tau: Faster decay, prefer recent work
# - tau=3.0 means paper loses ~63% value after 3 years
# - Typical range: 1.0-5.0 years
# - Default: 3.0 years (reasonable for academic research)
RECENCY_TAU = get_env_float('RECENCY_TAU', default=3.0)

# W_S: Weight for semantic similarity (embedding-based)
# - Contribution of SciBERT embedding similarity to hybrid score
# - Higher weight: Emphasize semantic/conceptual similarity
# - Range: 0.0-1.0
# - Default: 0.55 (primary signal)
W_S = get_env_float('W_S', default=0.55)

# W_L: Weight for lexical similarity (TF-IDF-based)
# - Contribution of keyword/term matching to hybrid score
# - Higher weight: Emphasize exact terminology matches
# - Range: 0.0-1.0
# - Default: 0.25 (secondary signal)
W_L = get_env_float('W_L', default=0.25)

# W_R: Weight for recency score
# - Contribution of publication recency to hybrid score
# - Higher weight: Strongly prefer recent publications
# - Range: 0.0-1.0
# - Default: 0.20 (tertiary signal)
W_R = get_env_float('W_R', default=0.20)

# Note: Weights don't need to sum to 1.0, but typical sum is ~1.0
# Current sum: 0.55 + 0.25 + 0.20 = 1.00


# ============================================================================
# PERFORMANCE PARAMETERS
# ============================================================================

# EMB_BATCH: Batch size for embedding computation
# - Number of texts processed at once during embedding generation
# - Higher values: Faster but more GPU/CPU memory
# - Lower values: Slower but less memory usage
# - Typical range: 1-32 (CPU), 4-128 (GPU)
# - Default: 4 (safe for CPU-only systems)
EMB_BATCH = get_env_int('EMB_BATCH', default=4)

# DB_BATCH: Batch size for database queries
# - Number of records fetched per query
# - Higher values: Fewer round trips, more memory
# - Lower values: More round trips, less memory
# - Typical range: 100-10000
# - Default: 1000 (good balance)
DB_BATCH = get_env_int('DB_BATCH', default=1000)

# FAISS_NPROBE: Number of clusters to probe in FAISS IVF index
# - Trade-off between speed and recall
# - Higher values: Better recall, slower search
# - Lower values: Faster search, may miss neighbors
# - Only used if FAISS index is IVF type
# - Typical range: 1-100
# - Default: 10 (reasonable for most cases)
FAISS_NPROBE = get_env_int('FAISS_NPROBE', default=10)


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# EMBEDDING_MODEL: SciBERT model identifier
# - HuggingFace model name for generating embeddings
# - Default: allenai/scibert_scivocab_uncased (scientific domain)
# - Alternatives: sentence-transformers/all-mpnet-base-v2 (general domain)
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'allenai/scibert_scivocab_uncased')

# EMBEDDING_DIM: Dimension of embedding vectors
# - Must match the model output dimension
# - SciBERT: 768
# - MPNet: 768
# - Default: 768 (standard BERT dimension)
EMBEDDING_DIM = get_env_int('EMBEDDING_DIM', default=768)

# MAX_SEQ_LENGTH: Maximum sequence length for transformer models
# - Longer sequences: Better context, slower, more memory
# - Shorter sequences: Faster, less memory, may truncate
# - SciBERT supports up to 512 tokens
# - Default: 512 (full context)
MAX_SEQ_LENGTH = get_env_int('MAX_SEQ_LENGTH', default=512)


# ============================================================================
# DATABASE PARAMETERS
# ============================================================================

# DB_PATH: Path to SQLite database
# - Relative to backend/ directory
# - Default: data/papers.db
DB_PATH = os.environ.get('DB_PATH', 'data/papers.db')

# MODELS_DIR: Directory containing trained models
# - TF-IDF vectorizer, FAISS index, LightGBM ranker
# - Default: models/
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')


# ============================================================================
# API PARAMETERS
# ============================================================================

# API_HOST: Host address for FastAPI server
# - Default: 0.0.0.0 (all interfaces)
API_HOST = os.environ.get('API_HOST', '0.0.0.0')

# API_PORT: Port for FastAPI server
# - Default: 8000
API_PORT = get_env_int('API_PORT', default=8000)

# CORS_ORIGINS: Allowed CORS origins (comma-separated)
# - Default: * (allow all, change for production)
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

# MAX_QUERY_LENGTH: Maximum length for query text
# - Prevent excessively long queries
# - Default: 10000 characters
MAX_QUERY_LENGTH = get_env_int('MAX_QUERY_LENGTH', default=10000)


# ============================================================================
# CONFLICT-OF-INTEREST PARAMETERS
# ============================================================================

# COI_ENABLE: Enable conflict-of-interest filtering
# - Set to False to disable all COI checks
# - Default: True
COI_ENABLE = get_env_bool('COI_ENABLE', default=True)

# COI_SAME_PERSON: Filter candidates with same name as query author
# - Default: True
COI_SAME_PERSON = get_env_bool('COI_SAME_PERSON', default=True)

# COI_COAUTHOR: Filter candidates who are coauthors with query author
# - Default: True
COI_COAUTHOR = get_env_bool('COI_COAUTHOR', default=True)

# COI_SAME_AFFILIATION: Filter candidates with same affiliation as query author
# - Default: True
COI_SAME_AFFILIATION = get_env_bool('COI_SAME_AFFILIATION', default=True)


# ============================================================================
# VALIDATION AND WARNINGS
# ============================================================================

def validate_config():
    """
    Validate configuration parameters and emit warnings for potential issues.
    
    Checks:
    - N1_FAISS and N2_TFIDF >= TOPK_RETURN
    - Weights are in valid range [0, 1]
    - Batch sizes are positive
    - Recency tau is positive
    """
    warnings = []
    
    # Check retrieval sizes
    if N1_FAISS < TOPK_RETURN:
        warnings.append(f"N1_FAISS ({N1_FAISS}) < TOPK_RETURN ({TOPK_RETURN}), "
                       "may not have enough candidates")
    
    if N2_TFIDF < TOPK_RETURN:
        warnings.append(f"N2_TFIDF ({N2_TFIDF}) < TOPK_RETURN ({TOPK_RETURN}), "
                       "may not have enough candidates")
    
    # Check weights
    if not (0.0 <= W_S <= 1.0):
        warnings.append(f"W_S ({W_S}) should be in range [0, 1]")
    
    if not (0.0 <= W_L <= 1.0):
        warnings.append(f"W_L ({W_L}) should be in range [0, 1]")
    
    if not (0.0 <= W_R <= 1.0):
        warnings.append(f"W_R ({W_R}) should be in range [0, 1]")
    
    # Check batch sizes
    if EMB_BATCH <= 0:
        warnings.append(f"EMB_BATCH ({EMB_BATCH}) must be positive")
    
    if DB_BATCH <= 0:
        warnings.append(f"DB_BATCH ({DB_BATCH}) must be positive")
    
    # Check recency tau
    if RECENCY_TAU <= 0:
        warnings.append(f"RECENCY_TAU ({RECENCY_TAU}) must be positive")
    
    # Emit warnings
    if warnings:
        logger.warning("Configuration validation issues:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    return len(warnings) == 0


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config():
    """Print current configuration for debugging."""
    print("=" * 80)
    print("REVIEWER RECOMMENDATION SYSTEM - CONFIGURATION")
    print("=" * 80)
    
    print("\n[RANKING PARAMETERS]")
    print(f"  TOPK_RETURN:    {TOPK_RETURN:>6}  # Top K reviewers to return")
    print(f"  N1_FAISS:       {N1_FAISS:>6}  # FAISS candidates (semantic)")
    print(f"  N2_TFIDF:       {N2_TFIDF:>6}  # TF-IDF candidates (lexical)")
    
    print("\n[FEATURE ENGINEERING]")
    print(f"  RECENCY_TAU:    {RECENCY_TAU:>6.2f}  # Time decay (years)")
    print(f"  W_S (semantic): {W_S:>6.2f}  # Embedding weight")
    print(f"  W_L (lexical):  {W_L:>6.2f}  # TF-IDF weight")
    print(f"  W_R (recency):  {W_R:>6.2f}  # Recency weight")
    print(f"  Weight sum:     {W_S + W_L + W_R:>6.2f}")
    
    print("\n[PERFORMANCE]")
    print(f"  EMB_BATCH:      {EMB_BATCH:>6}  # Embedding batch size")
    print(f"  DB_BATCH:       {DB_BATCH:>6}  # Database batch size")
    print(f"  FAISS_NPROBE:   {FAISS_NPROBE:>6}  # FAISS clusters to probe")
    
    print("\n[MODEL]")
    print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"  EMBEDDING_DIM:   {EMBEDDING_DIM}")
    print(f"  MAX_SEQ_LENGTH:  {MAX_SEQ_LENGTH}")
    
    print("\n[DATABASE]")
    print(f"  DB_PATH:      {DB_PATH}")
    print(f"  MODELS_DIR:   {MODELS_DIR}")
    
    print("\n[API]")
    print(f"  API_HOST:     {API_HOST}")
    print(f"  API_PORT:     {API_PORT}")
    print(f"  CORS_ORIGINS: {CORS_ORIGINS}")
    print(f"  MAX_QUERY_LENGTH: {MAX_QUERY_LENGTH}")
    
    print("\n[CONFLICT-OF-INTEREST]")
    print(f"  COI_ENABLE:           {COI_ENABLE}")
    print(f"  COI_SAME_PERSON:      {COI_SAME_PERSON}")
    print(f"  COI_COAUTHOR:         {COI_COAUTHOR}")
    print(f"  COI_SAME_AFFILIATION: {COI_SAME_AFFILIATION}")
    
    print("\n" + "=" * 80)


# Run validation on import
validate_config()


# ============================================================================
# DOCUMENTATION
# ============================================================================

"""
Configuration Parameter Guide
==============================

RANKING PARAMETERS
------------------

TOPK_RETURN (default: 10)
    Number of top reviewers to return in final ranking.
    
    Tuning Guide:
    - Increase for more reviewer options (slower response)
    - Decrease for faster response (fewer options)
    - Typical values: 5 (fast), 10 (balanced), 20-50 (comprehensive)
    
    Environment: TOPK_RETURN=20


N1_FAISS (default: 200)
    Number of candidates retrieved from FAISS index (semantic similarity).
    
    Tuning Guide:
    - Increase for better recall (slower, more memory)
    - Decrease for speed (may miss relevant candidates)
    - Must be >= TOPK_RETURN
    - Typical values: 50 (fast), 200 (balanced), 500 (thorough)
    
    Impact on Memory: ~8KB per candidate (embedding vectors)
    Impact on Speed: Linear with N1_FAISS
    
    Environment: N1_FAISS=500


N2_TFIDF (default: 200)
    Number of candidates retrieved from TF-IDF index (lexical similarity).
    
    Tuning Guide:
    - Increase for better keyword coverage (slower)
    - Decrease for speed (may miss keyword matches)
    - Must be >= TOPK_RETURN
    - Typical values: 50 (fast), 200 (balanced), 500 (thorough)
    
    Impact on Speed: Linear with N2_TFIDF
    
    Environment: N2_TFIDF=500


FEATURE ENGINEERING PARAMETERS
-------------------------------

RECENCY_TAU (default: 3.0 years)
    Time decay constant for recency score.
    Formula: recency = exp(-age_in_years / TAU)
    
    Tuning Guide:
    - Increase to value older papers more (slower decay)
    - Decrease to strongly prefer recent work (faster decay)
    - tau=1.0: 37% value after 1 year, 13% after 2 years
    - tau=3.0: 72% value after 1 year, 51% after 2 years
    - tau=5.0: 82% value after 1 year, 67% after 2 years
    
    Typical values: 1.0-5.0 years
    
    Environment: RECENCY_TAU=5.0


W_S (default: 0.55)
    Weight for semantic similarity (embedding-based).
    
    Tuning Guide:
    - Increase to emphasize conceptual/semantic similarity
    - Decrease if embeddings are less reliable
    - Must be in range [0, 1]
    
    Environment: W_S=0.6


W_L (default: 0.25)
    Weight for lexical similarity (TF-IDF-based).
    
    Tuning Guide:
    - Increase to emphasize exact keyword matches
    - Decrease if TF-IDF is less reliable
    - Must be in range [0, 1]
    
    Environment: W_L=0.3


W_R (default: 0.20)
    Weight for recency score.
    
    Tuning Guide:
    - Increase to strongly prefer recent publications
    - Decrease to value all time periods equally
    - Must be in range [0, 1]
    
    Environment: W_R=0.1


PERFORMANCE PARAMETERS
----------------------

EMB_BATCH (default: 4)
    Batch size for embedding computation.
    
    Tuning Guide:
    - Increase for faster processing (more memory)
    - Decrease for memory-constrained systems
    - CPU systems: 1-8
    - GPU systems: 8-128
    
    Impact on Memory: ~200MB per batch item (with SciBERT)
    
    Environment: EMB_BATCH=8


DB_BATCH (default: 1000)
    Batch size for database queries.
    
    Tuning Guide:
    - Increase for fewer database round trips
    - Decrease for memory-constrained systems
    - Typical values: 100-10000
    
    Environment: DB_BATCH=5000


FAISS_NPROBE (default: 10)
    Number of clusters to probe in FAISS IVF index.
    
    Tuning Guide:
    - Increase for better recall (slower search)
    - Decrease for faster search (may miss neighbors)
    - Only used if FAISS index is IVF type
    - Typical values: 1 (fast), 10 (balanced), 100 (thorough)
    
    Environment: FAISS_NPROBE=20


USAGE EXAMPLES
--------------

Example 1: Fast Mode (minimal resources)
    export TOPK_RETURN=5
    export N1_FAISS=50
    export N2_TFIDF=50
    export EMB_BATCH=1
    
    Result: ~5x faster, less comprehensive


Example 2: Thorough Mode (high recall)
    export TOPK_RETURN=50
    export N1_FAISS=500
    export N2_TFIDF=500
    export EMB_BATCH=16
    
    Result: ~3x slower, better recall


Example 3: Recent Work Emphasis
    export RECENCY_TAU=1.5
    export W_R=0.4
    export W_S=0.4
    export W_L=0.2
    
    Result: Strongly prefer recent publications


Example 4: Keyword Focus
    export W_L=0.5
    export W_S=0.3
    export W_R=0.2
    
    Result: Emphasize exact terminology matches


MONITORING
----------

To verify configuration:
    python -c "import config; config.print_config()"

To validate configuration:
    python -c "import config; config.validate_config()"

To check environment variables:
    env | grep -E "(TOPK|N1|N2|RECENCY|W_S|W_L|W_R|EMB_BATCH)"
"""

if __name__ == "__main__":
    print_config()
    print("\nValidation result:", validate_config())

"""
FastAPI Inference API for Reviewer Recommendation

This API provides endpoints for recommending reviewers for academic papers.
Supports both file uploads (PDF) and JSON input with title/abstract.

Features:
- Model loading at startup (TF-IDF, FAISS, BERTopic, LightGBM)
- Dual input: File upload or JSON
- Complete ranking pipeline with COI detection
- Evidence-based recommendations with top matching papers
- CORS support for frontend integration
- Graceful handling of missing models

Endpoints:
- POST /recommend: Get reviewer recommendations
- GET /health: Health check endpoint

Author: Applied AI Assignment
Date: December 2024
"""

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import faiss

# Import local modules
from db_utils import get_connection
from coauthor_graph import has_conflict
from ranker import make_features_for_query
import config  # Centralized configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration (imported from config.py)
# ============================================================================

# Database and API settings
BACKEND_DB = config.DB_PATH
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# Model paths
TFIDF_MODEL_PATH = Path("models/tfidf_vectorizer.pkl")
FAISS_INDEX_PATH = Path("data/faiss_index.faiss")
FAISS_ID_MAP_PATH = Path("data/id_map.npy")
BERTOPIC_MODEL_PATH = Path("models/bertopic_model.pkl")
LGBM_MODEL_PATH = Path("models/lgbm_ranker.pkl")


# ============================================================================
# Global Model Storage
# ============================================================================

class ModelStore:
    """Global storage for loaded models."""
    
    def __init__(self):
        self.db_path: Optional[Path] = None
        self.tfidf_vectorizer = None
        self.faiss_index = None
        self.id_map = None
        self.bertopic_model = None
        self.lgbm_model = None
        self.embeddings = None  # For encoding queries
        
    def is_ready(self) -> bool:
        """Check if essential models are loaded."""
        return (
            self.db_path is not None and
            self.tfidf_vectorizer is not None and
            self.faiss_index is not None and
            self.id_map is not None
        )


models = ModelStore()


# ============================================================================
# Pydantic Models
# ============================================================================

class RecommendRequest(BaseModel):
    """Request model for JSON-based recommendations."""
    
    title: str = Field(..., description="Paper title", min_length=1, max_length=config.MAX_QUERY_LENGTH)
    abstract: str = Field(..., description="Paper abstract", min_length=1, max_length=config.MAX_QUERY_LENGTH)
    authors: Optional[List[str]] = Field(default=None, description="List of author names")
    affiliations: Optional[List[str]] = Field(default=None, description="List of author affiliations")
    k: int = Field(default=config.TOPK_RETURN, description="Number of recommendations to return", ge=1, le=100)
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Deep Learning for Computer Vision",
                "abstract": "This paper presents a novel approach to image classification using deep neural networks...",
                "authors": ["John Smith", "Jane Doe"],
                "affiliations": ["MIT", "Stanford"],
                "k": 10
            }
        }


class PaperEvidence(BaseModel):
    """Evidence for a recommendation (matching paper)."""
    
    paper_title: str
    similarity: float = Field(..., description="Similarity score (0-1)")
    year: Optional[int] = None


class RecommendationResult(BaseModel):
    """Single recommendation result."""
    
    author_id: int
    name: str
    affiliation: Optional[str] = None
    score: float = Field(..., description="Overall recommendation score")
    evidence: List[PaperEvidence] = Field(default_factory=list, description="Top matching papers")


class RecommendResponse(BaseModel):
    """Response model for recommendations."""
    
    recommendations: List[RecommendationResult]
    total_candidates: int
    filtered_by_coi: int
    model_used: str = Field(..., description="Ranking model used (lgbm or weighted)")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    models_loaded: Dict[str, bool]
    database_path: Optional[str] = None


# ============================================================================
# Startup and Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    
    # Startup: Load models
    logger.info("=" * 80)
    logger.info("Starting FastAPI Reviewer Recommendation API")
    logger.info("=" * 80)
    
    # Load database path
    db_path = Path(BACKEND_DB)
    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}")
        logger.warning("API will start but recommendations will fail")
    else:
        models.db_path = db_path
        logger.info(f"✓ Database: {db_path}")
    
    # Load TF-IDF vectorizer (REQUIRED)
    if TFIDF_MODEL_PATH.exists():
        try:
            from tfidf_engine import TFIDFEngine
            models.tfidf_vectorizer = TFIDFEngine.load(str(TFIDF_MODEL_PATH))
            logger.info(f"✓ TF-IDF vectorizer loaded from {TFIDF_MODEL_PATH}")
        except Exception as e:
            logger.error(f"✗ Failed to load TF-IDF vectorizer: {e}")
    else:
        logger.warning(f"✗ TF-IDF vectorizer not found at {TFIDF_MODEL_PATH}")
    
    # Load FAISS index (REQUIRED)
    if FAISS_INDEX_PATH.exists():
        try:
            models.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info(f"✓ FAISS index loaded from {FAISS_INDEX_PATH}")
            logger.info(f"  Index size: {models.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"✗ Failed to load FAISS index: {e}")
    else:
        logger.warning(f"✗ FAISS index not found at {FAISS_INDEX_PATH}")
    
    # Load FAISS ID map (REQUIRED)
    if FAISS_ID_MAP_PATH.exists():
        try:
            # Load from .npy file
            models.id_map = np.load(str(FAISS_ID_MAP_PATH), allow_pickle=True)
            logger.info(f"✓ FAISS ID map loaded from {FAISS_ID_MAP_PATH}")
            logger.info(f"  Mapped papers: {len(models.id_map)}")
        except Exception as e:
            logger.error(f"✗ Failed to load FAISS ID map: {e}")
    else:
        logger.warning(f"✗ FAISS ID map not found at {FAISS_ID_MAP_PATH}")
    
    # Don't load Embeddings model at startup to save memory (lazy load on first use)
    # This reduces initial memory footprint on free tier
    models.embeddings = None
    logger.info("⚠ Embeddings model will be lazy-loaded on first request (memory optimization)")
    
    # Load BERTopic model (OPTIONAL)
    if BERTOPIC_MODEL_PATH.exists():
        try:
            with open(BERTOPIC_MODEL_PATH, 'rb') as f:
                models.bertopic_model = pickle.load(f)
            logger.info(f"✓ BERTopic model loaded from {BERTOPIC_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"⚠ Failed to load BERTopic model: {e}")
    else:
        logger.info(f"⚠ BERTopic model not found at {BERTOPIC_MODEL_PATH} (optional)")
    
    # Load LightGBM model (OPTIONAL)
    if LGBM_MODEL_PATH.exists():
        try:
            with open(LGBM_MODEL_PATH, 'rb') as f:
                models.lgbm_model = pickle.load(f)
            logger.info(f"✓ LightGBM model loaded from {LGBM_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"⚠ Failed to load LightGBM model: {e}")
    else:
        logger.info(f"⚠ LightGBM model not found at {LGBM_MODEL_PATH} (optional)")
    
    # Summary
    logger.info("=" * 80)
    if models.is_ready():
        logger.info("✓ API ready for recommendations")
    else:
        logger.warning("⚠ Some essential models missing - recommendations may fail")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Reviewer Recommendation API",
    description="API for recommending academic paper reviewers based on content similarity and expertise",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
cors_origins = config.CORS_ORIGINS if config.CORS_ORIGINS != ['*'] else [FRONTEND_ORIGIN, "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def extract_text_from_pdf(pdf_path: Path) -> Dict[str, str]:
    """
    Extract text from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with 'title' and 'abstract' keys
        
    Raises:
        Exception if extraction fails
    """
    try:
        from parser import extract_text_with_fallback
        
        # Extract text
        text, metadata = extract_text_with_fallback(pdf_path)
        
        if not text:
            raise ValueError("No text extracted from PDF")
        
        # Simple heuristic: First line is title, rest is abstract
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            raise ValueError("No text extracted from PDF")
        
        title = lines[0] if lines else "Untitled"
        abstract = ' '.join(lines[1:]) if len(lines) > 1 else text
        
        # Truncate if too long
        if len(abstract) > 5000:
            abstract = abstract[:5000]
        
        return {
            "title": title,
            "abstract": abstract
        }
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise


def compute_weighted_score(features: pd.Series) -> float:
    """
    Compute weighted score from features (fallback when LGBM not available).
    
    Uses weights from config.py:
    - W_S: Embedding similarity (emb_max)
    - W_L: TF-IDF similarity (tfidf_max)
    - W_R: Recency (recency_max)
    
    Args:
        features: Feature series
        
    Returns:
        Weighted score (0-1)
    """
    score = 0.0
    
    # Embedding similarity (default: 55%)
    if 'emb_max' in features:
        score += config.W_S * float(features['emb_max'])
    
    # TF-IDF similarity (default: 25%)
    if 'tfidf_max' in features:
        score += config.W_L * float(features['tfidf_max'])
    
    # Recency (default: 20%)
    if 'recency_max' in features:
        score += config.W_R * float(features['recency_max'])
    
    return score


def get_evidence_papers(
    db_path: Path,
    author_id: int,
    query_title: str,
    query_abstract: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Get evidence papers for a candidate author.
    
    Finds the top matching papers by this author based on
    TF-IDF similarity to the query.
    
    Args:
        db_path: Path to database
        author_id: Author ID
        query_title: Query paper title
        query_abstract: Query paper abstract
        top_k: Number of evidence papers to return
        
    Returns:
        List of dictionaries with paper_title, similarity, year
    """
    evidence = []
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get author's papers
            cursor.execute("""
                SELECT p.id, p.title, p.abstract, p.year
                FROM papers p
                WHERE p.author_id = ?
                AND p.title IS NOT NULL
                AND p.abstract IS NOT NULL
                ORDER BY p.year DESC
                LIMIT 20
            """, (author_id,))
            
            papers = cursor.fetchall()
            
            if not papers or not models.tfidf_vectorizer:
                return []
            
            # Compute TF-IDF similarity for each paper
            query_text = f"{query_title} {query_abstract}"
            query_vec = models.tfidf_vectorizer.transform([query_text])
            
            paper_scores = []
            for paper_id, title, abstract, year in papers:
                paper_text = f"{title} {abstract}"
                paper_vec = models.tfidf_vectorizer.transform([paper_text])
                
                # Cosine similarity
                similarity = float(query_vec.multiply(paper_vec.T).toarray()[0, 0])
                
                paper_scores.append({
                    "paper_title": title,
                    "similarity": similarity,
                    "year": year
                })
            
            # Sort by similarity and take top-k
            paper_scores.sort(key=lambda x: x['similarity'], reverse=True)
            evidence = paper_scores[:top_k]
            
    except Exception as e:
        logger.error(f"Error getting evidence papers for author {author_id}: {e}")
    
    return evidence


def rank_candidates(
    features_df: pd.DataFrame,
    query_authors: List[str],
    query_affiliations: List[str],
    k: int
) -> List[Dict[str, Any]]:
    """
    Rank candidates using LGBM or weighted scoring.
    
    Args:
        features_df: DataFrame with candidate features
        query_authors: List of query author names (for COI detection)
        query_affiliations: List of query affiliations (for COI detection)
        k: Number of recommendations to return
        
    Returns:
        List of ranked candidates with scores
    """
    if features_df.empty:
        return []
    
    # Determine which features to use for scoring
    feature_cols = [
        'tfidf_max', 'tfidf_mean', 'emb_max', 'emb_mean',
        'topic_overlap', 'recency_mean', 'recency_max', 'pub_count', 'coi_flag'
    ]
    
    # Check which features are available
    available_features = [col for col in feature_cols if col in features_df.columns]
    
    if not available_features:
        logger.error("No feature columns found in features_df")
        return []
    
    # Compute scores
    if models.lgbm_model is not None:
        # Use LightGBM model
        try:
            X = features_df[available_features].fillna(0)
            scores = models.lgbm_model.predict(X)
            model_used = "lgbm"
            logger.info(f"Using LightGBM model for ranking")
        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}, falling back to weighted scoring")
            scores = features_df.apply(compute_weighted_score, axis=1).values
            model_used = "weighted"
    else:
        # Use weighted scoring
        scores = features_df.apply(compute_weighted_score, axis=1).values
        model_used = "weighted"
        logger.info(f"Using weighted scoring for ranking")
    
    # Demote candidates with COI
    coi_penalty = 0.5  # Multiply score by 0.5 if COI detected
    if 'coi_flag' in features_df.columns:
        scores = scores * np.where(features_df['coi_flag'] == 1, coi_penalty, 1.0)
    
    # Add scores to dataframe
    features_df = features_df.copy()
    features_df['score'] = scores
    
    # Sort by score descending
    features_df = features_df.sort_values('score', ascending=False)
    
    # Take top-k
    top_k = features_df.head(k)
    
    # Convert to list of dicts
    results = []
    for idx, row in top_k.iterrows():
        results.append({
            'author_id': int(row['author_id']),
            'name': row.get('name', 'Unknown'),
            'affiliation': row.get('affiliation'),
            'score': float(row['score'])
        })
    
    return results, model_used


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Reviewer Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    
    models_status = {
        "database": models.db_path is not None and models.db_path.exists(),
        "tfidf_vectorizer": models.tfidf_vectorizer is not None,
        "faiss_index": models.faiss_index is not None,
        "id_map": models.id_map is not None,
        "bertopic_model": models.bertopic_model is not None,
        "lgbm_model": models.lgbm_model is not None
    }
    
    status = "ready" if models.is_ready() else "degraded"
    
    return HealthResponse(
        status=status,
        models_loaded=models_status,
        database_path=str(models.db_path) if models.db_path else None
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(
    # File upload (optional)
    file: Optional[UploadFile] = File(None),
    # JSON fields (optional, can use form data too)
    title: Optional[str] = Form(None),
    abstract: Optional[str] = Form(None),
    authors: Optional[str] = Form(None),  # JSON string
    affiliations: Optional[str] = Form(None),  # JSON string
    k: int = Form(config.TOPK_RETURN)
):
    """
    Get reviewer recommendations for a paper.
    
    Supports two input modes:
    1. File upload: Upload PDF file
    2. JSON/Form: Provide title, abstract, authors, affiliations
    
    At least one mode must be used (file OR title+abstract).
    
    Args:
        file: PDF file (optional)
        title: Paper title (optional if file provided)
        abstract: Paper abstract (optional if file provided)
        authors: Comma-separated author names (optional)
        affiliations: Comma-separated affiliations (optional)
        k: Number of recommendations (default: 10)
        
    Returns:
        RecommendResponse with ranked recommendations
        
    Raises:
        HTTPException if models not loaded or invalid input
    """
    # Check if models are ready
    if not models.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready - essential models not loaded"
        )
    
    # Parse input
    query_title = None
    query_abstract = None
    query_authors = []
    query_affiliations = []
    
    # Mode 1: File upload
    if file is not None:
        logger.info(f"Processing file upload: {file.filename}")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = Path(tmp_file.name)
            content = await file.read()
            tmp_file.write(content)
        
        try:
            # Extract text from PDF
            extracted = extract_text_from_pdf(tmp_path)
            query_title = extracted['title']
            query_abstract = extracted['abstract']
            logger.info(f"Extracted text from PDF: title='{query_title[:50]}...'")
        finally:
            # Clean up temp file
            tmp_path.unlink()
    
    # Mode 2: JSON/Form input
    if title is not None:
        query_title = title
    if abstract is not None:
        query_abstract = abstract
    
    # Validate we have required inputs
    if not query_title or not query_abstract:
        raise HTTPException(
            status_code=400,
            detail="Must provide either file OR (title + abstract)"
        )
    
    # Parse authors (comma-separated or JSON string)
    if authors:
        try:
            import json
            query_authors = json.loads(authors)
        except:
            query_authors = [a.strip() for a in authors.split(',') if a.strip()]
    
    # Parse affiliations (comma-separated or JSON string)
    if affiliations:
        try:
            import json
            query_affiliations = json.loads(affiliations)
        except:
            query_affiliations = [a.strip() for a in affiliations.split(',') if a.strip()]
    
    logger.info(f"Query: title='{query_title[:50]}...', authors={query_authors}, affiliations={query_affiliations}")
    
    # Generate features using make_features_for_query
    try:
        # Lazy load embeddings model on first use (memory optimization for free tier)
        if models.embeddings is None:
            logger.info("Loading embeddings model on first request...")
            from embedding import Embeddings
            models.embeddings = Embeddings()
            logger.info("✓ Embeddings model loaded")
        
        # Combine title and abstract for query text
        query_text = f"{query_title}. {query_abstract}" if query_abstract else query_title
        
        features_df = make_features_for_query(
            query_text=query_text,
            db=str(models.db_path),
            tfidf_engine=models.tfidf_vectorizer,
            faiss_index=models.faiss_index,
            id_map=models.id_map,
            embedding_model=models.embeddings,
            topic_model=None,  # Not used currently
            query_authors=query_authors if query_authors else None,
            query_affiliation=query_affiliations[0] if query_affiliations else None
        )
        
        logger.info(f"Generated features for {len(features_df)} candidates")
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Feature generation failed: {str(e)}"
        )
    
    if features_df.empty:
        return RecommendResponse(
            recommendations=[],
            total_candidates=0,
            filtered_by_coi=0,
            model_used="none"
        )
    
    # Count COI candidates
    coi_count = int(features_df['coi_flag'].sum()) if 'coi_flag' in features_df.columns else 0
    total_candidates = len(features_df)
    
    # Rank candidates
    try:
        ranked, model_used = rank_candidates(
            features_df,
            query_authors,
            query_affiliations,
            k
        )
        
        logger.info(f"Ranked {len(ranked)} candidates using {model_used}")
        
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ranking failed: {str(e)}"
        )
    
    # Get evidence papers for top candidates
    recommendations = []
    for candidate in ranked:
        evidence = get_evidence_papers(
            db_path=models.db_path,
            author_id=candidate['author_id'],
            query_title=query_title,
            query_abstract=query_abstract,
            top_k=3
        )
        
        recommendations.append(RecommendationResult(
            author_id=candidate['author_id'],
            name=candidate['name'],
            affiliation=candidate['affiliation'],
            score=candidate['score'],
            evidence=[PaperEvidence(**e) for e in evidence]
        ))
    
    return RecommendResponse(
        recommendations=recommendations,
        total_candidates=total_candidates,
        filtered_by_coi=coi_count,
        model_used=model_used
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("Starting Reviewer Recommendation API")
    print("=" * 80)
    print(f"Database: {BACKEND_DB}")
    print(f"CORS Origin: {FRONTEND_ORIGIN}")
    print("=" * 80)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

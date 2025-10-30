"""
Feature Assembly for Author Ranking

This module aggregates paper-level similarity scores into author-level features
for training and ranking reviewer recommendation models.

Key Features:
- TF-IDF and embedding similarity aggregation
- Topic overlap computation (optional)
- Recency weighting
- Publication count features
- Conflict of interest detection
- Paper-to-author aggregation with max/mean statistics

Example:
    >>> from ranker import make_features_for_query
    >>> features_df = make_features_for_query(
    ...     query_text="deep learning",
    ...     db="data/papers.db",
    ...     tfidf_engine=tfidf_engine,
    ...     faiss_index=faiss_index,
    ...     id_map=id_map
    ... )
    >>> features_df.columns
    Index(['author_id', 'author_name', 'tfidf_max', 'tfidf_mean', ...])
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available")

# Import project modules
try:
    from db_utils import get_connection, get_all_papers, list_authors, get_author_papers
    from tfidf_engine import TFIDFEngine
    from utils import recency_weight
    from coauthor_graph import has_conflict
    from topic_model import topic_overlap_score, is_available as topic_available
    import config  # Centralized configuration
except ImportError as e:
    logging.warning(f"Failed to import modules: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_tfidf_similarities(
    query_text: str,
    tfidf_engine: TFIDFEngine,
    topn: int = None
) -> Dict[int, float]:
    """
    Compute TF-IDF similarities between query and all papers.
    
    Args:
        query_text: Query string
        tfidf_engine: Fitted TFIDFEngine instance
        topn: Number of top results to return (default: config.N2_TFIDF)
        
    Returns:
        Dictionary mapping paper_id -> similarity_score
    """
    if topn is None:
        topn = config.N2_TFIDF
    logger.debug(f"Computing TF-IDF similarities for query: {query_text[:50]}...")
    
    # Get similarity scores
    results = tfidf_engine.most_similar(query_text, topn=topn, return_scores=True)
    
    # Convert to dictionary
    sim_dict = {paper_id: score for paper_id, score in results}
    
    logger.debug(f"Found {len(sim_dict)} papers with non-zero TF-IDF similarity")
    
    return sim_dict


def compute_embedding_similarities(
    query_text: str,
    embedding_model: Any,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    topn: int = None
) -> Dict[int, float]:
    """
    Compute embedding similarities using FAISS.
    
    Args:
        query_text: Query string
        embedding_model: Embeddings instance for encoding query
        faiss_index: FAISS index with paper embeddings
        id_map: Array mapping index positions to paper IDs
        topn: Number of top results to return (default: config.N1_FAISS)
        
    Returns:
        Dictionary mapping paper_id -> similarity_score
    """
    if topn is None:
        topn = config.N1_FAISS
    logger.debug(f"Computing embedding similarities for query: {query_text[:50]}...")
    
    # Encode query
    query_vec = embedding_model.encode_texts([query_text], normalize=True)
    query_vec = query_vec.astype(np.float32)
    
    # Search FAISS index
    similarities, indices = faiss_index.search(query_vec, topn)
    
    # Map to paper IDs
    sim_dict = {}
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(id_map):
            paper_id = int(id_map[idx])
            score = float(similarities[0][i])
            if score > 0:
                sim_dict[paper_id] = score
    
    logger.debug(f"Found {len(sim_dict)} papers with non-zero embedding similarity")
    
    return sim_dict


def get_paper_author_mapping(db_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Get mapping from paper_id to author information.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary mapping paper_id -> {
            'author_id': int,
            'author_name': str,
            'affiliation': str,
            'year': int,
            'coauthors': List[str]
        }
    """
    logger.debug("Loading paper-author mapping from database...")
    
    papers = get_all_papers(db_path)
    
    paper_map = {}
    for paper in papers:
        paper_id = paper['id']
        
        # Get coauthors from paper_authors table
        coauthors = []
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT person_name 
                FROM paper_authors 
                WHERE paper_id = ?
            """, (paper_id,))
            coauthors = [row['person_name'] for row in cursor.fetchall()]
        
        paper_map[paper_id] = {
            'author_id': paper['author_id'],
            'author_name': paper.get('author_name', ''),
            'affiliation': paper.get('author_affiliation', ''),
            'year': paper.get('year') or 2020,  # Default year if missing
            'coauthors': coauthors
        }
    
    logger.debug(f"Loaded mapping for {len(paper_map)} papers")
    
    return paper_map


def aggregate_paper_features_to_authors(
    paper_tfidf_scores: Dict[int, float],
    paper_emb_scores: Dict[int, float],
    paper_author_map: Dict[int, Dict[str, Any]],
    db_path: str,
    query_authors: List[str] = None,
    query_affiliation: str = None,
    ref_year: int = None
) -> pd.DataFrame:
    """
    Aggregate paper-level scores into author-level features.
    
    This is the core aggregation function that transforms paper similarities
    into author features suitable for ranking.
    
    Args:
        paper_tfidf_scores: Dict of paper_id -> TF-IDF similarity
        paper_emb_scores: Dict of paper_id -> embedding similarity
        paper_author_map: Dict of paper_id -> author info
        db_path: Path to database for author queries
        query_authors: List of query paper authors (for COI detection)
        query_affiliation: Query paper affiliation (for COI detection)
        ref_year: Reference year for recency (default: current year)
        
    Returns:
        DataFrame with columns:
        - author_id: int
        - author_name: str
        - tfidf_max: float (max TF-IDF score across author's papers)
        - tfidf_mean: float (mean TF-IDF score across author's papers)
        - emb_max: float (max embedding score across author's papers)
        - emb_mean: float (mean embedding score across author's papers)
        - recency_mean: float (mean recency weight across author's papers)
        - recency_max: float (max recency weight across author's papers)
        - pub_count: int (number of papers by author in database)
        - coi_flag: int (1 if conflict of interest, 0 otherwise)
    """
    logger.info("Aggregating paper-level features to author-level...")
    
    if ref_year is None:
        from datetime import datetime
        ref_year = datetime.now().year
    
    # Collect all unique paper IDs
    all_paper_ids = set(paper_tfidf_scores.keys()) | set(paper_emb_scores.keys())
    
    # Group papers by author
    author_papers = defaultdict(list)
    for paper_id in all_paper_ids:
        if paper_id in paper_author_map:
            author_id = paper_author_map[paper_id]['author_id']
            author_papers[author_id].append(paper_id)
    
    logger.info(f"Found {len(author_papers)} authors with relevant papers")
    
    # Aggregate features per author
    author_features = []
    
    for author_id, paper_ids in author_papers.items():
        # Get author info from first paper
        first_paper = paper_author_map[paper_ids[0]]
        author_name = first_paper['author_name']
        author_affiliation = first_paper['affiliation']
        
        # Collect scores for this author's papers
        tfidf_scores = [paper_tfidf_scores.get(pid, 0.0) for pid in paper_ids]
        emb_scores = [paper_emb_scores.get(pid, 0.0) for pid in paper_ids]
        years = [paper_author_map[pid]['year'] for pid in paper_ids]
        
        # Compute recency weights
        recency_weights = [recency_weight(year, ref_year) for year in years]
        
        # Aggregate statistics
        tfidf_max = max(tfidf_scores) if tfidf_scores else 0.0
        tfidf_mean = np.mean(tfidf_scores) if tfidf_scores else 0.0
        emb_max = max(emb_scores) if emb_scores else 0.0
        emb_mean = np.mean(emb_scores) if emb_scores else 0.0
        recency_max = max(recency_weights) if recency_weights else 0.0
        recency_mean = np.mean(recency_weights) if recency_weights else 0.0
        
        # Get total publication count for this author
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers WHERE author_id = ?", (author_id,))
            pub_count = cursor.fetchone()[0]
        
        # Compute COI flag
        coi_flag = 0
        if query_authors and query_affiliation:
            # Get all coauthors for this author's papers
            all_coauthors = []
            for pid in paper_ids:
                all_coauthors.extend(paper_author_map[pid]['coauthors'])
            
            # Check for conflicts
            if has_conflict(
                candidate_name=author_name,
                paper_authors=query_authors,
                paper_affiliations=[query_affiliation] if query_affiliation else [],
                candidate_affiliation=author_affiliation
            ):
                coi_flag = 1
        
        # Build feature dict
        features = {
            'author_id': author_id,
            'author_name': author_name,
            'tfidf_max': float(tfidf_max),
            'tfidf_mean': float(tfidf_mean),
            'emb_max': float(emb_max),
            'emb_mean': float(emb_mean),
            'recency_mean': float(recency_mean),
            'recency_max': float(recency_max),
            'pub_count': int(pub_count),
            'coi_flag': int(coi_flag)
        }
        
        author_features.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(author_features)
    
    # Sort by combined score (for convenience)
    # Uses weights from config.py
    df['combined_score'] = (
        df['emb_max'] * config.W_S +      # Semantic similarity
        df['tfidf_max'] * config.W_L +    # Lexical similarity
        df['recency_max'] * config.W_R +  # Recency
        np.log1p(df['pub_count']) * 0.1 +  # Publication count (log-scaled)
        (1 - df['coi_flag']) * 0.1          # COI penalty
    )
    df = df.sort_values('combined_score', ascending=False)
    df = df.drop('combined_score', axis=1)
    
    logger.info(f"Generated features for {len(df)} authors")
    
    return df


def make_features_for_query(
    query_text: str,
    db: str,
    tfidf_engine: TFIDFEngine,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    embedding_model: Any = None,
    topic_model: Any = None,
    query_authors: List[str] = None,
    query_affiliation: str = None,
    topn_papers: int = None,
    ref_year: int = None
) -> pd.DataFrame:
    """
    Compute author-level features for a query.
    
    This is the main function that orchestrates feature computation:
    1. Compute paper-level TF-IDF similarities
    2. Compute paper-level embedding similarities
    3. Aggregate to author-level features
    4. Add topic overlap (if topic model provided)
    5. Add COI flags (if query author info provided)
    
    Args:
        query_text: Query string (paper title + abstract)
        db: Path to SQLite database
        tfidf_engine: Fitted TFIDFEngine instance
        faiss_index: FAISS index with paper embeddings
        id_map: Array mapping FAISS indices to paper IDs
        embedding_model: Embeddings instance for query encoding (optional, required for embedding scores)
        topic_model: Topic model instance (optional, for topic overlap)
        query_authors: List of query paper authors (optional, for COI)
        query_affiliation: Query paper affiliation (optional, for COI)
        topn_papers: Number of top papers to retrieve per method (default: max(config.N1_FAISS, config.N2_TFIDF))
        ref_year: Reference year for recency (default: current year)
        
    Returns:
        DataFrame with one row per author and columns:
        - author_id: int
        - author_name: str
        - tfidf_max: float
        - tfidf_mean: float
        - emb_max: float
        - emb_mean: float
        - topic_overlap: float (if topic_model provided, else 0.0)
        - recency_mean: float
        - recency_max: float
        - pub_count: int
        - coi_flag: int (1 if conflict, 0 otherwise)
        
    Example:
        >>> from tfidf_engine import TFIDFEngine
        >>> from embedding import Embeddings, load_index
        >>> from ranker import make_features_for_query
        >>> 
        >>> # Load models
        >>> tfidf_engine = TFIDFEngine.load('models/tfidf_vectorizer.pkl')
        >>> emb_model = Embeddings()
        >>> faiss_index, id_map = load_index('data/faiss_index')
        >>> 
        >>> # Compute features
        >>> df = make_features_for_query(
        ...     query_text="Deep learning for image recognition",
        ...     db="data/papers.db",
        ...     tfidf_engine=tfidf_engine,
        ...     faiss_index=faiss_index,
        ...     id_map=id_map,
        ...     embedding_model=emb_model
        ... )
        >>> 
        >>> # Top 10 authors
        >>> print(df.head(10))
    """
    # Use config values if not specified
    if topn_papers is None:
        topn_papers = max(config.N1_FAISS, config.N2_TFIDF)
    
    logger.info("=" * 80)
    logger.info("Making features for query")
    logger.info("=" * 80)
    logger.info(f"Query: {query_text[:100]}...")
    logger.info(f"Database: {db}")
    logger.info(f"Top-N TF-IDF: {config.N2_TFIDF}")
    logger.info(f"Top-N FAISS: {config.N1_FAISS}")
    logger.info(f"Weights: W_S={config.W_S}, W_L={config.W_L}, W_R={config.W_R}")
    logger.info("")
    
    # Step 1: Compute TF-IDF similarities
    logger.info("Step 1: Computing TF-IDF similarities...")
    paper_tfidf_scores = compute_tfidf_similarities(
        query_text=query_text,
        tfidf_engine=tfidf_engine,
        topn=config.N2_TFIDF
    )
    logger.info(f"  Found {len(paper_tfidf_scores)} papers with TF-IDF similarity")
    
    # Step 2: Compute embedding similarities (if model provided)
    logger.info("")
    logger.info("Step 2: Computing embedding similarities...")
    if embedding_model is not None and FAISS_AVAILABLE:
        paper_emb_scores = compute_embedding_similarities(
            query_text=query_text,
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            id_map=id_map,
            topn=config.N1_FAISS
        )
        logger.info(f"  Found {len(paper_emb_scores)} papers with embedding similarity")
    else:
        logger.warning("  Embedding model not provided or FAISS not available, skipping...")
        paper_emb_scores = {}
    
    # Step 3: Load paper-author mapping
    logger.info("")
    logger.info("Step 3: Loading paper-author mapping...")
    paper_author_map = get_paper_author_mapping(db)
    logger.info(f"  Loaded mapping for {len(paper_author_map)} papers")
    
    # Step 4: Aggregate to author-level features
    logger.info("")
    logger.info("Step 4: Aggregating paper features to author features...")
    df = aggregate_paper_features_to_authors(
        paper_tfidf_scores=paper_tfidf_scores,
        paper_emb_scores=paper_emb_scores,
        paper_author_map=paper_author_map,
        db_path=db,
        query_authors=query_authors,
        query_affiliation=query_affiliation,
        ref_year=ref_year
    )
    logger.info(f"  Generated features for {len(df)} authors")
    
    # Step 5: Add topic overlap (if topic model provided)
    if topic_model is not None and topic_available():
        logger.info("")
        logger.info("Step 5: Computing topic overlap...")
        # TODO: Implement topic overlap computation
        # For now, set to 0.0
        df['topic_overlap'] = 0.0
        logger.info("  Topic overlap set to 0.0 (placeholder)")
    else:
        logger.info("")
        logger.info("Step 5: Topic model not provided, skipping...")
        df['topic_overlap'] = 0.0
    
    # Reorder columns
    column_order = [
        'author_id',
        'author_name',
        'tfidf_max',
        'tfidf_mean',
        'emb_max',
        'emb_mean',
        'topic_overlap',
        'recency_mean',
        'recency_max',
        'pub_count',
        'coi_flag'
    ]
    df = df[column_order]
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Feature generation complete")
    logger.info("=" * 80)
    logger.info(f"Authors: {len(df)}")
    logger.info(f"Features: {len(df.columns)}")
    logger.info("")
    
    return df


def save_features_example(df: pd.DataFrame, output_path: str) -> None:
    """
    Save features to JSON for inspection.
    
    Args:
        df: Features DataFrame
        output_path: Path to save JSON file
    """
    # Convert to dict and save
    features_dict = df.head(10).to_dict(orient='records')
    
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features_dict, f, indent=2)
    
    logger.info(f"Saved example features to: {output_path}")


# Example features.json structure
FEATURES_JSON_EXAMPLE = {
    "description": "Author-level features for reviewer ranking",
    "features": [
        {
            "author_id": 1,
            "author_name": "John Smith",
            "tfidf_max": 0.85,
            "tfidf_mean": 0.62,
            "emb_max": 0.78,
            "emb_mean": 0.54,
            "topic_overlap": 0.67,
            "recency_mean": 0.82,
            "recency_max": 0.95,
            "pub_count": 15,
            "coi_flag": 0
        },
        {
            "author_id": 2,
            "author_name": "Jane Doe",
            "tfidf_max": 0.72,
            "tfidf_mean": 0.58,
            "emb_max": 0.81,
            "emb_mean": 0.63,
            "topic_overlap": 0.45,
            "recency_mean": 0.76,
            "recency_max": 0.89,
            "pub_count": 23,
            "coi_flag": 0
        },
        {
            "author_id": 3,
            "author_name": "Bob Johnson",
            "tfidf_max": 0.91,
            "tfidf_mean": 0.74,
            "emb_max": 0.69,
            "emb_mean": 0.51,
            "topic_overlap": 0.88,
            "recency_mean": 0.65,
            "recency_max": 0.78,
            "pub_count": 8,
            "coi_flag": 1
        }
    ],
    "feature_descriptions": {
        "author_id": "Unique author identifier from database",
        "author_name": "Author's full name",
        "tfidf_max": "Maximum TF-IDF similarity across author's papers",
        "tfidf_mean": "Mean TF-IDF similarity across author's papers",
        "emb_max": "Maximum embedding similarity across author's papers",
        "emb_mean": "Mean embedding similarity across author's papers",
        "topic_overlap": "Topic distribution overlap between query and author (0-1)",
        "recency_mean": "Mean recency weight across author's papers (0-1, 1=recent)",
        "recency_max": "Max recency weight across author's papers (0-1, 1=recent)",
        "pub_count": "Total number of papers by author in database",
        "coi_flag": "Conflict of interest flag (1=conflict, 0=no conflict)"
    },
    "aggregation_logic": {
        "paper_to_author": "Papers are grouped by author_id, then aggregated",
        "max_scores": "Takes maximum score across all author's papers (captures best match)",
        "mean_scores": "Takes average score across all author's papers (captures overall relevance)",
        "recency": "Computed using exponential decay: exp(-age/tau) where age = ref_year - pub_year",
        "coi": "Detected by checking: same name, coauthorship, or same affiliation"
    }
}


if __name__ == '__main__':
    # Save example features.json
    import json
    output_path = Path(__file__).parent / 'features.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(FEATURES_JSON_EXAMPLE, f, indent=2)
    print(f"Saved example features to: {output_path}")

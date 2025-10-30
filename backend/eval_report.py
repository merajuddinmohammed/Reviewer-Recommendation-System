"""
Evaluation Report Script - Ranking Method Comparison

This script evaluates different ranking methods for reviewer recommendation:
- Method A: TF-IDF only (lexical matching)
- Method B: Embeddings only (semantic matching)
- Method C: Hybrid weighted (0.55*emb + 0.25*tfidf + 0.20*recency)
- Method D: LambdaRank (learned ranker)

Evaluation Metrics:
- Precision@5 (P@5): Fraction of relevant items in top 5
- nDCG@10: Normalized Discounted Cumulative Gain at 10

Uses weak positive labels from training data generation:
- Coauthors
- Top TF-IDF neighbors
- Top embedding neighbors

Output:
- CSV: data/eval_metrics_per_query.csv (detailed metrics per query)
- Markdown: data/eval_report.md (summary tables and analysis)

CLI Arguments:
- --db: Path to database (default: data/papers.db)
- --queries: Number of queries to sample (default: 100)
- --seed: Random seed for reproducibility (default: 42)
- --out-csv: Output CSV path
- --out-md: Output markdown report path

Author: Applied AI Assignment
Date: December 2024
"""

import logging
import argparse
import sys
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# Import required modules
from db_utils import get_connection, get_all_papers, list_authors
from tfidf_engine import TFIDFEngine
from embedding import Embeddings
from ranker import make_features_for_query, compute_tfidf_similarities, compute_embedding_similarities
from coauthor_graph import get_coauthors_for_author, has_conflict
from utils import recency_weight
import faiss
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def precision_at_k(relevant_items: Set[int], ranked_list: List[int], k: int = 5) -> float:
    """
    Calculate Precision@K.
    
    Args:
        relevant_items: Set of relevant item IDs
        ranked_list: Ordered list of recommended item IDs
        k: Cutoff position
        
    Returns:
        Precision@K score (0.0 to 1.0)
        
    Examples:
        >>> relevant = {1, 3, 5}
        >>> ranked = [1, 2, 3, 4, 5]
        >>> precision_at_k(relevant, ranked, k=5)
        0.6
    """
    if not ranked_list or k <= 0:
        return 0.0
    
    # Get top k items
    top_k = ranked_list[:k]
    
    # Count how many are relevant
    hits = sum(1 for item in top_k if item in relevant_items)
    
    return hits / k


def dcg_at_k(relevant_items: Set[int], ranked_list: List[int], k: int = 10) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    DCG = sum_{i=1}^{k} (2^{rel_i} - 1) / log_2(i + 1)
    
    For binary relevance: rel_i = 1 if relevant, 0 otherwise
    
    Args:
        relevant_items: Set of relevant item IDs
        ranked_list: Ordered list of recommended item IDs
        k: Cutoff position
        
    Returns:
        DCG@K score
    """
    if not ranked_list or k <= 0:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k], start=1):
        if item in relevant_items:
            # Binary relevance: rel = 1
            # DCG formula: (2^1 - 1) / log2(i+1) = 1 / log2(i+1)
            dcg += 1.0 / np.log2(i + 1)
    
    return dcg


def ndcg_at_k(relevant_items: Set[int], ranked_list: List[int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    nDCG = DCG / IDCG
    
    where IDCG is the DCG of the ideal ranking (all relevant items first)
    
    Args:
        relevant_items: Set of relevant item IDs
        ranked_list: Ordered list of recommended item IDs
        k: Cutoff position
        
    Returns:
        nDCG@K score (0.0 to 1.0)
        
    Examples:
        >>> relevant = {1, 3, 5}
        >>> ranked = [1, 3, 2, 4, 5]
        >>> ndcg_at_k(relevant, ranked, k=5)
        # Close to 1.0 since most relevant items are near top
    """
    if not relevant_items or not ranked_list or k <= 0:
        return 0.0
    
    # Calculate DCG for actual ranking
    actual_dcg = dcg_at_k(relevant_items, ranked_list, k)
    
    # Calculate ideal DCG (all relevant items ranked first)
    # Ideal ranking: put min(len(relevant), k) relevant items at top
    ideal_ranking = list(relevant_items)[:k] + [0] * max(0, k - len(relevant_items))
    ideal_dcg = dcg_at_k(relevant_items, ideal_ranking, k)
    
    if ideal_dcg == 0.0:
        return 0.0
    
    return actual_dcg / ideal_dcg


# ============================================================================
# Sample Query Papers
# ============================================================================

def sample_query_papers(
    db_path: Path,
    n_queries: int,
    min_year: Optional[int] = 2020,
    seed: int = 42
) -> List[Dict]:
    """
    Sample N papers to use as queries for evaluation.
    
    Args:
        db_path: Path to SQLite database
        n_queries: Number of queries to sample
        min_year: Minimum publication year
        seed: Random seed
        
    Returns:
        List of query paper dicts with id, title, abstract, author_id, year
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Query papers with valid title and abstract
            query = """
                SELECT id, title, abstract, author_id, year
                FROM papers
                WHERE title IS NOT NULL
                  AND abstract IS NOT NULL
                  AND LENGTH(abstract) > 50
            """
            
            if min_year:
                query += f" AND year >= {min_year}"
            
            cursor.execute(query)
            all_papers = cursor.fetchall()
            
            if not all_papers:
                logger.error("No papers found in database")
                return []
            
            # Sample n_queries papers
            n_queries = min(n_queries, len(all_papers))
            sampled = random.sample(all_papers, n_queries)
            
            query_papers = []
            for paper_id, title, abstract, author_id, year in sampled:
                query_papers.append({
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "author_id": author_id,
                    "year": year or 2020
                })
            
            logger.info(f"Sampled {len(query_papers)} query papers")
            return query_papers
            
    except Exception as e:
        logger.error(f"Error sampling query papers: {e}")
        return []


# ============================================================================
# Generate Weak Positive Labels
# ============================================================================

def generate_positive_labels(
    query_paper: Dict,
    db_path: Path,
    tfidf_engine: TFIDFEngine,
    embedding_model: Embeddings,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    topn: int = 10
) -> Set[int]:
    """
    Generate weak positive labels using same heuristics as training data.
    
    Positive label heuristics:
    1. Coauthors of query paper author
    2. Top TF-IDF neighbors
    3. Top embedding neighbors
    
    Excludes:
    - Query paper's own author (COI)
    - Authors with conflicts of interest
    
    Args:
        query_paper: Dict with query paper info
        db_path: Path to database
        tfidf_engine: TF-IDF engine
        embedding_model: Embedding model
        faiss_index: FAISS index
        id_map: ID mapping
        topn: Number of top neighbors to consider
        
    Returns:
        Set of positive author IDs
    """
    positive_authors = set()
    query_author_id = query_paper["author_id"]
    
    # Get query text
    query_text = f"{query_paper['title']} {query_paper['abstract']}"
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get query author name
            cursor.execute("SELECT name FROM authors WHERE id = ?", (query_author_id,))
            author_row = cursor.fetchone()
            query_author_name = author_row[0] if author_row else None
            
            # 1. Add coauthors
            if query_author_name:
                coauthors = get_coauthors_for_author(db_path, query_author_name)
                for coauthor in coauthors:
                    if coauthor["author_id"] != query_author_id:
                        positive_authors.add(coauthor["author_id"])
            
            # 2. Add authors from top TF-IDF neighbors
            tfidf_sims = compute_tfidf_similarities(query_text, tfidf_engine, topn=topn)
            for paper_id in list(tfidf_sims.keys())[:topn]:
                if paper_id != query_paper["id"]:
                    cursor.execute("SELECT author_id FROM papers WHERE id = ?", (paper_id,))
                    result = cursor.fetchone()
                    if result and result[0] != query_author_id:
                        positive_authors.add(result[0])
            
            # 3. Add authors from top embedding neighbors
            emb_sims = compute_embedding_similarities(
                query_text,
                embedding_model,
                faiss_index,
                id_map,
                topn=topn
            )
            for paper_id in list(emb_sims.keys())[:topn]:
                if paper_id != query_paper["id"]:
                    cursor.execute("SELECT author_id FROM papers WHERE id = ?", (paper_id,))
                    result = cursor.fetchone()
                    if result and result[0] != query_author_id:
                        positive_authors.add(result[0])
    
    except Exception as e:
        logger.error(f"Error generating positive labels: {e}")
    
    logger.debug(f"Generated {len(positive_authors)} positive labels for query {query_paper['id']}")
    return positive_authors


# ============================================================================
# Ranking Methods
# ============================================================================

def rank_by_tfidf(
    query_paper: Dict,
    db_path: Path,
    tfidf_engine: TFIDFEngine,
    topn: int = 100
) -> List[int]:
    """
    Rank authors by TF-IDF similarity only.
    
    Args:
        query_paper: Query paper dict
        db_path: Database path
        tfidf_engine: TF-IDF engine
        topn: Number of results
        
    Returns:
        List of author IDs ranked by TF-IDF score
    """
    query_text = f"{query_paper['title']} {query_paper['abstract']}"
    
    # Get paper similarities
    paper_sims = compute_tfidf_similarities(query_text, tfidf_engine, topn=topn)
    
    # Aggregate to author level (max similarity)
    author_scores = defaultdict(float)
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            for paper_id, sim in paper_sims.items():
                cursor.execute("SELECT author_id FROM papers WHERE id = ?", (paper_id,))
                result = cursor.fetchone()
                if result:
                    author_id = result[0]
                    # Exclude query author (COI)
                    if author_id != query_paper["author_id"]:
                        author_scores[author_id] = max(author_scores[author_id], sim)
    
    except Exception as e:
        logger.error(f"Error in rank_by_tfidf: {e}")
        return []
    
    # Sort by score
    ranked = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
    return [author_id for author_id, score in ranked]


def rank_by_embeddings(
    query_paper: Dict,
    db_path: Path,
    embedding_model: Embeddings,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    topn: int = 100
) -> List[int]:
    """
    Rank authors by embedding similarity only.
    
    Args:
        query_paper: Query paper dict
        db_path: Database path
        embedding_model: Embedding model
        faiss_index: FAISS index
        id_map: ID mapping
        topn: Number of results
        
    Returns:
        List of author IDs ranked by embedding score
    """
    query_text = f"{query_paper['title']} {query_paper['abstract']}"
    
    # Get paper similarities
    paper_sims = compute_embedding_similarities(
        query_text,
        embedding_model,
        faiss_index,
        id_map,
        topn=topn
    )
    
    # Aggregate to author level (max similarity)
    author_scores = defaultdict(float)
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            for paper_id, sim in paper_sims.items():
                cursor.execute("SELECT author_id FROM papers WHERE id = ?", (paper_id,))
                result = cursor.fetchone()
                if result:
                    author_id = result[0]
                    # Exclude query author (COI)
                    if author_id != query_paper["author_id"]:
                        author_scores[author_id] = max(author_scores[author_id], sim)
    
    except Exception as e:
        logger.error(f"Error in rank_by_embeddings: {e}")
        return []
    
    # Sort by score
    ranked = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
    return [author_id for author_id, score in ranked]


def rank_by_hybrid(
    query_paper: Dict,
    db_path: Path,
    tfidf_engine: TFIDFEngine,
    embedding_model: Embeddings,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    topn: int = 100
) -> List[int]:
    """
    Rank authors by hybrid weighted score.
    
    Score = 0.55 * emb_max + 0.25 * tfidf_max + 0.20 * recency_max
    
    Args:
        query_paper: Query paper dict
        db_path: Database path
        tfidf_engine: TF-IDF engine
        embedding_model: Embedding model
        faiss_index: FAISS index
        id_map: ID mapping
        topn: Number of results
        
    Returns:
        List of author IDs ranked by hybrid score
    """
    query_text = f"{query_paper['title']} {query_paper['abstract']}"
    
    # Get similarities
    tfidf_sims = compute_tfidf_similarities(query_text, tfidf_engine, topn=topn)
    emb_sims = compute_embedding_similarities(
        query_text,
        embedding_model,
        faiss_index,
        id_map,
        topn=topn
    )
    
    # Combine paper IDs
    all_paper_ids = set(tfidf_sims.keys()) | set(emb_sims.keys())
    
    # Aggregate to author level
    author_data = defaultdict(lambda: {"tfidf": 0.0, "emb": 0.0, "recency": 0.0})
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            for paper_id in all_paper_ids:
                cursor.execute("SELECT author_id, year FROM papers WHERE id = ?", (paper_id,))
                result = cursor.fetchone()
                if result:
                    author_id, year = result
                    # Exclude query author (COI)
                    if author_id != query_paper["author_id"]:
                        tfidf_sim = tfidf_sims.get(paper_id, 0.0)
                        emb_sim = emb_sims.get(paper_id, 0.0)
                        recency = recency_weight(year or 2020)
                        
                        # Max aggregation
                        author_data[author_id]["tfidf"] = max(author_data[author_id]["tfidf"], tfidf_sim)
                        author_data[author_id]["emb"] = max(author_data[author_id]["emb"], emb_sim)
                        author_data[author_id]["recency"] = max(author_data[author_id]["recency"], recency)
    
    except Exception as e:
        logger.error(f"Error in rank_by_hybrid: {e}")
        return []
    
    # Compute weighted scores
    author_scores = {}
    for author_id, data in author_data.items():
        score = (
            0.55 * data["emb"] +
            0.25 * data["tfidf"] +
            0.20 * data["recency"]
        )
        author_scores[author_id] = score
    
    # Sort by score
    ranked = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
    return [author_id for author_id, score in ranked]


def rank_by_lambdarank(
    query_paper: Dict,
    db_path: Path,
    tfidf_engine: TFIDFEngine,
    embedding_model: Embeddings,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    lgbm_model: Any,
    topn: int = 100
) -> List[int]:
    """
    Rank authors using LambdaRank model.
    
    Args:
        query_paper: Query paper dict
        db_path: Database path
        tfidf_engine: TF-IDF engine
        embedding_model: Embedding model
        faiss_index: FAISS index
        id_map: ID mapping
        lgbm_model: Trained LightGBM model
        topn: Number of results
        
    Returns:
        List of author IDs ranked by LambdaRank score
    """
    query_text = f"{query_paper['title']} {query_paper['abstract']}"
    
    # Get features for all candidates
    try:
        features_df = make_features_for_query(
            query_text=query_text,
            db=db_path,
            tfidf_engine=tfidf_engine,
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            id_map=id_map,
            query_authors=[],  # No authors to exclude
            query_affiliation=None
        )
        
        if features_df.empty:
            return []
        
        # Exclude query author
        features_df = features_df[features_df["author_id"] != query_paper["author_id"]]
        
        # Get feature columns (exclude author_id, author_name)
        feature_cols = [col for col in features_df.columns if col not in ["author_id", "author_name"]]
        X = features_df[feature_cols].values
        
        # Predict scores
        scores = lgbm_model.predict(X)
        
        # Rank by score
        features_df["score"] = scores
        features_df = features_df.sort_values("score", ascending=False)
        
        return features_df["author_id"].tolist()[:topn]
    
    except Exception as e:
        logger.error(f"Error in rank_by_lambdarank: {e}")
        return []


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_methods(
    db_path: Path,
    n_queries: int = 100,
    seed: int = 42,
    out_csv: Optional[Path] = None,
    out_md: Optional[Path] = None
):
    """
    Evaluate all ranking methods and generate report.
    
    Args:
        db_path: Database path
        n_queries: Number of queries to evaluate
        seed: Random seed
        out_csv: Output CSV path
        out_md: Output markdown path
    """
    logger.info("=" * 80)
    logger.info("Evaluation Report - Ranking Method Comparison")
    logger.info("=" * 80)
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Load models
    logger.info("\n1. Loading models...")
    
    try:
        # TF-IDF
        tfidf_path = Path("models/tfidf_vectorizer.pkl")
        if not tfidf_path.exists():
            logger.error(f"TF-IDF model not found: {tfidf_path}")
            return
        
        logger.info(f"   Loading TF-IDF model from {tfidf_path}")
        tfidf_data = joblib.load(tfidf_path)
        tfidf_engine = TFIDFEngine()
        tfidf_engine.vectorizer = tfidf_data["vectorizer"]
        tfidf_engine.corpus_matrix = tfidf_data["corpus_matrix"]
        tfidf_engine.paper_ids = tfidf_data["paper_ids"]
        
        # FAISS
        faiss_path = Path("data/faiss_index.faiss")
        id_map_path = Path("data/id_map.npy")
        if not faiss_path.exists() or not id_map_path.exists():
            logger.error(f"FAISS index not found: {faiss_path}")
            return
        
        logger.info(f"   Loading FAISS index from {faiss_path}")
        faiss_index = faiss.read_index(str(faiss_path))
        id_map = np.load(str(id_map_path), allow_pickle=True)
        
        # Embeddings (for query encoding)
        logger.info("   Loading embedding model...")
        embedding_model = Embeddings(model_name="allenai/scibert_scivocab_uncased")
        
        # LightGBM (optional)
        lgbm_path = Path("models/lgbm_ranker.pkl")
        lgbm_model = None
        if lgbm_path.exists():
            logger.info(f"   Loading LightGBM model from {lgbm_path}")
            import pickle
            with open(lgbm_path, "rb") as f:
                lgbm_model = pickle.load(f)
        else:
            logger.warning("   LightGBM model not found, skipping Method D")
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return
    
    # Sample query papers
    logger.info(f"\n2. Sampling {n_queries} query papers...")
    query_papers = sample_query_papers(db_path, n_queries, min_year=2020, seed=seed)
    
    if not query_papers:
        logger.error("No query papers sampled")
        return
    
    logger.info(f"   Sampled {len(query_papers)} queries")
    
    # Evaluate each query
    logger.info(f"\n3. Evaluating queries...")
    
    results = []
    
    for i, query_paper in enumerate(query_papers, 1):
        logger.info(f"\n   Query {i}/{len(query_papers)}: {query_paper['title'][:60]}...")
        
        # Generate positive labels
        positive_authors = generate_positive_labels(
            query_paper,
            db_path,
            tfidf_engine,
            embedding_model,
            faiss_index,
            id_map,
            topn=10
        )
        
        if not positive_authors:
            logger.warning(f"   No positive labels for query {i}, skipping")
            continue
        
        logger.info(f"   Generated {len(positive_authors)} positive labels")
        
        # Method A: TF-IDF only
        ranked_a = rank_by_tfidf(query_paper, db_path, tfidf_engine, topn=100)
        p5_a = precision_at_k(positive_authors, ranked_a, k=5)
        ndcg10_a = ndcg_at_k(positive_authors, ranked_a, k=10)
        
        # Method B: Embeddings only
        ranked_b = rank_by_embeddings(query_paper, db_path, embedding_model, faiss_index, id_map, topn=100)
        p5_b = precision_at_k(positive_authors, ranked_b, k=5)
        ndcg10_b = ndcg_at_k(positive_authors, ranked_b, k=10)
        
        # Method C: Hybrid
        ranked_c = rank_by_hybrid(query_paper, db_path, tfidf_engine, embedding_model, faiss_index, id_map, topn=100)
        p5_c = precision_at_k(positive_authors, ranked_c, k=5)
        ndcg10_c = ndcg_at_k(positive_authors, ranked_c, k=10)
        
        # Method D: LambdaRank (if available)
        p5_d, ndcg10_d = None, None
        if lgbm_model:
            ranked_d = rank_by_lambdarank(
                query_paper, db_path, tfidf_engine, embedding_model,
                faiss_index, id_map, lgbm_model, topn=100
            )
            p5_d = precision_at_k(positive_authors, ranked_d, k=5)
            ndcg10_d = ndcg_at_k(positive_authors, ranked_d, k=10)
        
        # Store results
        results.append({
            "query_id": query_paper["id"],
            "query_title": query_paper["title"],
            "n_positives": len(positive_authors),
            "tfidf_p5": p5_a,
            "tfidf_ndcg10": ndcg10_a,
            "emb_p5": p5_b,
            "emb_ndcg10": ndcg10_b,
            "hybrid_p5": p5_c,
            "hybrid_ndcg10": ndcg10_c,
            "lambdarank_p5": p5_d,
            "lambdarank_ndcg10": ndcg10_d
        })
        
        logger.info(f"   TF-IDF: P@5={p5_a:.3f}, nDCG@10={ndcg10_a:.3f}")
        logger.info(f"   Embeddings: P@5={p5_b:.3f}, nDCG@10={ndcg10_b:.3f}")
        logger.info(f"   Hybrid: P@5={p5_c:.3f}, nDCG@10={ndcg10_c:.3f}")
        if lgbm_model:
            logger.info(f"   LambdaRank: P@5={p5_d:.3f}, nDCG@10={ndcg10_d:.3f}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save CSV
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        logger.info(f"\n4. Saved detailed metrics to {out_csv}")
    
    # Generate markdown report
    if out_md:
        generate_markdown_report(results_df, out_md, lgbm_model is not None)
        logger.info(f"5. Saved report to {out_md}")
    
    # Print summary
    print_summary(results_df, lgbm_model is not None)


def generate_markdown_report(results_df: pd.DataFrame, out_path: Path, has_lambdarank: bool):
    """Generate markdown evaluation report."""
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute summary statistics
    methods = [
        ("TF-IDF Only", "tfidf"),
        ("Embeddings Only", "emb"),
        ("Hybrid Weighted", "hybrid")
    ]
    
    if has_lambdarank:
        methods.append(("LambdaRank", "lambdarank"))
    
    with open(out_path, "w") as f:
        f.write("# Evaluation Report - Ranking Method Comparison\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Number of Queries**: {len(results_df)}\n\n")
        f.write(f"**Evaluation Metrics**: Precision@5 (P@5), nDCG@10\n\n")
        f.write("---\n\n")
        
        # Summary table
        f.write("## Summary Statistics\n\n")
        f.write("| Method | P@5 Mean | P@5 Std | nDCG@10 Mean | nDCG@10 Std |\n")
        f.write("|--------|----------|---------|--------------|-------------|\n")
        
        for method_name, method_prefix in methods:
            p5_col = f"{method_prefix}_p5"
            ndcg_col = f"{method_prefix}_ndcg10"
            
            if p5_col in results_df.columns and not results_df[p5_col].isna().all():
                p5_mean = results_df[p5_col].mean()
                p5_std = results_df[p5_col].std()
                ndcg_mean = results_df[ndcg_col].mean()
                ndcg_std = results_df[ndcg_col].std()
                
                f.write(f"| {method_name} | {p5_mean:.4f} | {p5_std:.4f} | {ndcg_mean:.4f} | {ndcg_std:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Best method
        f.write("## Best Performing Method\n\n")
        
        # Find best by P@5
        best_p5_method = None
        best_p5_score = 0.0
        for method_name, method_prefix in methods:
            p5_col = f"{method_prefix}_p5"
            if p5_col in results_df.columns and not results_df[p5_col].isna().all():
                mean_score = results_df[p5_col].mean()
                if mean_score > best_p5_score:
                    best_p5_score = mean_score
                    best_p5_method = method_name
        
        f.write(f"**By P@5**: {best_p5_method} ({best_p5_score:.4f})\n\n")
        
        # Find best by nDCG@10
        best_ndcg_method = None
        best_ndcg_score = 0.0
        for method_name, method_prefix in methods:
            ndcg_col = f"{method_prefix}_ndcg10"
            if ndcg_col in results_df.columns and not results_df[ndcg_col].isna().all():
                mean_score = results_df[ndcg_col].mean()
                if mean_score > best_ndcg_score:
                    best_ndcg_score = mean_score
                    best_ndcg_method = method_name
        
        f.write(f"**By nDCG@10**: {best_ndcg_method} ({best_ndcg_score:.4f})\n\n")
        
        f.write("---\n\n")
        
        # Analysis
        f.write("## Analysis\n\n")
        f.write("### Lexical vs Semantic\n\n")
        
        if "tfidf_p5" in results_df.columns and "emb_p5" in results_df.columns:
            tfidf_mean = results_df["tfidf_p5"].mean()
            emb_mean = results_df["emb_p5"].mean()
            
            if emb_mean > tfidf_mean:
                f.write(f"**Semantic matching (embeddings)** outperforms lexical matching (TF-IDF) by {(emb_mean - tfidf_mean):.4f} on P@5. ")
                f.write("This suggests that semantic understanding is more important than keyword matching for reviewer recommendation.\n\n")
            else:
                f.write(f"**Lexical matching (TF-IDF)** outperforms semantic matching (embeddings) by {(tfidf_mean - emb_mean):.4f} on P@5. ")
                f.write("This suggests that keyword matching is surprisingly effective for reviewer recommendation.\n\n")
        
        f.write("### Hybrid Approach\n\n")
        
        if "hybrid_p5" in results_df.columns:
            hybrid_mean = results_df["hybrid_p5"].mean()
            tfidf_mean = results_df["tfidf_p5"].mean()
            emb_mean = results_df["emb_p5"].mean()
            
            if hybrid_mean > max(tfidf_mean, emb_mean):
                f.write(f"**Hybrid weighted method** achieves the best performance ({hybrid_mean:.4f}), ")
                f.write("demonstrating that combining multiple signals (embeddings, TF-IDF, recency) is beneficial.\n\n")
            else:
                f.write(f"**Hybrid method** does not improve over individual methods. ")
                f.write("This suggests that simple max aggregation may be sufficient, or that weight tuning is needed.\n\n")
        
        f.write("### Learning-to-Rank\n\n")
        
        if has_lambdarank and "lambdarank_p5" in results_df.columns:
            lr_mean = results_df["lambdarank_p5"].mean()
            hybrid_mean = results_df["hybrid_p5"].mean()
            
            if lr_mean > hybrid_mean:
                f.write(f"**LambdaRank** achieves the best performance ({lr_mean:.4f}), ")
                f.write("demonstrating that learning optimal feature weights and non-linear combinations improves ranking quality.\n\n")
            else:
                f.write(f"**LambdaRank** does not improve over hybrid method. ")
                f.write("This may be due to limited training data or weak positive labels. ")
                f.write("More training data or better labels could improve performance.\n\n")
        else:
            f.write("LambdaRank model not available for evaluation.\n\n")
        
        f.write("---\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("1. **Semantic vs Lexical**: ")
        
        if "tfidf_p5" in results_df.columns and "emb_p5" in results_df.columns:
            tfidf_mean = results_df["tfidf_p5"].mean()
            emb_mean = results_df["emb_p5"].mean()
            winner = "Semantic (embeddings)" if emb_mean > tfidf_mean else "Lexical (TF-IDF)"
            f.write(f"{winner} is more effective for reviewer recommendation.\n\n")
        
        f.write("2. **Hybrid Approach**: ")
        if "hybrid_p5" in results_df.columns:
            f.write("Combining multiple signals can improve performance, but weight tuning is important.\n\n")
        
        f.write("3. **Learning-to-Rank**: ")
        if has_lambdarank:
            f.write("Learning optimal feature weights can further improve ranking quality with sufficient training data.\n\n")
        else:
            f.write("Not evaluated (model not available).\n\n")
        
        f.write("4. **Recommendations**:\n")
        f.write(f"   - Deploy **{best_p5_method}** for best P@5 performance\n")
        f.write("   - Consider A/B testing different methods in production\n")
        f.write("   - Collect user feedback to improve labels and training data\n")
        f.write("   - Monitor metrics over time to detect degradation\n\n")
    
    logger.info(f"   Generated markdown report: {out_path}")


def print_summary(results_df: pd.DataFrame, has_lambdarank: bool):
    """Print summary to console."""
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    methods = [
        ("TF-IDF Only", "tfidf"),
        ("Embeddings Only", "emb"),
        ("Hybrid Weighted", "hybrid")
    ]
    
    if has_lambdarank:
        methods.append(("LambdaRank", "lambdarank"))
    
    print(f"\nNumber of queries: {len(results_df)}")
    print("\nMethod Performance:")
    print("-" * 80)
    print(f"{'Method':<20} {'P@5 Mean':<12} {'P@5 Std':<12} {'nDCG@10 Mean':<15} {'nDCG@10 Std':<15}")
    print("-" * 80)
    
    for method_name, method_prefix in methods:
        p5_col = f"{method_prefix}_p5"
        ndcg_col = f"{method_prefix}_ndcg10"
        
        if p5_col in results_df.columns and not results_df[p5_col].isna().all():
            p5_mean = results_df[p5_col].mean()
            p5_std = results_df[p5_col].std()
            ndcg_mean = results_df[ndcg_col].mean()
            ndcg_std = results_df[ndcg_col].std()
            
            print(f"{method_name:<20} {p5_mean:<12.4f} {p5_std:<12.4f} {ndcg_mean:<15.4f} {ndcg_std:<15.4f}")
    
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ranking methods for reviewer recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/papers.db"),
        help="Path to SQLite database (default: data/papers.db)"
    )
    
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of queries to sample (default: 100)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/eval_metrics_per_query.csv"),
        help="Output CSV path (default: data/eval_metrics_per_query.csv)"
    )
    
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("data/eval_report.md"),
        help="Output markdown path (default: data/eval_report.md)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)
    
    # Run evaluation
    evaluate_methods(
        db_path=args.db,
        n_queries=args.queries,
        seed=args.seed,
        out_csv=args.out_csv,
        out_md=args.out_md
    )


if __name__ == "__main__":
    main()

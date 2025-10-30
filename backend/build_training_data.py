"""
Build Training Data for Learning-to-Rank Model

This script generates synthetic training data by:
1. Sampling M papers as queries
2. Labeling authors as positive (proxy) or negative
3. Computing features for each (query, author) pair
4. Outputting train.parquet with group column for LTR

Positive Label Heuristics:
- Authors in paper's reference list (if available)
- Authors in co-author neighborhood
- Top TF-IDF/embedding neighbors (weak positives)
- Exclude: Same author as query paper (COI)
- Exclude: Authors with conflicts of interest

Negative Label Heuristics:
- Random authors not in positive set
- Sampled uniformly from corpus

Features:
- All features from ranker.make_features_for_query()
- 11 features: tfidf_max/mean, emb_max/mean, topic_overlap, 
  recency_mean/max, pub_count, coi_flag

Output Schema:
- query_id: int - Paper ID used as query
- author_id: int - Candidate author ID
- y: int - Label (1=positive proxy, 0=negative)
- group: int - Group ID for LTR (one group per query)
- <11 feature columns>

CLI Arguments:
- --db: Path to database
- --out: Output parquet file path
- --queries: Number of queries to sample (default: 100)
- --positives: Number of positive labels per query (default: 3)
- --negatives: Number of negative labels per query (default: 20)
- --min-year: Minimum publication year for query papers
- --seed: Random seed for reproducibility

Author: Applied AI Assignment
Date: December 2024
"""

import logging
import sqlite3
import argparse
import sys
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

# Import required modules
from db_utils import get_connection
from tfidf_engine import TFIDFEngine
from embedding import Embeddings
from ranker import make_features_for_query
from coauthor_graph import has_conflict, get_coauthors_for_author
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Query Paper Sampling
# ============================================================================

def sample_query_papers(
    db_path: Path,
    n_queries: int,
    min_year: Optional[int] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Sample N papers to use as queries.
    
    Selects papers with:
    - Valid title and abstract
    - Publication year >= min_year (if specified)
    - Randomly sampled
    
    Args:
        db_path: Path to SQLite database
        n_queries: Number of queries to sample
        min_year: Minimum publication year (optional)
        seed: Random seed for reproducibility
        
    Returns:
        List of dicts with query paper info:
        [
            {
                'paper_id': int,
                'title': str,
                'abstract': str,
                'author_id': int,
                'author_name': str,
                'year': int
            },
            ...
        ]
    """
    random.seed(seed)
    queries = []
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Build query with optional year filter
            sql = """
                SELECT p.id, p.title, p.abstract, p.author_id, a.name, p.year
                FROM papers p
                JOIN authors a ON p.author_id = a.id
                WHERE p.title IS NOT NULL 
                AND p.title != ''
                AND (p.abstract IS NOT NULL AND p.abstract != '')
            """
            
            params = []
            if min_year:
                sql += " AND p.year >= ?"
                params.append(min_year)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning("No papers found matching criteria")
                return []
            
            # Randomly sample
            if len(rows) <= n_queries:
                logger.warning(f"Only {len(rows)} papers available, using all")
                sampled_rows = rows
            else:
                sampled_rows = random.sample(rows, n_queries)
            
            # Convert to dicts
            for row in sampled_rows:
                queries.append({
                    'paper_id': row[0],
                    'title': row[1],
                    'abstract': row[2],
                    'author_id': row[3],
                    'author_name': row[4],
                    'year': row[5]
                })
            
            logger.info(f"Sampled {len(queries)} query papers")
            return queries
            
    except Exception as e:
        logger.error(f"Error sampling query papers: {e}")
        return []


# ============================================================================
# Positive Label Generation
# ============================================================================

def get_reference_authors(
    db_path: Path,
    paper_id: int
) -> Set[int]:
    """
    Get author IDs from paper's reference list.
    
    Queries the references table and extracts author IDs
    from referenced papers.
    
    Args:
        db_path: Path to SQLite database
        paper_id: Query paper ID
        
    Returns:
        Set of author IDs from references
    """
    author_ids = set()
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if references table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='references'
            """)
            
            if not cursor.fetchone():
                logger.debug(f"References table does not exist")
                return author_ids
            
            # Check if paper has references
            cursor.execute("""
                SELECT COUNT(*) FROM references WHERE paper_id = ?
            """, (paper_id,))
            
            ref_count = cursor.fetchone()[0]
            
            if ref_count == 0:
                logger.debug(f"Paper {paper_id} has no references")
                return author_ids
            
            # Get author IDs from referenced papers
            cursor.execute("""
                SELECT DISTINCT p.author_id
                FROM references r
                JOIN papers p ON r.ref_paper_id = p.id
                WHERE r.paper_id = ?
                AND p.author_id IS NOT NULL
            """, (paper_id,))
            
            author_ids = {row[0] for row in cursor.fetchall()}
            logger.debug(f"Paper {paper_id} references {len(author_ids)} authors")
            
    except Exception as e:
        logger.error(f"Error getting reference authors for paper {paper_id}: {e}")
    
    return author_ids


def get_coauthor_neighborhood(
    db_path: Path,
    author_name: str,
    max_authors: int = 10
) -> Set[int]:
    """
    Get author IDs from co-author neighborhood.
    
    Args:
        db_path: Path to SQLite database
        author_name: Query author name
        max_authors: Maximum coauthors to return
        
    Returns:
        Set of coauthor author IDs
    """
    try:
        coauthors = get_coauthors_for_author(db_path, author_name)
        
        # Take top N by paper count
        top_coauthors = coauthors[:max_authors]
        
        author_ids = {c['author_id'] for c in top_coauthors}
        logger.debug(f"Author '{author_name}' has {len(author_ids)} coauthors")
        
        return author_ids
        
    except Exception as e:
        logger.error(f"Error getting coauthor neighborhood: {e}")
        return set()


def get_weak_positive_authors(
    query_text: str,
    db_path: Path,
    tfidf_engine: TFIDFEngine,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    emb_model: Embeddings,
    query_author_id: int,
    topn: int = 10
) -> Set[int]:
    """
    Get weak positive author IDs using TF-IDF/embedding similarity.
    
    Finds papers most similar to query, then extracts their authors.
    Excludes the query author to avoid COI.
    
    Args:
        query_text: Query paper text
        db_path: Path to database
        tfidf_engine: TFIDFEngine instance
        faiss_index: FAISS index
        id_map: FAISS ID mapping
        emb_model: Embeddings model
        query_author_id: Author ID of query paper (to exclude)
        topn: Number of similar papers to consider
        
    Returns:
        Set of weak positive author IDs
    """
    author_ids = set()
    
    try:
        # Compute TF-IDF similarities
        tfidf_sims = tfidf_engine.transform_query(query_text, return_scores=True)
        
        # Compute embedding similarities
        query_emb = emb_model.encode([query_text])[0]
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        # FAISS search
        distances, indices = faiss_index.search(query_emb, min(topn * 2, faiss_index.ntotal))
        
        # Get paper IDs from FAISS results
        similar_paper_ids = [int(id_map[idx]) for idx in indices[0] if idx < len(id_map)]
        
        # Also get top TF-IDF paper IDs
        tfidf_paper_ids = [pid for pid, score in sorted(
            tfidf_sims.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:topn]]
        
        # Combine and deduplicate
        all_paper_ids = list(set(similar_paper_ids[:topn] + tfidf_paper_ids))
        
        # Get author IDs, excluding query author
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            for paper_id in all_paper_ids:
                cursor.execute("""
                    SELECT author_id FROM papers WHERE id = ?
                """, (paper_id,))
                
                row = cursor.fetchone()
                if row and row[0] and row[0] != query_author_id:
                    author_ids.add(row[0])
        
        logger.debug(f"Found {len(author_ids)} weak positive authors")
        
    except Exception as e:
        logger.error(f"Error getting weak positive authors: {e}")
    
    return author_ids


def get_positive_authors(
    query: Dict,
    db_path: Path,
    tfidf_engine: TFIDFEngine,
    faiss_index: "faiss.Index",
    id_map: np.ndarray,
    emb_model: Embeddings,
    n_positives: int
) -> Set[int]:
    """
    Get positive author IDs for a query paper.
    
    Priority:
    1. Authors from reference list (if available)
    2. Authors from co-author neighborhood
    3. Weak positives from TF-IDF/embedding similarity
    
    Excludes:
    - Query author (COI)
    - Authors with conflicts of interest
    
    Args:
        query: Query paper dict
        db_path: Path to database
        tfidf_engine: TFIDFEngine instance
        faiss_index: FAISS index
        id_map: FAISS ID mapping
        emb_model: Embeddings model
        n_positives: Number of positives to return
        
    Returns:
        Set of positive author IDs
    """
    positive_ids = set()
    query_author_id = query['author_id']
    query_author_name = query['author_name']
    
    # Strategy 1: Reference authors
    ref_authors = get_reference_authors(db_path, query['paper_id'])
    positive_ids.update(ref_authors)
    logger.debug(f"Query {query['paper_id']}: {len(ref_authors)} reference authors")
    
    # Strategy 2: Co-author neighborhood
    if len(positive_ids) < n_positives:
        coauthor_ids = get_coauthor_neighborhood(db_path, query_author_name, max_authors=10)
        positive_ids.update(coauthor_ids)
        logger.debug(f"Query {query['paper_id']}: {len(coauthor_ids)} coauthor neighbors")
    
    # Strategy 3: Weak positives (TF-IDF + embedding similarity)
    if len(positive_ids) < n_positives:
        query_text = f"{query['title']} {query['abstract']}"
        weak_positives = get_weak_positive_authors(
            query_text=query_text,
            db_path=db_path,
            tfidf_engine=tfidf_engine,
            faiss_index=faiss_index,
            id_map=id_map,
            emb_model=emb_model,
            query_author_id=query_author_id,
            topn=n_positives * 2
        )
        positive_ids.update(weak_positives)
        logger.debug(f"Query {query['paper_id']}: {len(weak_positives)} weak positives")
    
    # Remove query author (COI)
    positive_ids.discard(query_author_id)
    
    # Check for conflicts of interest
    final_positives = set()
    for author_id in positive_ids:
        try:
            with get_connection(db_path) as conn:
                cursor = conn.cursor()
                
                # Get author name and affiliation
                cursor.execute("""
                    SELECT name, affiliation FROM authors WHERE id = ?
                """, (author_id,))
                
                row = cursor.fetchone()
                if not row:
                    continue
                
                author_name, author_affiliation = row[0], row[1]
                
                # Check for COI
                if not has_conflict(
                    candidate_name=author_name,
                    query_author_names=[query_author_name],
                    query_affiliations=[],  # Could add query author affiliation
                    db_path=db_path,
                    candidate_affiliation=author_affiliation
                ):
                    final_positives.add(author_id)
                else:
                    logger.debug(f"Excluded author {author_id} due to COI")
        
        except Exception as e:
            logger.error(f"Error checking COI for author {author_id}: {e}")
            continue
    
    # Sample if we have too many
    if len(final_positives) > n_positives:
        final_positives = set(random.sample(list(final_positives), n_positives))
    
    logger.info(f"Query {query['paper_id']}: Selected {len(final_positives)} positive authors")
    return final_positives


# ============================================================================
# Negative Label Generation
# ============================================================================

def get_negative_authors(
    db_path: Path,
    query_author_id: int,
    positive_ids: Set[int],
    n_negatives: int,
    seed: int = 42
) -> Set[int]:
    """
    Sample negative author IDs.
    
    Negatives are authors not in positive set, sampled uniformly.
    Excludes query author.
    
    Args:
        db_path: Path to database
        query_author_id: Query author ID (to exclude)
        positive_ids: Set of positive author IDs (to exclude)
        n_negatives: Number of negatives to sample
        seed: Random seed
        
    Returns:
        Set of negative author IDs
    """
    random.seed(seed)
    negative_ids = set()
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all author IDs
            cursor.execute("SELECT id FROM authors")
            all_author_ids = {row[0] for row in cursor.fetchall()}
            
            # Remove query author and positives
            candidate_negatives = all_author_ids - positive_ids - {query_author_id}
            
            if len(candidate_negatives) < n_negatives:
                logger.warning(f"Only {len(candidate_negatives)} candidate negatives available")
                negative_ids = candidate_negatives
            else:
                negative_ids = set(random.sample(list(candidate_negatives), n_negatives))
            
            logger.debug(f"Sampled {len(negative_ids)} negative authors")
            
    except Exception as e:
        logger.error(f"Error sampling negative authors: {e}")
    
    return negative_ids


# ============================================================================
# Training Data Generation
# ============================================================================

def build_training_data(
    db_path: Path,
    tfidf_path: Path,
    faiss_index_path: Path,
    id_map_path: Path,
    output_path: Path,
    n_queries: int = 100,
    n_positives: int = 3,
    n_negatives: int = 20,
    min_year: Optional[int] = None,
    seed: int = 42
) -> bool:
    """
    Build training dataset for learning-to-rank model.
    
    Args:
        db_path: Path to SQLite database
        tfidf_path: Path to TF-IDF model
        faiss_index_path: Path to FAISS index
        id_map_path: Path to ID mapping
        output_path: Path to output parquet file
        n_queries: Number of queries to sample
        n_positives: Number of positive labels per query
        n_negatives: Number of negative labels per query
        min_year: Minimum publication year for queries
        seed: Random seed
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Building Training Data for Learning-to-Rank Model")
    logger.info("=" * 80)
    logger.info(f"Queries: {n_queries}")
    logger.info(f"Positives per query: {n_positives}")
    logger.info(f"Negatives per query: {n_negatives}")
    logger.info(f"Total samples: {n_queries * (n_positives + n_negatives)}")
    logger.info("")
    
    # -------------------------------------------------------------------------
    # Step 1: Load models
    # -------------------------------------------------------------------------
    logger.info("Step 1: Loading models and indices...")
    
    try:
        # Load TF-IDF engine
        logger.info(f"  Loading TF-IDF model from {tfidf_path}...")
        tfidf_engine = TFIDFEngine.load(str(tfidf_path))
        logger.info(f"  ✓ TF-IDF engine loaded: {len(tfidf_engine.paper_ids)} papers")
        
        # Load FAISS index
        logger.info(f"  Loading FAISS index from {faiss_index_path}...")
        faiss_index = faiss.read_index(str(faiss_index_path))
        id_map = np.load(str(id_map_path))
        logger.info(f"  ✓ FAISS index loaded: {faiss_index.ntotal} vectors")
        
        # Load embedding model
        logger.info(f"  Loading SciBERT model...")
        emb_model = Embeddings()
        logger.info(f"  ✓ Embedding model loaded: {emb_model.model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False
    
    # -------------------------------------------------------------------------
    # Step 2: Sample query papers
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Sampling query papers...")
    
    queries = sample_query_papers(
        db_path=db_path,
        n_queries=n_queries,
        min_year=min_year,
        seed=seed
    )
    
    if not queries:
        logger.error("No query papers sampled")
        return False
    
    logger.info(f"  ✓ Sampled {len(queries)} query papers")
    
    # -------------------------------------------------------------------------
    # Step 3: Generate labels and features
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Generating labels and features...")
    
    training_rows = []
    
    for group_id, query in enumerate(queries):
        logger.info(f"  Processing query {group_id + 1}/{len(queries)}: "
                   f"Paper {query['paper_id']} by {query['author_name']}")
        
        # Get positive authors
        positive_ids = get_positive_authors(
            query=query,
            db_path=db_path,
            tfidf_engine=tfidf_engine,
            faiss_index=faiss_index,
            id_map=id_map,
            emb_model=emb_model,
            n_positives=n_positives
        )
        
        # Get negative authors
        negative_ids = get_negative_authors(
            db_path=db_path,
            query_author_id=query['author_id'],
            positive_ids=positive_ids,
            n_negatives=n_negatives,
            seed=seed + group_id
        )
        
        # Combine positive and negative
        all_candidate_ids = list(positive_ids) + list(negative_ids)
        labels = [1] * len(positive_ids) + [0] * len(negative_ids)
        
        if len(all_candidate_ids) == 0:
            logger.warning(f"  Query {query['paper_id']}: No candidates found, skipping")
            continue
        
        logger.info(f"    Positives: {len(positive_ids)}, Negatives: {len(negative_ids)}")
        
        # Generate features using ranker
        query_text = f"{query['title']} {query['abstract']}"
        
        try:
            features_df = make_features_for_query(
                query_text=query_text,
                db=str(db_path),
                tfidf_engine=tfidf_engine,
                faiss_index=faiss_index,
                id_map=id_map,
                embedding_model=emb_model,
                topn_papers=100
            )
            
            # Filter to candidate authors
            features_df = features_df[features_df['author_id'].isin(all_candidate_ids)]
            
            # Add query_id, label, and group columns
            for candidate_id, label in zip(all_candidate_ids, labels):
                # Find features for this author
                author_features = features_df[features_df['author_id'] == candidate_id]
                
                if len(author_features) == 0:
                    # Author not in features (no relevant papers), use zeros
                    row = {
                        'query_id': query['paper_id'],
                        'author_id': candidate_id,
                        'y': label,
                        'group': group_id,
                        'tfidf_max': 0.0,
                        'tfidf_mean': 0.0,
                        'emb_max': 0.0,
                        'emb_mean': 0.0,
                        'topic_overlap': 0.0,
                        'recency_mean': 0.0,
                        'recency_max': 0.0,
                        'pub_count': 0,
                        'coi_flag': 0
                    }
                else:
                    # Use computed features
                    author_row = author_features.iloc[0]
                    row = {
                        'query_id': query['paper_id'],
                        'author_id': candidate_id,
                        'y': label,
                        'group': group_id,
                        'tfidf_max': author_row['tfidf_max'],
                        'tfidf_mean': author_row['tfidf_mean'],
                        'emb_max': author_row['emb_max'],
                        'emb_mean': author_row['emb_mean'],
                        'topic_overlap': author_row['topic_overlap'],
                        'recency_mean': author_row['recency_mean'],
                        'recency_max': author_row['recency_max'],
                        'pub_count': author_row['pub_count'],
                        'coi_flag': author_row['coi_flag']
                    }
                
                training_rows.append(row)
        
        except Exception as e:
            logger.error(f"  Error generating features for query {query['paper_id']}: {e}")
            continue
    
    if not training_rows:
        logger.error("No training rows generated")
        return False
    
    # -------------------------------------------------------------------------
    # Step 4: Create DataFrame and save
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 4: Creating DataFrame and saving...")
    
    df = pd.DataFrame(training_rows)
    
    # Reorder columns
    column_order = [
        'query_id', 'author_id', 'y', 'group',
        'tfidf_max', 'tfidf_mean', 'emb_max', 'emb_mean',
        'topic_overlap', 'recency_mean', 'recency_max',
        'pub_count', 'coi_flag'
    ]
    df = df[column_order]
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"  ✓ Saved training data to {output_path}")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Total queries: {df['query_id'].nunique()}")
    logger.info(f"  Total groups: {df['group'].nunique()}")
    logger.info(f"  Positive samples: {df['y'].sum()}")
    logger.info(f"  Negative samples: {(df['y'] == 0).sum()}")
    logger.info(f"  Positivity rate: {df['y'].mean():.2%}")
    
    # Verify grouping
    logger.info("")
    logger.info("Verifying grouping...")
    group_counts = df.groupby('group').size()
    logger.info(f"  Group sizes: min={group_counts.min()}, "
               f"max={group_counts.max()}, mean={group_counts.mean():.1f}")
    
    # Verify one group per query
    query_groups = df.groupby('query_id')['group'].nunique()
    assert (query_groups == 1).all(), "Each query should have exactly one group"
    logger.info(f"  ✓ Verified: One group per query")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Data Generation Complete!")
    logger.info("=" * 80)
    
    return True


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Build training data for learning-to-rank model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 queries with default settings
  python build_training_data.py
  
  # Generate 200 queries with more positives/negatives
  python build_training_data.py --queries 200 --positives 5 --negatives 30
  
  # Use recent papers only (2020+)
  python build_training_data.py --min-year 2020
  
  # Custom output path
  python build_training_data.py --out data/train_custom.parquet
        """
    )
    
    parser.add_argument(
        '--db',
        type=str,
        default='data/papers.db',
        help='Path to SQLite database (default: data/papers.db)'
    )
    
    parser.add_argument(
        '--tfidf',
        type=str,
        default='models/tfidf_vectorizer.pkl',
        help='Path to TF-IDF model (default: models/tfidf_vectorizer.pkl)'
    )
    
    parser.add_argument(
        '--faiss-index',
        type=str,
        default='data/faiss_index.faiss',
        help='Path to FAISS index (default: data/faiss_index.faiss)'
    )
    
    parser.add_argument(
        '--id-map',
        type=str,
        default='data/id_map.npy',
        help='Path to ID mapping (default: data/id_map.npy)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='data/train.parquet',
        help='Path to output parquet file (default: data/train.parquet)'
    )
    
    parser.add_argument(
        '--queries',
        type=int,
        default=100,
        help='Number of query papers to sample (default: 100)'
    )
    
    parser.add_argument(
        '--positives',
        type=int,
        default=3,
        help='Number of positive labels per query (default: 3)'
    )
    
    parser.add_argument(
        '--negatives',
        type=int,
        default=20,
        help='Number of negative labels per query (default: 20)'
    )
    
    parser.add_argument(
        '--min-year',
        type=int,
        default=None,
        help='Minimum publication year for query papers (optional)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    db_path = Path(args.db)
    tfidf_path = Path(args.tfidf)
    faiss_index_path = Path(args.faiss_index)
    id_map_path = Path(args.id_map)
    output_path = Path(args.out)
    
    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run ingest.py first to create database")
        sys.exit(1)
    
    if not tfidf_path.exists():
        logger.error(f"TF-IDF model not found: {tfidf_path}")
        logger.error("Run build_tfidf.py first")
        sys.exit(1)
    
    if not faiss_index_path.exists():
        logger.error(f"FAISS index not found: {faiss_index_path}")
        logger.error("Run build_vectors.py first")
        sys.exit(1)
    
    if not id_map_path.exists():
        logger.error(f"ID mapping not found: {id_map_path}")
        logger.error("Run build_vectors.py first")
        sys.exit(1)
    
    # Build training data
    success = build_training_data(
        db_path=db_path,
        tfidf_path=tfidf_path,
        faiss_index_path=faiss_index_path,
        id_map_path=id_map_path,
        output_path=output_path,
        n_queries=args.queries,
        n_positives=args.positives,
        n_negatives=args.negatives,
        min_year=args.min_year,
        seed=args.seed
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

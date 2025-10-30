#!/usr/bin/env python3
"""
Build TF-IDF Model for Keyword Search

This script trains a TF-IDF vectorizer on all papers in the database,
creates a sparse matrix representation, and saves the model for fast
keyword-based similarity search.

Usage:
    python build_tfidf.py
    python build_tfidf.py --db data/papers.db --out models/tfidf_vectorizer.pkl
    python build_tfidf.py --min-df 2 --max-df 0.85 --max-features 50000

Example:
    # Default settings
    python build_tfidf.py
    
    # Custom document frequency filters
    python build_tfidf.py --min-df 3 --max-df 0.9
    
    # Custom vocabulary size
    python build_tfidf.py --max-features 30000
    
Output Files:
    1. TF-IDF vectorizer pickle (--out argument)
       Contains: vectorizer, corpus_matrix, paper_ids mapping
    2. Log file: build_tfidf.log
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from db_utils import get_connection, get_all_papers
    from tfidf_engine import TFIDFEngine
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure db_utils.py and tfidf_engine.py are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('build_tfidf.log')
    ]
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text for TF-IDF vectorization.
    
    Removes excessive whitespace and normalizes text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove excessive punctuation
    text = text.replace('..', '.')
    text = text.replace('  ', ' ')
    
    return text.strip()


def extract_paper_text(paper: Dict, max_chars: int = 2000) -> str:
    """
    Extract text from paper for TF-IDF.
    
    Priority:
    1. Abstract (if exists)
    2. First 2k chars of fulltext (if exists)
    3. Title only (fallback)
    
    Args:
        paper: Paper dictionary from database
        max_chars: Maximum characters to extract from fulltext
        
    Returns:
        Cleaned text string
    """
    # Try abstract first (preferred)
    abstract = paper.get('abstract') or ''
    if abstract and len(abstract.strip()) > 50:
        return clean_text(abstract)
    
    # Try fulltext
    fulltext = paper.get('fulltext') or ''
    if fulltext and len(fulltext.strip()) > 50:
        # Take first max_chars characters
        truncated = fulltext[:max_chars]
        return clean_text(truncated)
    
    # Fallback to title
    title = paper.get('title') or ''
    if title and title.strip():
        return clean_text(title)
    
    # Last resort: empty string
    return ""


def get_top_idf_terms(engine: TFIDFEngine, topn: int = 30) -> List[Tuple[str, float]]:
    """
    Get top terms by IDF score.
    
    Higher IDF = more discriminative (appears in fewer documents).
    
    Args:
        engine: Fitted TFIDFEngine
        topn: Number of top terms to return
        
    Returns:
        List of (term, idf) tuples sorted by IDF descending
    """
    if engine.vectorizer is None:
        return []
    
    # Get IDF values
    idf_values = engine.vectorizer.idf_
    
    # Get vocabulary (term -> index mapping)
    vocabulary = engine.vectorizer.vocabulary_
    
    # Create list of (term, idf)
    term_idf = [(term, idf_values[idx]) for term, idx in vocabulary.items()]
    
    # Sort by IDF descending
    term_idf.sort(key=lambda x: x[1], reverse=True)
    
    return term_idf[:topn]


def build_tfidf_from_db(
    db_path: str,
    min_df: int,
    max_df: float,
    max_features: int,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Tuple[TFIDFEngine, int]:
    """
    Build TF-IDF model from database papers.
    
    Args:
        db_path: Path to SQLite database
        min_df: Minimum document frequency
        max_df: Maximum document frequency (fraction)
        max_features: Maximum vocabulary size
        ngram_range: N-gram range
        
    Returns:
        Tuple of (fitted_engine, skipped_count)
    """
    logger.info("=" * 80)
    logger.info("Building TF-IDF model from database")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Min DF: {min_df}")
    logger.info(f"Max DF: {max_df}")
    logger.info(f"Max features: {max_features}")
    logger.info(f"N-gram range: {ngram_range}")
    logger.info("")
    
    # Load papers
    logger.info("Loading papers from database...")
    papers = get_all_papers(db_path)
    logger.info(f"Found {len(papers)} papers in database")
    
    if not papers:
        raise ValueError("No papers found in database. Run ingest.py first.")
    
    # Extract texts
    logger.info("")
    logger.info("Extracting texts from papers...")
    corpus_texts = []
    paper_ids = []
    skipped_count = 0
    skipped_reasons = {
        'no_text': 0,
        'too_short': 0
    }
    
    for paper in papers:
        paper_id = paper['id']
        text = extract_paper_text(paper)
        
        if not text:
            skipped_count += 1
            skipped_reasons['no_text'] += 1
            logger.debug(f"Skipping paper {paper_id}: No text available")
            continue
        
        if len(text) < 10:
            skipped_count += 1
            skipped_reasons['too_short'] += 1
            logger.debug(f"Skipping paper {paper_id}: Text too short ({len(text)} chars)")
            continue
        
        corpus_texts.append(text)
        paper_ids.append(paper_id)
    
    logger.info(f"Extracted {len(corpus_texts)} valid texts")
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} papers:")
        logger.warning(f"  - No text: {skipped_reasons['no_text']}")
        logger.warning(f"  - Too short: {skipped_reasons['too_short']}")
    
    if not corpus_texts:
        raise ValueError(
            "No valid texts extracted from papers. "
            "Papers may be missing abstracts and fulltext."
        )
    
    # Initialize TF-IDF engine
    logger.info("")
    logger.info("Initializing TF-IDF engine...")
    engine = TFIDFEngine(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    
    # Fit on corpus
    logger.info("")
    logger.info("Fitting TF-IDF vectorizer...")
    engine.fit(corpus_texts, paper_ids)
    
    logger.info("")
    logger.info("✓ TF-IDF model built successfully")
    
    return engine, skipped_count


def main():
    """Main entry point for build_tfidf CLI."""
    parser = argparse.ArgumentParser(
        description="Build TF-IDF model for keyword search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings
  python build_tfidf.py
  
  # Custom database and output paths
  python build_tfidf.py --db data/papers.db --out models/tfidf_vectorizer.pkl
  
  # Custom document frequency filters
  python build_tfidf.py --min-df 3 --max-df 0.9
  
  # Custom vocabulary size
  python build_tfidf.py --max-features 30000
  
  # All custom parameters
  python build_tfidf.py --db data/papers.db --out models/custom_tfidf.pkl \\
                        --min-df 2 --max-df 0.85 --max-features 50000

Document Frequency Filters:
  --min-df: Ignore terms appearing in fewer than N documents
            Higher = fewer rare terms, smaller vocabulary
            Default: 2 (appears in at least 2 documents)
  
  --max-df: Ignore terms appearing in more than X% of documents
            Lower = fewer common terms (like stopwords)
            Default: 0.85 (appears in at most 85% of documents)
        """
    )
    
    # Arguments
    parser.add_argument(
        '--db',
        type=str,
        default='data/papers.db',
        help='Path to SQLite database (default: data/papers.db)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='models/tfidf_vectorizer.pkl',
        help='Path to save TF-IDF model pickle (default: models/tfidf_vectorizer.pkl)'
    )
    
    parser.add_argument(
        '--min-df',
        type=int,
        default=2,
        help='Minimum document frequency - ignore terms appearing in fewer documents (default: 2)'
    )
    
    parser.add_argument(
        '--max-df',
        type=float,
        default=0.85,
        help='Maximum document frequency - ignore terms appearing in more than this fraction of documents (default: 0.85)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=50000,
        help='Maximum vocabulary size (default: 50000)'
    )
    
    parser.add_argument(
        '--ngram-min',
        type=int,
        default=1,
        help='Minimum n-gram size (default: 1)'
    )
    
    parser.add_argument(
        '--ngram-max',
        type=int,
        default=2,
        help='Maximum n-gram size (default: 2)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if output file exists'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.min_df < 1:
        logger.error("--min-df must be >= 1")
        sys.exit(1)
    
    if not (0.0 < args.max_df <= 1.0):
        logger.error("--max-df must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.max_features < 1:
        logger.error("--max-features must be >= 1")
        sys.exit(1)
    
    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        logger.error("Invalid n-gram range")
        sys.exit(1)
    
    # Print header
    logger.info("")
    logger.info("=" * 80)
    logger.info("Build TF-IDF Model")
    logger.info("=" * 80)
    logger.info(f"Database:     {args.db}")
    logger.info(f"Output file:  {args.out}")
    logger.info(f"Min DF:       {args.min_df}")
    logger.info(f"Max DF:       {args.max_df}")
    logger.info(f"Max features: {args.max_features}")
    logger.info(f"N-gram range: ({args.ngram_min}, {args.ngram_max})")
    logger.info(f"Force:        {args.force}")
    logger.info("=" * 80)
    logger.info("")
    
    # Check if database exists
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run ingest.py first to create database")
        sys.exit(1)
    
    # Check if output already exists
    out_path = Path(args.out)
    
    if not args.force and out_path.exists():
        logger.warning(f"Output file already exists: {out_path}")
        logger.warning("Use --force to rebuild")
        
        response = input("Rebuild anyway? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Aborted by user")
            sys.exit(0)
    
    try:
        # Step 1: Build TF-IDF model
        logger.info("Step 1: Building TF-IDF model from database...")
        logger.info("")
        
        engine, skipped_count = build_tfidf_from_db(
            db_path=str(db_path),
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max)
        )
        
        # Step 2: Get top IDF terms
        logger.info("")
        logger.info("Step 2: Analyzing vocabulary...")
        logger.info("")
        
        top_idf_terms = get_top_idf_terms(engine, topn=30)
        
        logger.info("Top 30 terms by IDF (most discriminative):")
        for i, (term, idf) in enumerate(top_idf_terms, 1):
            logger.info(f"  {i:2d}. {term:30s} (IDF: {idf:.4f})")
        
        # Step 3: Save model
        logger.info("")
        logger.info("Step 3: Saving TF-IDF model...")
        logger.info("")
        
        # Create output directory
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using the engine's save method
        logger.info(f"Saving model to: {out_path}")
        try:
            engine.save(str(out_path))
            logger.info(f"✓ TF-IDF model saved")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("BUILD COMPLETE ✓")
        logger.info("=" * 80)
        logger.info(f"Documents processed: {engine.corpus_matrix.shape[0]}")
        logger.info(f"Documents skipped:   {skipped_count}")
        logger.info(f"Vocabulary size:     {len(engine.vectorizer.vocabulary_)}")
        logger.info(f"Feature count:       {engine.corpus_matrix.shape[1]}")
        logger.info(f"Matrix sparsity:     {1.0 - (engine.corpus_matrix.nnz / (engine.corpus_matrix.shape[0] * engine.corpus_matrix.shape[1])):.4f}")
        logger.info(f"Non-zero elements:   {engine.corpus_matrix.nnz:,}")
        logger.info("")
        logger.info("Parameters:")
        logger.info(f"  Min DF:       {args.min_df}")
        logger.info(f"  Max DF:       {args.max_df}")
        logger.info(f"  Max features: {args.max_features}")
        logger.info(f"  N-gram range: ({args.ngram_min}, {args.ngram_max})")
        logger.info("")
        logger.info("Output file:")
        logger.info(f"  {out_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Test search with: python -c \"from tfidf_engine import TFIDFEngine; "
                   f"engine = TFIDFEngine.load('{out_path}'); print(f'Loaded {{engine.corpus_matrix.shape[0]}} documents')\"")
        logger.info("  2. Use in search API with engine.most_similar(query_text)")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("")
        logger.warning("Interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("BUILD FAILED ✗")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Check build_tfidf.log for details")
        logger.error("=" * 80)
        raise


if __name__ == '__main__':
    main()

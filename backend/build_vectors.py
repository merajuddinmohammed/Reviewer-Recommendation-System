#!/usr/bin/env python3
"""
Build Embeddings and FAISS Index for Semantic Search

This script generates dense embeddings for all papers in the database and
builds a FAISS index for fast similarity search. It extracts text from
abstracts (preferred) or full text, encodes them using SciBERT/SPECTER,
and saves the FAISS index with ID mappings.

Usage:
    python build_vectors.py
    python build_vectors.py --db data/papers.db --index data/faiss_index.faiss
    python build_vectors.py --model allenai/specter --batch-size 16

Environment Variables:
    EMB_BATCH: Batch size for encoding (default: 8)
    EMB_MODEL: Model name (default: allenai/scibert_scivocab_uncased)

Example:
    # Default settings
    python build_vectors.py
    
    # Custom batch size
    EMB_BATCH=4 python build_vectors.py
    
    # Custom model
    python build_vectors.py --model allenai/specter
    
Output Files:
    1. FAISS index file (--index argument)
    2. ID mapping file (same path with _id_map.npy suffix)
    3. Model info file (backend/models/bert_model/used_model.txt)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from db_utils import get_connection, get_all_papers
    from embedding import Embeddings, build_faiss_index, save_index, load_index
    import config  # Centralized configuration
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure db_utils.py and embedding.py are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('build_vectors.log')
    ]
)
logger = logging.getLogger(__name__)


def clean_text(text: Optional[str]) -> str:
    """
    Clean text for embedding encoding.
    
    Removes excessive whitespace, normalizes newlines, and ensures
    the text is suitable for encoding.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string (empty string if input is None)
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
    Extract text from paper for embedding.
    
    Priority:
    1. Abstract (if exists)
    2. First 2k chars of fulltext (if exists)
    3. Title only (fallback)
    
    Args:
        paper: Paper dictionary from database
        max_chars: Maximum characters to extract from fulltext
        
    Returns:
        Cleaned text string suitable for embedding
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
    
    # Last resort: empty string (will be logged and skipped)
    return ""


def build_vectors_from_db(
    db_path: str,
    model_name: str,
    batch_size: int,
    device: Optional[str] = None
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build embeddings for all papers in database.
    
    Args:
        db_path: Path to SQLite database
        model_name: Hugging Face model identifier
        batch_size: Batch size for encoding
        device: Device for computation (None = auto-detect)
        
    Returns:
        Tuple of:
        - embeddings: numpy array of shape (n_valid_papers, dim)
        - paper_ids: list of paper IDs corresponding to embeddings
        - skipped_ids: list of paper IDs that were skipped
    """
    logger.info("=" * 80)
    logger.info("Building vectors from database")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device or 'auto-detect'}")
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
    texts = []
    paper_ids = []
    skipped_ids = []
    skipped_reasons = {
        'no_text': 0,
        'too_short': 0,
        'encoding_error': 0
    }
    
    for paper in papers:
        paper_id = paper['id']
        text = extract_paper_text(paper)
        
        if not text:
            skipped_ids.append(paper_id)
            skipped_reasons['no_text'] += 1
            logger.debug(f"Skipping paper {paper_id}: No text available")
            continue
        
        if len(text) < 10:
            skipped_ids.append(paper_id)
            skipped_reasons['too_short'] += 1
            logger.debug(f"Skipping paper {paper_id}: Text too short ({len(text)} chars)")
            continue
        
        texts.append(text)
        paper_ids.append(paper_id)
    
    logger.info(f"Extracted {len(texts)} valid texts")
    if skipped_ids:
        logger.warning(f"Skipped {len(skipped_ids)} papers:")
        logger.warning(f"  - No text: {skipped_reasons['no_text']}")
        logger.warning(f"  - Too short: {skipped_reasons['too_short']}")
    
    if not texts:
        raise ValueError(
            "No valid texts extracted from papers. "
            "Papers may be missing abstracts and fulltext."
        )
    
    # Initialize embeddings model
    logger.info("")
    logger.info("Initializing embeddings model...")
    emb = Embeddings(
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )
    logger.info(f"Model loaded: {emb}")
    logger.info(f"Embedding dimension: {emb.dim}")
    
    # Encode texts
    logger.info("")
    logger.info("Encoding texts...")
    try:
        embeddings = emb.encode_texts(
            texts,
            show_progress_bar=True,
            normalize=True  # L2-normalize for cosine similarity
        )
    except Exception as e:
        logger.error(f"Failed to encode texts: {e}")
        raise
    
    logger.info("")
    logger.info("✓ Encoding complete")
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    logger.info(f"  Embeddings dtype: {embeddings.dtype}")
    logger.info(f"  Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    return embeddings, paper_ids, skipped_ids


def save_model_info(model_name: str, output_dir: Path) -> None:
    """
    Save model information to used_model.txt.
    
    Args:
        model_name: Hugging Face model identifier
        output_dir: Directory to save model info (e.g., backend/models/bert_model)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "used_model.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{model_name}\n")
        logger.info(f"✓ Saved model info to: {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save model info: {e}")


def main():
    """Main entry point for build_vectors CLI."""
    parser = argparse.ArgumentParser(
        description="Build embeddings and FAISS index for semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings
  python build_vectors.py
  
  # Custom database and index paths
  python build_vectors.py --db data/papers.db --index data/faiss_index
  
  # Custom model and batch size
  python build_vectors.py --model allenai/specter --batch-size 16
  
  # Custom ID map location
  python build_vectors.py --idmap data/paper_ids.npy
  
  # With custom batch size from environment
  EMB_BATCH=4 python build_vectors.py

Environment Variables:
  EMB_BATCH   Batch size for encoding (default: 8)
  EMB_MODEL   Model name (overrides --model)
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
        '--index',
        type=str,
        default='data/faiss_index.faiss',
        help='Path to save FAISS index (default: data/faiss_index.faiss)'
    )
    
    parser.add_argument(
        '--idmap',
        type=str,
        default='data/id_map.npy',
        help='Path to save ID mapping (default: data/id_map.npy)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=config.EMBEDDING_MODEL,
        help=f'Hugging Face model name (default: {config.EMBEDDING_MODEL}, configurable via config.py)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.EMB_BATCH,
        help=f'Batch size for encoding (default: {config.EMB_BATCH}, configurable via config.py or EMB_BATCH env var)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for computation (cuda/cpu, default: auto-detect)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if index exists'
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
    
    # Print header
    logger.info("")
    logger.info("=" * 80)
    logger.info("Build Embeddings and FAISS Index")
    logger.info("=" * 80)
    logger.info(f"Database:     {args.db}")
    logger.info(f"Index file:   {args.index}")
    logger.info(f"ID map:       {args.idmap}")
    logger.info(f"Model:        {args.model}")
    logger.info(f"Batch size:   {args.batch_size}")
    logger.info(f"Device:       {args.device or 'auto-detect'}")
    logger.info(f"Force:        {args.force}")
    logger.info("=" * 80)
    logger.info("")
    
    # Check if database exists
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run ingest.py first to create database")
        sys.exit(1)
    
    # Check if index already exists
    index_path = Path(args.index)
    idmap_path = Path(args.idmap)
    
    if not args.force:
        if index_path.exists() and idmap_path.exists():
            logger.warning(f"Index already exists: {index_path}")
            logger.warning(f"ID map already exists: {idmap_path}")
            logger.warning("Use --force to rebuild")
            
            response = input("Rebuild anyway? (y/N): ").strip().lower()
            if response != 'y':
                logger.info("Aborted by user")
                sys.exit(0)
    
    try:
        # Step 1: Build embeddings from database
        logger.info("Step 1: Building embeddings from database...")
        logger.info("")
        
        embeddings, paper_ids, skipped_ids = build_vectors_from_db(
            db_path=str(db_path),
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Step 2: Build FAISS index
        logger.info("")
        logger.info("Step 2: Building FAISS index...")
        logger.info("")
        
        dim = embeddings.shape[1]
        index = build_faiss_index(
            embeddings=embeddings,
            dim=dim,
            metric="ip"  # Inner product (cosine similarity with normalized embeddings)
        )
        
        # Step 3: Save index and ID mapping
        logger.info("")
        logger.info("Step 3: Saving index and ID mapping...")
        logger.info("")
        
        # Create output directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        idmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        logger.info(f"Saving FAISS index to: {index_path}")
        try:
            import faiss
            faiss.write_index(index, str(index_path))
            logger.info(f"✓ FAISS index saved")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
        
        # Save ID mapping
        logger.info(f"Saving ID mapping to: {idmap_path}")
        try:
            np.save(str(idmap_path), np.array(paper_ids, dtype=np.int32))
            logger.info(f"✓ ID mapping saved")
        except Exception as e:
            logger.error(f"Failed to save ID mapping: {e}")
            raise
        
        # Step 4: Save model info
        logger.info("")
        logger.info("Step 4: Saving model information...")
        logger.info("")
        
        model_dir = Path('models/bert_model')
        save_model_info(args.model, model_dir)
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("BUILD COMPLETE ✓")
        logger.info("=" * 80)
        logger.info(f"Vector count:     {index.ntotal}")
        logger.info(f"Dimension:        {index.d}")
        logger.info(f"Model:            {args.model}")
        logger.info(f"Papers processed: {len(paper_ids)}")
        logger.info(f"Papers skipped:   {len(skipped_ids)}")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - FAISS index:  {index_path}")
        logger.info(f"  - ID mapping:   {idmap_path}")
        logger.info(f"  - Model info:   {model_dir / 'used_model.txt'}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Test search with: python -c \"from embedding import load_index; "
                   "idx, ids = load_index('data/faiss_index'); print(f'Loaded {idx.ntotal} vectors')\"")
        logger.info("  2. Use in search API with semantic_search() function")
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
        logger.error("Check build_vectors.log for details")
        logger.error("=" * 80)
        raise


if __name__ == '__main__':
    main()

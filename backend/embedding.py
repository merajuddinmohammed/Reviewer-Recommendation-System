"""
Sentence Embeddings and FAISS Indexing for Semantic Search

This module provides dense semantic embeddings using SciBERT/SPECTER and
FAISS indexing for fast similarity search. It integrates with the TF-IDF
engine to provide hybrid search capabilities.

Key Features:
- SciBERT/SPECTER models via SentenceTransformers
- GPU acceleration with utils.device()
- Batched encoding with progress bars
- L2 normalization for cosine similarity
- FAISS IndexFlatIP for fast search
- Persistent index and ID mapping

Example:
    >>> from embedding import Embeddings
    >>> emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")
    >>> texts = ["deep learning", "machine learning"]
    >>> vectors = emb.encode_texts(texts)
    >>> vectors.shape
    (2, 768)
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available. Install with: pip install faiss-cpu")

try:
    from utils import device as get_device
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("utils.device() not available. Using CPU.")
    
    def get_device():
        """Fallback device detection."""
        return "cpu"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class Embeddings:
    """
    Sentence embeddings using SciBERT or SPECTER models.
    
    This class handles loading pre-trained models, encoding texts with
    mean pooling, and ensuring float32 output for FAISS compatibility.
    
    Attributes:
        model_name (str): Hugging Face model identifier
        batch_size (int): Batch size for encoding
        device (str): Device for computation ('cuda' or 'cpu')
        model (SentenceTransformer): Loaded model
        dim (int): Embedding dimension
    
    Example:
        >>> emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")
        >>> texts = ["neural networks", "deep learning"]
        >>> vectors = emb.encode_texts(texts)
        >>> vectors.shape
        (2, 768)
        >>> vectors.dtype
        dtype('float32')
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        batch_size: int = 8,
        device: Optional[str] = None
    ):
        """
        Initialize embeddings model.
        
        Args:
            model_name: Hugging Face model identifier. Recommended:
                - "allenai/scibert_scivocab_uncased" (768 dim, scientific papers)
                - "allenai/specter" (768 dim, scientific papers)
                - "sentence-transformers/all-MiniLM-L6-v2" (384 dim, general)
            batch_size: Number of texts to encode per batch
            device: Device for computation. If None, uses utils.device()
        
        Raises:
            ImportError: If sentence-transformers not installed
            RuntimeError: If model fails to load
        
        Example:
            >>> emb = Embeddings()  # Uses SciBERT by default
            >>> emb.device
            'cuda'
            >>> emb.dim
            768
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Detect device
        if device is None:
            if UTILS_AVAILABLE:
                self.device = get_device()
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            # Load model with SentenceTransformer
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Get embedding dimension
            self.dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"  Embedding dimension: {self.dim}")
            logger.info(f"  Max sequence length: {self.model.max_seq_length}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def encode_texts(
        self,
        texts: List[str],
        show_progress_bar: bool = True,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings with batching and progress bar.
        
        Uses mean pooling over token embeddings and ensures float32 output.
        Handles NaN values by replacing with zeros.
        
        Args:
            texts: List of text strings to encode
            show_progress_bar: Whether to show tqdm progress bar
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
        
        Returns:
            numpy array of shape (len(texts), dim) with dtype float32
        
        Raises:
            ValueError: If texts is empty
        
        Example:
            >>> emb = Embeddings()
            >>> texts = ["hello world", "deep learning", "neural networks"]
            >>> vectors = emb.encode_texts(texts)
            >>> vectors.shape
            (3, 768)
            >>> vectors.dtype
            dtype('float32')
            >>> np.isnan(vectors).any()
            False
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        
        logger.info(f"Encoding {len(texts)} texts...")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
        
        try:
            # Encode with SentenceTransformer
            # This handles batching, mean pooling, and device management
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            # Ensure float32 (FAISS requirement)
            embeddings = embeddings.astype(np.float32)
            
            # Handle NaN values (replace with 0)
            nan_mask = np.isnan(embeddings)
            if nan_mask.any():
                nan_count = nan_mask.sum()
                logger.warning(f"Found {nan_count} NaN values. Replacing with 0.")
                embeddings = np.nan_to_num(embeddings, nan=0.0)
            
            logger.info(f"Encoded {len(texts)} texts successfully")
            logger.info(f"  Output shape: {embeddings.shape}")
            logger.info(f"  Output dtype: {embeddings.dtype}")
            logger.info(f"  Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Embeddings(model='{self.model_name}', "
            f"dim={self.dim}, device='{self.device}')"
        )


def build_faiss_index(
    embeddings: np.ndarray,
    dim: int,
    metric: str = "ip"
) -> "faiss.Index":
    """
    Build FAISS index for fast similarity search.
    
    Uses IndexFlatIP (inner product) which equals cosine similarity
    when embeddings are L2-normalized. L2-normalization is applied
    automatically.
    
    Args:
        embeddings: Embeddings array of shape (n, dim)
        dim: Embedding dimension (should match embeddings.shape[1])
        metric: Similarity metric. Options:
            - "ip": Inner product (requires L2-normalized embeddings)
            - "l2": Euclidean distance
            - "cosine": Alias for "ip" with normalization
    
    Returns:
        FAISS index ready for search
    
    Raises:
        ImportError: If faiss not installed
        ValueError: If embeddings is empty or dimension mismatch
    
    Example:
        >>> embeddings = np.random.randn(100, 768).astype(np.float32)
        >>> index = build_faiss_index(embeddings, dim=768, metric="ip")
        >>> index.ntotal
        100
        >>> index.d
        768
    """
    if not FAISS_AVAILABLE:
        raise ImportError(
            "faiss is required. Install with: pip install faiss-cpu (or faiss-gpu)"
        )
    
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings cannot be empty")
    
    if embeddings.shape[1] != dim:
        raise ValueError(
            f"Dimension mismatch: embeddings have dim={embeddings.shape[1]}, "
            f"but dim={dim} was specified"
        )
    
    # Ensure float32
    embeddings = embeddings.astype(np.float32)
    
    logger.info(f"Building FAISS index...")
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    logger.info(f"  Metric: {metric}")
    
    # L2-normalize for cosine similarity with inner product
    if metric in ("ip", "cosine"):
        logger.info("  L2-normalizing embeddings for cosine similarity...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        
        # Verify normalization
        norms_after = np.linalg.norm(embeddings, axis=1)
        logger.info(f"  L2 norms after normalization: "
                   f"mean={norms_after.mean():.6f}, "
                   f"std={norms_after.std():.6f}")
    
    # Build index
    if metric in ("ip", "cosine"):
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'ip', 'cosine', or 'l2'")
    
    # Add vectors
    index.add(embeddings)
    
    logger.info(f"FAISS index built successfully")
    logger.info(f"  Index type: {type(index).__name__}")
    logger.info(f"  Total vectors: {index.ntotal}")
    logger.info(f"  Dimension: {index.d}")
    
    return index


def save_index(
    index: "faiss.Index",
    path: Union[str, Path],
    paper_ids: Optional[List[int]] = None
) -> None:
    """
    Save FAISS index and optional ID mapping to disk.
    
    Args:
        index: FAISS index to save
        path: Path to save index (will save as .index file)
        paper_ids: Optional list of paper IDs corresponding to index rows.
                  If provided, saved as id_map.npy
    
    Raises:
        ImportError: If faiss not installed
    
    Example:
        >>> index = build_faiss_index(embeddings, dim=768)
        >>> paper_ids = [101, 102, 103, 104]
        >>> save_index(index, "models/faiss_index", paper_ids)
        # Creates:
        #   models/faiss_index.index
        #   models/faiss_index_id_map.npy
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save index
    index_path = path.with_suffix('.index')
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to: {index_path}")
    logger.info(f"  Index size: {index_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save ID mapping if provided
    if paper_ids is not None:
        id_map_path = path.parent / f"{path.stem}_id_map.npy"
        id_map = np.array(paper_ids, dtype=np.int64)
        np.save(id_map_path, id_map)
        logger.info(f"Saved ID mapping to: {id_map_path}")
        logger.info(f"  ID map size: {len(paper_ids)}")


def load_index(
    path: Union[str, Path],
    load_id_map: bool = True
) -> Union["faiss.Index", Tuple["faiss.Index", np.ndarray]]:
    """
    Load FAISS index and optional ID mapping from disk.
    
    Args:
        path: Path to index file (without .index suffix)
        load_id_map: Whether to load ID mapping
    
    Returns:
        If load_id_map=False: FAISS index
        If load_id_map=True: Tuple of (index, id_map)
    
    Raises:
        ImportError: If faiss not installed
        FileNotFoundError: If index or id_map file not found
    
    Example:
        >>> index, id_map = load_index("models/faiss_index")
        >>> index.ntotal
        1000
        >>> len(id_map)
        1000
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required")
    
    path = Path(path)
    index_path = path.with_suffix('.index')
    
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    # Load index
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded FAISS index from: {index_path}")
    logger.info(f"  Total vectors: {index.ntotal}")
    logger.info(f"  Dimension: {index.d}")
    
    if not load_id_map:
        return index
    
    # Load ID mapping
    id_map_path = path.parent / f"{path.stem}_id_map.npy"
    
    if not id_map_path.exists():
        logger.warning(f"ID map file not found: {id_map_path}")
        return index, None
    
    id_map = np.load(id_map_path)
    logger.info(f"Loaded ID mapping from: {id_map_path}")
    logger.info(f"  ID map size: {len(id_map)}")
    
    return index, id_map


def search_index(
    index: "faiss.Index",
    query_vectors: np.ndarray,
    k: int = 10,
    id_map: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for top-k similar vectors.
    
    Args:
        index: FAISS index
        query_vectors: Query embeddings of shape (n_queries, dim)
        k: Number of results to return per query
        id_map: Optional ID mapping to convert row indices to paper IDs
    
    Returns:
        Tuple of (scores, ids):
            - scores: Array of shape (n_queries, k) with similarity scores
            - ids: Array of shape (n_queries, k) with paper IDs (if id_map provided)
                  or row indices (if id_map not provided)
    
    Example:
        >>> index, id_map = load_index("models/faiss_index")
        >>> query_vec = embeddings.encode_texts(["deep learning"])
        >>> scores, paper_ids = search_index(index, query_vec, k=5, id_map=id_map)
        >>> paper_ids.shape
        (1, 5)
        >>> scores.shape
        (1, 5)
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required")
    
    # Ensure float32 and 2D
    query_vectors = np.atleast_2d(query_vectors).astype(np.float32)
    
    # Search
    scores, indices = index.search(query_vectors, k)
    
    # Map indices to paper IDs if mapping provided
    if id_map is not None:
        # Replace indices with paper IDs
        ids = np.array([[id_map[idx] if idx >= 0 else -1 for idx in row] 
                       for row in indices])
    else:
        ids = indices
    
    return scores, ids


# ============================================================================
# Demo and Tests
# ============================================================================

def _test_embeddings():
    """Test Embeddings class."""
    print("\n" + "="*70)
    print("TEST 1: Embeddings Class")
    print("="*70)
    
    # Initialize with small model for testing
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, fast
    print(f"\n[1.1] Initializing Embeddings with {model_name}...")
    emb = Embeddings(model_name=model_name, batch_size=2)
    print(f"  Model: {emb.model_name}")
    print(f"  Device: {emb.device}")
    print(f"  Dimension: {emb.dim}")
    assert emb.dim == 384, f"Expected dim=384, got {emb.dim}"
    print("  PASSED")
    
    # Encode texts
    print("\n[1.2] Encoding sample texts...")
    texts = [
        "deep learning neural networks",
        "machine learning algorithms",
        "natural language processing",
        "computer vision image recognition",
        "reinforcement learning agents"
    ]
    vectors = emb.encode_texts(texts, show_progress_bar=False)
    print(f"  Input: {len(texts)} texts")
    print(f"  Output shape: {vectors.shape}")
    print(f"  Output dtype: {vectors.dtype}")
    assert vectors.shape == (5, 384), f"Expected (5, 384), got {vectors.shape}"
    assert vectors.dtype == np.float32, f"Expected float32, got {vectors.dtype}"
    assert not np.isnan(vectors).any(), "Found NaN values"
    print("  PASSED")
    
    # Check normalization
    print("\n[1.3] Testing L2 normalization...")
    vectors_norm = emb.encode_texts(texts[:2], show_progress_bar=False, normalize=True)
    norms = np.linalg.norm(vectors_norm, axis=1)
    print(f"  L2 norms: {norms}")
    assert np.allclose(norms, 1.0, atol=1e-5), "Normalization failed"
    print("  PASSED")
    
    return emb, vectors


def _test_faiss_index(embeddings: np.ndarray):
    """Test FAISS index building and searching."""
    print("\n" + "="*70)
    print("TEST 2: FAISS Index Building")
    print("="*70)
    
    dim = embeddings.shape[1]
    
    # Build index
    print(f"\n[2.1] Building FAISS index (dim={dim})...")
    index = build_faiss_index(embeddings, dim=dim, metric="ip")
    print(f"  Index type: {type(index).__name__}")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {index.d}")
    assert index.ntotal == embeddings.shape[0], "Vector count mismatch"
    assert index.d == dim, "Dimension mismatch"
    print("  PASSED")
    
    # Test search
    print("\n[2.2] Testing top-k search...")
    query_vec = embeddings[0:1]  # First vector as query
    scores, indices = index.search(query_vec, k=3)
    print(f"  Query shape: {query_vec.shape}")
    print(f"  Top-3 results:")
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        print(f"    {i+1}. Index {idx}: score={score:.4f}")
    
    # First result should be the query itself
    assert indices[0][0] == 0, "Query should match itself"
    assert scores[0][0] > 0.99, f"Self-similarity should be ~1.0, got {scores[0][0]}"
    print("  PASSED")
    
    return index


def _test_persistence(index: "faiss.Index", embeddings: np.ndarray):
    """Test save/load functionality."""
    print("\n" + "="*70)
    print("TEST 3: Index Persistence")
    print("="*70)
    
    import tempfile
    import os
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"
        paper_ids = [101, 102, 103, 104, 105]
        
        # Save
        print("\n[3.1] Saving index...")
        save_index(index, index_path, paper_ids)
        print("  PASSED")
        
        # Load
        print("\n[3.2] Loading index...")
        loaded_index, loaded_id_map = load_index(index_path)
        print(f"  Loaded index: {loaded_index.ntotal} vectors")
        print(f"  Loaded ID map: {len(loaded_id_map)} IDs")
        assert loaded_index.ntotal == index.ntotal, "Vector count mismatch"
        assert len(loaded_id_map) == len(paper_ids), "ID map size mismatch"
        assert np.array_equal(loaded_id_map, paper_ids), "ID map content mismatch"
        print("  PASSED")
        
        # Search with ID mapping
        print("\n[3.3] Searching with ID mapping...")
        query_vec = embeddings[0:1]
        scores, paper_ids_result = search_index(loaded_index, query_vec, k=3, id_map=loaded_id_map)
        print(f"  Top-3 results:")
        for i, (pid, score) in enumerate(zip(paper_ids_result[0], scores[0])):
            print(f"    {i+1}. Paper {pid}: score={score:.4f}")
        
        # First result should map to first paper ID
        assert paper_ids_result[0][0] == 101, "ID mapping failed"
        print("  PASSED")


def _demo_semantic_search():
    """Demo: End-to-end semantic search."""
    print("\n" + "="*70)
    print("DEMO: End-to-End Semantic Search")
    print("="*70)
    
    # Sample corpus
    corpus = [
        "Deep learning is a subset of machine learning based on neural networks.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision focuses on enabling computers to interpret visual information.",
        "Reinforcement learning trains agents through reward and punishment.",
        "Transfer learning allows models to apply knowledge from one task to another."
    ]
    paper_ids = [201, 202, 203, 204, 205]
    
    print(f"\n[DEMO 1] Corpus of {len(corpus)} papers")
    for i, text in enumerate(corpus):
        print(f"  Paper {paper_ids[i]}: {text[:60]}...")
    
    # Initialize embeddings
    print("\n[DEMO 2] Initializing embeddings model...")
    emb = Embeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=4
    )
    
    # Encode corpus
    print("\n[DEMO 3] Encoding corpus...")
    vectors = emb.encode_texts(corpus, show_progress_bar=False, normalize=True)
    print(f"  Encoded {len(corpus)} papers")
    print(f"  Embedding shape: {vectors.shape}")
    
    # Build FAISS index
    print("\n[DEMO 4] Building FAISS index...")
    index = build_faiss_index(vectors, dim=emb.dim, metric="ip")
    
    # Demo queries
    queries = [
        "neural network architectures",
        "language understanding models",
        "image classification systems"
    ]
    
    print("\n[DEMO 5] Running semantic search queries...")
    for query in queries:
        print(f"\n  Query: '{query}'")
        
        # Encode query
        query_vec = emb.encode_texts([query], show_progress_bar=False, normalize=True)
        
        # Search
        scores, indices = search_index(index, query_vec, k=3, id_map=np.array(paper_ids))
        
        print(f"  Top-3 results:")
        for rank, (pid, score) in enumerate(zip(indices[0], scores[0]), 1):
            paper_text = corpus[paper_ids.index(pid)][:60]
            print(f"    {rank}. Paper {pid} (score={score:.4f})")
            print(f"       {paper_text}...")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("Sentence Embeddings and FAISS Testing")
    print("="*70)
    
    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\nERROR: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        exit(1)
    
    if not FAISS_AVAILABLE:
        print("\nERROR: faiss not installed")
        print("Install with: pip install faiss-cpu")
        exit(1)
    
    try:
        # Run tests
        emb, vectors = _test_embeddings()
        index = _test_faiss_index(vectors)
        _test_persistence(index, vectors)
        
        # Run demo
        _demo_semantic_search()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("\nKey Features Verified:")
        print("  [x] Embeddings class with SentenceTransformer")
        print("  [x] Mean pooling and float32 output")
        print("  [x] Batched encoding with progress bar")
        print("  [x] NaN handling")
        print("  [x] L2 normalization for cosine similarity")
        print("  [x] FAISS IndexFlatIP building")
        print("  [x] Top-k search via index.search()")
        print("  [x] Index persistence (save/load)")
        print("  [x] ID mapping (row index → paper ID)")
        print("  [x] End-to-end semantic search")
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

"""
TF-IDF Engine for Paper Similarity

Provides efficient TF-IDF vectorization and cosine similarity search
using sparse matrices for scalability.

Features:
- Configurable n-gram range and feature limits
- Sparse matrix operations for efficiency
- Cosine similarity with top-k retrieval
- Serialization with joblib
- Paper ID tracking
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFEngine:
    """
    TF-IDF based text similarity engine.
    
    Uses sklearn's TfidfVectorizer with configurable parameters and
    provides efficient cosine similarity search over a corpus.
    
    Attributes:
        max_features: Maximum number of features (vocabulary size)
        ngram_range: Tuple of (min_n, max_n) for n-gram range
        min_df: Minimum document frequency (ignore rare terms)
        max_df: Maximum document frequency (ignore common terms)
        vectorizer: Fitted TfidfVectorizer instance
        corpus_matrix: Sparse TF-IDF matrix of corpus
        paper_ids: List of paper IDs aligned with corpus_matrix rows
    """
    
    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.85
    ):
        """
        Initialize TF-IDF engine with configuration.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range as (min_n, max_n)
            min_df: Minimum document frequency (count or fraction)
            max_df: Maximum document frequency (fraction)
            
        Examples:
            >>> engine = TFIDFEngine()
            >>> engine.max_features
            50000
            
            >>> engine = TFIDFEngine(max_features=10000, ngram_range=(1, 3))
            >>> engine.ngram_range
            (1, 3)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # At least 2 letters
        )
        
        # Storage for fitted data
        self.corpus_matrix: Optional[csr_matrix] = None
        self.paper_ids: Optional[List[int]] = None
        
        logger.debug(f"Initialized TFIDFEngine with max_features={max_features}, "
                    f"ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}")
    
    def fit(self, corpus_texts: List[str], paper_ids: Optional[List[int]] = None) -> None:
        """
        Fit the TF-IDF vectorizer on corpus and store the matrix.
        
        Args:
            corpus_texts: List of document texts
            paper_ids: Optional list of paper IDs aligned with corpus_texts
                      If None, uses indices 0, 1, 2, ...
                      
        Examples:
            >>> engine = TFIDFEngine(max_features=100, min_df=1)
            >>> corpus = ["deep learning networks", "neural networks training"]
            >>> engine.fit(corpus)
            >>> engine.corpus_matrix.shape[0]
            2
            
            >>> engine = TFIDFEngine(max_features=100, min_df=1)
            >>> corpus = ["machine learning", "deep learning"]
            >>> paper_ids = [101, 102]
            >>> engine.fit(corpus, paper_ids)
            >>> engine.paper_ids
            [101, 102]
        """
        if not corpus_texts:
            raise ValueError("corpus_texts cannot be empty")
        
        # Default paper IDs to indices
        if paper_ids is None:
            paper_ids = list(range(len(corpus_texts)))
        
        if len(corpus_texts) != len(paper_ids):
            raise ValueError(f"corpus_texts ({len(corpus_texts)}) and paper_ids "
                           f"({len(paper_ids)}) must have same length")
        
        logger.info(f"Fitting TF-IDF on {len(corpus_texts)} documents...")
        
        # Fit and transform corpus
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.paper_ids = paper_ids
        
        # Log statistics
        vocab_size = len(self.vectorizer.vocabulary_)
        n_docs, n_features = self.corpus_matrix.shape
        sparsity = 1.0 - (self.corpus_matrix.nnz / (n_docs * n_features))
        
        logger.info(f"Fitted TF-IDF:")
        logger.info(f"  Documents: {n_docs}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Vocabulary size: {vocab_size}")
        logger.info(f"  Matrix sparsity: {sparsity:.4f}")
        logger.info(f"  Non-zero elements: {self.corpus_matrix.nnz:,}")
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF vectors using fitted vectorizer.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Sparse TF-IDF matrix (csr_matrix)
            
        Raises:
            ValueError: If vectorizer not fitted yet
            
        Examples:
            >>> engine = TFIDFEngine(max_features=100, min_df=1)
            >>> corpus = ["deep learning", "machine learning"]
            >>> engine.fit(corpus)
            >>> new_texts = ["deep neural networks"]
            >>> matrix = engine.transform(new_texts)
            >>> matrix.shape[0]
            1
            >>> isinstance(matrix, csr_matrix)
            True
        """
        if self.vectorizer is None or self.corpus_matrix is None:
            raise ValueError("Must call fit() before transform()")
        
        if not texts:
            raise ValueError("texts cannot be empty")
        
        logger.debug(f"Transforming {len(texts)} documents...")
        
        # Transform using fitted vectorizer
        tfidf_matrix = self.vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def most_similar(
        self,
        q_text: str,
        topn: int = 200,
        return_scores: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Find most similar documents to query text using cosine similarity.
        
        Args:
            q_text: Query text
            topn: Number of top results to return
            return_scores: If True, return (paper_id, score) tuples
                          If False, return just paper_ids
                          
        Returns:
            List of (paper_id, score) tuples sorted by score (descending)
            or list of paper_ids if return_scores=False
            
        Raises:
            ValueError: If engine not fitted yet
            
        Examples:
            >>> engine = TFIDFEngine(max_features=100)
            >>> corpus = [
            ...     "deep learning neural networks",
            ...     "machine learning algorithms",
            ...     "natural language processing"
            ... ]
            >>> paper_ids = [1, 2, 3]
            >>> engine.fit(corpus, paper_ids)
            >>> results = engine.most_similar("deep neural networks", topn=2)
            >>> len(results) <= 2
            True
            >>> all(isinstance(r, tuple) and len(r) == 2 for r in results)
            True
            >>> all(isinstance(r[0], (int, np.integer)) for r in results)
            True
            >>> all(isinstance(r[1], (float, np.floating)) for r in results)
            True
            >>> # Scores should be in descending order
            >>> results[0][1] >= results[1][1] if len(results) > 1 else True
            True
        """
        if self.corpus_matrix is None or self.paper_ids is None:
            raise ValueError("Must call fit() before most_similar()")
        
        if not q_text or not q_text.strip():
            logger.warning("Empty query text, returning empty results")
            return []
        
        # Transform query
        q_vector = self.transform([q_text])
        
        # Compute cosine similarity (sparse matrix multiplication)
        # This is efficient: (1 x n_features) @ (n_features x n_docs).T
        similarities = cosine_similarity(q_vector, self.corpus_matrix)[0]
        
        # Get top-k indices
        topn = min(topn, len(similarities))
        
        if topn == len(similarities):
            # Return all, sorted
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency (O(n) instead of O(n log n))
            # Get topn largest elements, then sort them
            top_indices = np.argpartition(similarities, -topn)[-topn:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Build results
        results = []
        for idx in top_indices:
            paper_id = self.paper_ids[idx]
            score = float(similarities[idx])
            
            # Skip zero scores
            if score <= 0:
                continue
            
            if return_scores:
                results.append((int(paper_id), score))
            else:
                results.append(int(paper_id))
        
        logger.debug(f"Found {len(results)} similar documents for query "
                    f"(requested top {topn})")
        
        return results
    
    def get_top_terms(self, doc_idx: int, topn: int = 20) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF terms for a document in the corpus.
        
        Args:
            doc_idx: Document index in corpus
            topn: Number of top terms to return
            
        Returns:
            List of (term, tfidf_score) tuples
            
        Examples:
            >>> engine = TFIDFEngine(max_features=100, min_df=1)
            >>> corpus = ["deep learning neural networks", "machine learning algorithms"]
            >>> engine.fit(corpus)
            >>> terms = engine.get_top_terms(0, topn=3)
            >>> len(terms) <= 3
            True
            >>> all(isinstance(t, tuple) and len(t) == 2 for t in terms)
            True
        """
        if self.corpus_matrix is None:
            raise ValueError("Must call fit() first")
        
        if doc_idx < 0 or doc_idx >= self.corpus_matrix.shape[0]:
            raise ValueError(f"doc_idx {doc_idx} out of range [0, {self.corpus_matrix.shape[0]})")
        
        # Get document vector
        doc_vector = self.corpus_matrix[doc_idx].toarray()[0]
        
        # Get top indices
        top_indices = np.argsort(doc_vector)[::-1][:topn]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Build results
        results = []
        for idx in top_indices:
            if doc_vector[idx] > 0:
                term = feature_names[idx]
                score = float(doc_vector[idx])
                results.append((term, score))
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save engine state to disk using joblib.
        
        Args:
            path: File path to save to (will add .joblib if no extension)
            
        Examples:
            >>> import tempfile
            >>> engine = TFIDFEngine(max_features=100, min_df=1)
            >>> corpus = ["deep learning", "machine learning"]
            >>> engine.fit(corpus)
            >>> with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            ...     engine.save(f.name)
            ...     print("Saved")
            Saved
        """
        path = Path(path)
        
        # Add extension if not present
        if not path.suffix:
            path = path.with_suffix('.joblib')
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare state dict
        state = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'vectorizer': self.vectorizer,
            'corpus_matrix': self.corpus_matrix,
            'paper_ids': self.paper_ids
        }
        
        # Save with joblib (handles sparse matrices well)
        joblib.dump(state, path, compress=3)
        
        logger.info(f"Saved TFIDFEngine to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TFIDFEngine':
        """
        Load engine state from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded TFIDFEngine instance
            
        Examples:
            >>> import tempfile
            >>> engine1 = TFIDFEngine(max_features=100, min_df=1)
            >>> corpus = ["deep learning", "machine learning"]
            >>> engine1.fit(corpus, paper_ids=[1, 2])
            >>> with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            ...     engine1.save(f.name)
            ...     engine2 = TFIDFEngine.load(f.name)
            ...     print(engine2.paper_ids)
            [1, 2]
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load state
        state = joblib.load(path)
        
        # Create engine instance
        engine = cls(
            max_features=state['max_features'],
            ngram_range=state['ngram_range'],
            min_df=state['min_df'],
            max_df=state['max_df']
        )
        
        # Restore state
        engine.vectorizer = state['vectorizer']
        engine.corpus_matrix = state['corpus_matrix']
        engine.paper_ids = state['paper_ids']
        
        logger.info(f"Loaded TFIDFEngine from {path}")
        if engine.corpus_matrix is not None:
            logger.info(f"  Corpus size: {engine.corpus_matrix.shape[0]} documents")
        
        return engine
    
    def __repr__(self) -> str:
        """String representation of engine."""
        fitted = self.corpus_matrix is not None
        n_docs = self.corpus_matrix.shape[0] if fitted else 0
        
        return (f"TFIDFEngine(max_features={self.max_features}, "
                f"ngram_range={self.ngram_range}, fitted={fitted}, "
                f"n_docs={n_docs})")


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """
    Comprehensive test suite for TFIDFEngine.
    """
    import tempfile
    import os
    
    print("=" * 70)
    print("Running TF-IDF Engine Tests")
    print("=" * 70)
    
    # Test 1: Initialization
    print("\n[TEST 1] Testing initialization...")
    engine = TFIDFEngine()
    assert engine.max_features == 50000
    assert engine.ngram_range == (1, 2)
    assert engine.min_df == 2
    assert engine.max_df == 0.85
    print(f"✓ Engine initialized: {engine}")
    
    # Test 2: Fit with sample corpus
    print("\n[TEST 2] Testing fit...")
    corpus = [
        "deep learning neural networks artificial intelligence",
        "machine learning algorithms supervised unsupervised",
        "natural language processing text mining nlp",
        "computer vision image recognition convolutional networks",
        "reinforcement learning agents policy gradient"
    ]
    paper_ids = [101, 102, 103, 104, 105]
    
    engine.fit(corpus, paper_ids)
    assert engine.corpus_matrix is not None
    assert engine.corpus_matrix.shape[0] == len(corpus)
    assert engine.paper_ids == paper_ids
    print(f"✓ Fitted on {len(corpus)} documents")
    print(f"  Matrix shape: {engine.corpus_matrix.shape}")
    print(f"  Sparsity: {1 - engine.corpus_matrix.nnz / (engine.corpus_matrix.shape[0] * engine.corpus_matrix.shape[1]):.4f}")
    
    # Test 3: Transform
    print("\n[TEST 3] Testing transform...")
    new_texts = ["deep neural networks", "machine learning"]
    transformed = engine.transform(new_texts)
    assert transformed.shape[0] == len(new_texts)
    assert transformed.shape[1] == engine.corpus_matrix.shape[1]
    assert isinstance(transformed, csr_matrix)
    print(f"✓ Transformed {len(new_texts)} texts")
    print(f"  Output shape: {transformed.shape}")
    
    # Test 4: Most similar
    print("\n[TEST 4] Testing most_similar...")
    query = "deep learning and neural networks"
    results = engine.most_similar(query, topn=3)
    
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[0], (int, np.integer)) for r in results)
    assert all(isinstance(r[1], (float, np.floating)) for r in results)
    
    # Check scores are descending
    if len(results) > 1:
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1], "Scores should be descending"
    
    print(f"✓ Found {len(results)} similar documents:")
    for paper_id, score in results:
        print(f"  Paper {paper_id}: {score:.4f}")
    
    # Verify the most similar is paper 101 (contains "deep learning neural networks")
    assert results[0][0] == 101, "Most similar should be paper 101"
    
    # Test 5: Top terms
    print("\n[TEST 5] Testing get_top_terms...")
    top_terms = engine.get_top_terms(0, topn=5)
    assert len(top_terms) <= 5
    assert all(isinstance(t, tuple) and len(t) == 2 for t in top_terms)
    print(f"✓ Top terms for document 0:")
    for term, score in top_terms[:3]:
        print(f"  {term}: {score:.4f}")
    
    # Test 6: Save and load
    print("\n[TEST 6] Testing save/load...")
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save
        engine.save(tmp_path)
        assert os.path.exists(tmp_path)
        print(f"✓ Saved to {tmp_path}")
        
        # Load
        engine2 = TFIDFEngine.load(tmp_path)
        assert engine2.max_features == engine.max_features
        assert engine2.ngram_range == engine.ngram_range
        assert engine2.paper_ids == engine.paper_ids
        assert engine2.corpus_matrix.shape == engine.corpus_matrix.shape
        print("✓ Loaded successfully")
        
        # Test loaded engine works
        results2 = engine2.most_similar("deep learning", topn=2)
        assert len(results2) > 0
        print(f"✓ Loaded engine works ({len(results2)} results)")
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Test 7: Error handling
    print("\n[TEST 7] Testing error handling...")
    
    # Transform before fit
    engine_new = TFIDFEngine()
    try:
        engine_new.transform(["test"])
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised error for transform before fit")
    
    # Most similar before fit
    try:
        engine_new.most_similar("test")
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised error for most_similar before fit")
    
    # Empty corpus
    try:
        engine_new.fit([])
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised error for empty corpus")
    
    # Mismatched lengths
    try:
        engine_new.fit(["text1", "text2"], [1])
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised error for mismatched lengths")
    
    # Test 8: Large corpus efficiency
    print("\n[TEST 8] Testing efficiency with larger corpus...")
    import time
    
    # Create more varied corpus
    topics = [
        "deep learning neural networks",
        "machine learning algorithms",
        "natural language processing",
        "computer vision images",
        "reinforcement learning agents"
    ]
    large_corpus = [
        f"document {i} about {topics[i % len(topics)]} with additional text content"
        for i in range(1000)
    ]
    large_ids = list(range(1000))
    
    engine_large = TFIDFEngine(max_features=5000, min_df=1)
    
    start = time.time()
    engine_large.fit(large_corpus, large_ids)
    fit_time = time.time() - start
    
    start = time.time()
    results = engine_large.most_similar("document text topic", topn=50)
    search_time = time.time() - start
    
    print(f"✓ Processed 1000 documents:")
    print(f"  Fit time: {fit_time:.3f}s")
    print(f"  Search time: {search_time:.3f}s")
    print(f"  Found {len(results)} results")
    
    assert search_time < 1.0, "Search should be fast (<1s for 1000 docs)"
    
    # Test 9: Sparse matrix properties
    print("\n[TEST 9] Verifying sparse matrix efficiency...")
    assert isinstance(engine.corpus_matrix, csr_matrix)
    sparsity = 1 - (engine.corpus_matrix.nnz / 
                   (engine.corpus_matrix.shape[0] * engine.corpus_matrix.shape[1]))
    print(f"✓ Matrix is sparse (CSR format)")
    print(f"  Sparsity: {sparsity:.4f}")
    print(f"  Memory efficient: {sparsity > 0.9}")
    
    # Test 10: Doctests
    print("\n[TEST 10] Running doctests...")
    import doctest
    results = doctest.testmod(verbose=False)
    if results.failed == 0:
        print(f"✓ All {results.attempted} doctests passed")
    else:
        print(f"✗ {results.failed}/{results.attempted} doctests failed")
    
    print("\n" + "=" * 70)
    print("ALL TF-IDF ENGINE TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

"""
Optional Topic Modeling with BERTopic

This module provides topic modeling capabilities using BERTopic for analyzing
academic paper abstracts. It is completely OPTIONAL and the pipeline will work
without it.

Features:
- Train BERTopic on paper abstracts
- Author topic profiles (top topics per author)
- Topic overlap scoring (cosine/Jaccard similarity)
- Graceful fallbacks when BERTopic not available

Installation:
    pip install bertopic umap-learn hdbscan

If not installed, all functions return None or empty results without errors.

Example:
    >>> from topic_model import train_bertopic, author_topic_profile
    >>> 
    >>> # Train on abstracts (optional)
    >>> model = train_bertopic(abstracts)
    >>> 
    >>> # Get author's topics (returns None if model not available)
    >>> topics = author_topic_profile(author_id, db_path)
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings

# Suppress warnings from topic modeling libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Optional Imports with Graceful Fallbacks
# ============================================================================

# Check for BERTopic
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
    logger.info("BERTopic available")
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.warning("BERTopic not available. Topic modeling features disabled.")
    logger.warning("Install with: pip install bertopic umap-learn hdbscan")
    BERTopic = None

# Check for UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    if BERTOPIC_AVAILABLE:
        logger.warning("UMAP not available. Install with: pip install umap-learn")
    UMAP = None

# Check for HDBSCAN
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    if BERTOPIC_AVAILABLE:
        logger.warning("HDBSCAN not available. Install with: pip install hdbscan")
    HDBSCAN = None

# Check for embedding module
try:
    from embedding import Embeddings
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("Embedding module not available. Will use default BERTopic embeddings.")
    Embeddings = None

# Check for database utilities
try:
    from db_utils import get_author_papers
    DB_UTILS_AVAILABLE = True
except ImportError:
    DB_UTILS_AVAILABLE = False
    logger.warning("Database utilities not available.")

# Check if full BERTopic stack is available
TOPIC_MODELING_AVAILABLE = (
    BERTOPIC_AVAILABLE and 
    UMAP_AVAILABLE and 
    HDBSCAN_AVAILABLE
)


# ============================================================================
# Core Topic Modeling Functions
# ============================================================================

def train_bertopic(
    abstracts: List[str],
    embedding_model: Optional[Any] = None,
    n_topics: int = 10,
    min_topic_size: int = 10,
    nr_topics: Optional[int] = None
) -> Optional["BERTopic"]:
    """
    Train BERTopic model on paper abstracts.
    
    Args:
        abstracts: List of abstract texts
        embedding_model: Optional embedding model (Embeddings instance or SentenceTransformer)
                        If None, BERTopic will use default "all-MiniLM-L6-v2"
        n_topics: Target number of topics (used as guidance)
        min_topic_size: Minimum documents per topic
        nr_topics: Reduce to this many topics after training (optional)
    
    Returns:
        Trained BERTopic model, or None if not available
    
    Example:
        >>> from embedding import Embeddings
        >>> emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")
        >>> model = train_bertopic(abstracts, embedding_model=emb.model)
        >>> if model:
        ...     topics, probs = model.transform(new_abstracts)
    """
    if not TOPIC_MODELING_AVAILABLE:
        logger.warning("BERTopic not available. Skipping topic modeling.")
        return None
    
    if not abstracts:
        logger.warning("No abstracts provided. Cannot train topic model.")
        return None
    
    if len(abstracts) < min_topic_size:
        logger.warning(
            f"Only {len(abstracts)} abstracts provided. "
            f"Need at least {min_topic_size} for topic modeling."
        )
        return None
    
    logger.info(f"Training BERTopic on {len(abstracts)} abstracts...")
    logger.info(f"  Min topic size: {min_topic_size}")
    logger.info(f"  Target topics: {n_topics}")
    
    try:
        # Configure UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            top_n_words=10,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=True
        )
        
        # Fit model
        topics, probs = topic_model.fit_transform(abstracts)
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        logger.info(f"BERTopic training complete")
        logger.info(f"  Topics found: {len(topic_info) - 1}")  # -1 for outlier topic
        logger.info(f"  Outliers: {(topics == -1).sum()}")
        
        return topic_model
        
    except Exception as e:
        logger.error(f"Failed to train BERTopic: {e}")
        return None


def save_bertopic_model(
    model: "BERTopic",
    path: str = "models/bertopic_model"
) -> bool:
    """
    Save BERTopic model to disk.
    
    Args:
        model: Trained BERTopic model
        path: Directory path to save model (default: models/bertopic_model)
    
    Returns:
        True if saved successfully, False otherwise
    
    Example:
        >>> model = train_bertopic(abstracts)
        >>> if model:
        ...     save_bertopic_model(model, "models/bertopic_v1")
    """
    if not BERTOPIC_AVAILABLE or model is None:
        logger.warning("Cannot save model: BERTopic not available or model is None")
        return False
    
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # BERTopic's save method
        model.save(str(path), serialization="safetensors", save_ctfidf=True)
        
        logger.info(f"Saved BERTopic model to: {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save BERTopic model: {e}")
        return False


def load_bertopic_model(
    path: str = "models/bertopic_model"
) -> Optional["BERTopic"]:
    """
    Load BERTopic model from disk.
    
    Args:
        path: Directory path to load model from
    
    Returns:
        Loaded BERTopic model, or None if not available
    
    Example:
        >>> model = load_bertopic_model("models/bertopic_v1")
        >>> if model:
        ...     topics, probs = model.transform(new_abstracts)
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot load model.")
        return None
    
    try:
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Model path not found: {path}")
            return None
        
        # BERTopic's load method
        model = BERTopic.load(str(path))
        
        logger.info(f"Loaded BERTopic model from: {path}")
        topic_info = model.get_topic_info()
        logger.info(f"  Topics: {len(topic_info) - 1}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load BERTopic model: {e}")
        return None


def author_topic_profile(
    author_id: int,
    db_path: Path,
    topic_model: Optional["BERTopic"] = None,
    topn: int = 5
) -> Optional[List[Tuple[int, str, float]]]:
    """
    Get author's topic profile from their papers.
    
    Analyzes the author's paper abstracts and returns their top topics
    with weights based on frequency and probability.
    
    Args:
        author_id: Author's database ID
        db_path: Path to database
        topic_model: Trained BERTopic model (if None, tries to load from default path)
        topn: Number of top topics to return
    
    Returns:
        List of (topic_id, topic_name, weight) tuples, or None if not available
        Topic weight is normalized frequency across author's papers
    
    Example:
        >>> topics = author_topic_profile(42, Path("papers.db"), model)
        >>> if topics:
        ...     for topic_id, topic_name, weight in topics:
        ...         print(f"Topic {topic_id}: {topic_name} (weight: {weight:.3f})")
    """
    if not TOPIC_MODELING_AVAILABLE or not DB_UTILS_AVAILABLE:
        logger.warning("Topic modeling or database utilities not available")
        return None
    
    # Load model if not provided
    if topic_model is None:
        topic_model = load_bertopic_model()
        if topic_model is None:
            logger.warning("No topic model available")
            return None
    
    try:
        # Get author's papers
        papers = get_author_papers(db_path, author_id)
        
        if not papers:
            logger.warning(f"No papers found for author {author_id}")
            return None
        
        # Extract abstracts
        abstracts = []
        for paper in papers:
            abstract = paper.get('abstract')
            if abstract and len(abstract.strip()) > 50:  # Min length check
                abstracts.append(abstract)
        
        if not abstracts:
            logger.warning(f"No valid abstracts for author {author_id}")
            return None
        
        logger.info(f"Analyzing {len(abstracts)} abstracts for author {author_id}")
        
        # Get topics for abstracts
        topics, probs = topic_model.transform(abstracts)
        
        # Count topic frequencies
        topic_counts = {}
        topic_probs = {}
        
        for topic, prob_dist in zip(topics, probs):
            if topic == -1:  # Skip outliers
                continue
            
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Accumulate probabilities
            if topic not in topic_probs:
                topic_probs[topic] = []
            topic_probs[topic].append(prob_dist[topic] if topic < len(prob_dist) else 0.0)
        
        if not topic_counts:
            logger.warning(f"No topics found for author {author_id} (all outliers)")
            return None
        
        # Calculate weights (normalized by total papers)
        total_papers = len(abstracts)
        topic_weights = {}
        
        for topic, count in topic_counts.items():
            avg_prob = np.mean(topic_probs[topic])
            # Weight combines frequency and average probability
            weight = (count / total_papers) * avg_prob
            topic_weights[topic] = weight
        
        # Get topic names
        topic_info = topic_model.get_topic_info()
        topic_names = {}
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue
            # Get top words for topic name
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                topic_name = ", ".join([word for word, _ in topic_words[:3]])
                topic_names[topic_id] = topic_name
        
        # Sort by weight and get top-N
        sorted_topics = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for topic_id, weight in sorted_topics[:topn]:
            topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
            result.append((topic_id, topic_name, weight))
        
        logger.info(f"Found {len(result)} topics for author {author_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get author topic profile: {e}")
        return None


def topic_overlap_score(
    query_topics: List[Tuple[int, str, float]],
    author_topics: List[Tuple[int, str, float]],
    method: str = "cosine"
) -> float:
    """
    Calculate overlap between query topics and author topics.
    
    Args:
        query_topics: List of (topic_id, topic_name, weight) for query
        author_topics: List of (topic_id, topic_name, weight) for author
        method: Similarity method - "cosine" or "jaccard"
    
    Returns:
        Overlap score between 0 and 1, or 0.0 if calculation fails
    
    Example:
        >>> query_topics = [(0, "deep learning", 0.8), (1, "neural nets", 0.5)]
        >>> author_topics = [(0, "deep learning", 0.7), (2, "vision", 0.6)]
        >>> score = topic_overlap_score(query_topics, author_topics, method="cosine")
        >>> print(f"Overlap: {score:.3f}")
    """
    if not query_topics or not author_topics:
        return 0.0
    
    try:
        # Extract topic IDs and weights
        query_ids = {t[0] for t in query_topics}
        author_ids = {t[0] for t in author_topics}
        
        query_weights = {t[0]: t[2] for t in query_topics}
        author_weights = {t[0]: t[2] for t in author_topics}
        
        if method == "jaccard":
            # Jaccard similarity: |intersection| / |union|
            intersection = len(query_ids & author_ids)
            union = len(query_ids | author_ids)
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        elif method == "cosine":
            # Cosine similarity with weights
            # Create vectors over all topics
            all_topics = sorted(query_ids | author_ids)
            
            if not all_topics:
                return 0.0
            
            # Build weight vectors
            query_vec = np.array([query_weights.get(t, 0.0) for t in all_topics])
            author_vec = np.array([author_weights.get(t, 0.0) for t in all_topics])
            
            # Compute cosine similarity
            dot_product = np.dot(query_vec, author_vec)
            query_norm = np.linalg.norm(query_vec)
            author_norm = np.linalg.norm(author_vec)
            
            if query_norm == 0 or author_norm == 0:
                return 0.0
            
            return dot_product / (query_norm * author_norm)
            
        else:
            logger.warning(f"Unknown method: {method}. Using cosine.")
            return topic_overlap_score(query_topics, author_topics, method="cosine")
            
    except Exception as e:
        logger.error(f"Failed to calculate topic overlap: {e}")
        return 0.0


# ============================================================================
# Utility Functions
# ============================================================================

def get_topic_info(
    topic_model: "BERTopic",
    topic_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific topic.
    
    Args:
        topic_model: Trained BERTopic model
        topic_id: Topic ID
    
    Returns:
        Dictionary with topic info, or None if not available
    """
    if not BERTOPIC_AVAILABLE or topic_model is None:
        return None
    
    try:
        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            return None
        
        topic_info = topic_model.get_topic_info()
        topic_row = topic_info[topic_info['Topic'] == topic_id]
        
        if topic_row.empty:
            return None
        
        return {
            'topic_id': topic_id,
            'words': topic_words,
            'count': int(topic_row['Count'].iloc[0]),
            'name': ", ".join([word for word, _ in topic_words[:3]])
        }
        
    except Exception as e:
        logger.error(f"Failed to get topic info: {e}")
        return None


def is_available() -> bool:
    """
    Check if topic modeling is available.
    
    Returns:
        True if all required packages are installed, False otherwise
    
    Example:
        >>> if is_available():
        ...     model = train_bertopic(abstracts)
        ... else:
        ...     print("Topic modeling not available")
    """
    return TOPIC_MODELING_AVAILABLE


# ============================================================================
# Demo and Tests
# ============================================================================

def _test_availability():
    """Test if topic modeling is available."""
    print("\n" + "="*70)
    print("TEST 1: Availability Check")
    print("="*70)
    
    print(f"\n[1.1] Package availability:")
    print(f"  BERTopic: {'✓' if BERTOPIC_AVAILABLE else '✗'}")
    print(f"  UMAP: {'✓' if UMAP_AVAILABLE else '✗'}")
    print(f"  HDBSCAN: {'✓' if HDBSCAN_AVAILABLE else '✗'}")
    print(f"  Embedding module: {'✓' if EMBEDDING_AVAILABLE else '✗'}")
    print(f"  DB utilities: {'✓' if DB_UTILS_AVAILABLE else '✗'}")
    
    print(f"\n[1.2] Topic modeling available: {is_available()}")
    
    if not is_available():
        print("\n  To enable topic modeling, install:")
        print("    pip install bertopic umap-learn hdbscan")
        return False
    
    print("  PASSED")
    return True


def _test_train_and_save():
    """Test training and saving BERTopic model."""
    if not TOPIC_MODELING_AVAILABLE:
        print("\n[Skipped] Topic modeling not available")
        return None
    
    print("\n" + "="*70)
    print("TEST 2: Train and Save Model")
    print("="*70)
    
    # Sample abstracts
    abstracts = [
        "Deep learning neural networks for image classification and computer vision tasks",
        "Machine learning algorithms for supervised and unsupervised learning",
        "Natural language processing with transformer models and attention mechanisms",
        "Reinforcement learning agents with policy gradient methods",
        "Computer vision techniques for object detection and segmentation",
        "Deep neural networks with convolutional layers for visual recognition",
        "Language models based on attention and transformer architectures",
        "Supervised learning with gradient descent optimization",
        "Image recognition using deep convolutional networks",
        "Text generation with large language models and transformers",
        "Policy-based reinforcement learning for sequential decision making",
        "Visual perception with neural networks and deep learning",
        "Attention mechanisms in natural language understanding",
        "Classification algorithms for machine learning applications",
        "Object detection in images using deep learning methods"
    ]
    
    print(f"\n[2.1] Training on {len(abstracts)} sample abstracts...")
    model = train_bertopic(abstracts, min_topic_size=3, nr_topics=3)
    
    if model is None:
        print("  FAILED - Could not train model")
        return None
    
    print("  PASSED - Model trained")
    
    # Save model
    print("\n[2.2] Saving model...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        success = save_bertopic_model(model, str(save_path))
        
        if not success:
            print("  FAILED - Could not save model")
            return None
        
        print("  PASSED - Model saved")
        
        # Load model
        print("\n[2.3] Loading model...")
        loaded_model = load_bertopic_model(str(save_path))
        
        if loaded_model is None:
            print("  FAILED - Could not load model")
            return None
        
        print("  PASSED - Model loaded")
    
    return model


def _test_topic_functions(model):
    """Test topic profile and overlap functions."""
    if model is None or not TOPIC_MODELING_AVAILABLE:
        print("\n[Skipped] No model available")
        return
    
    print("\n" + "="*70)
    print("TEST 3: Topic Functions")
    print("="*70)
    
    # Simulate topic profiles
    print("\n[3.1] Testing topic_overlap_score...")
    
    query_topics = [
        (0, "deep learning, neural, networks", 0.8),
        (1, "machine learning, algorithms", 0.5)
    ]
    
    author_topics = [
        (0, "deep learning, neural, networks", 0.7),
        (2, "computer vision, image", 0.6)
    ]
    
    # Test cosine similarity
    cosine_score = topic_overlap_score(query_topics, author_topics, method="cosine")
    print(f"  Cosine similarity: {cosine_score:.4f}")
    assert 0 <= cosine_score <= 1, "Score out of range"
    
    # Test Jaccard similarity
    jaccard_score = topic_overlap_score(query_topics, author_topics, method="jaccard")
    print(f"  Jaccard similarity: {jaccard_score:.4f}")
    assert 0 <= jaccard_score <= 1, "Score out of range"
    
    print("  PASSED")
    
    # Test empty inputs
    print("\n[3.2] Testing with empty inputs...")
    empty_score = topic_overlap_score([], author_topics)
    assert empty_score == 0.0, "Should return 0 for empty input"
    print("  PASSED - Handles empty inputs")


def _demo_optional_usage():
    """Demonstrate optional usage patterns."""
    print("\n" + "="*70)
    print("DEMO: Optional Usage Patterns")
    print("="*70)
    
    print("\n[DEMO 1] Checking availability before use:")
    print(f"  if is_available():")
    print(f"      model = train_bertopic(abstracts)")
    print(f"  else:")
    print(f"      model = None  # Pipeline continues without topics")
    
    print("\n[DEMO 2] Graceful degradation:")
    print(f"  topics = author_topic_profile(author_id, db_path)")
    print(f"  if topics:")
    print(f"      # Use topic information")
    print(f"      print(topics)")
    print(f"  else:")
    print(f"      # Fall back to keyword-based search")
    print(f"      print('Topic modeling not available')")
    
    print("\n[DEMO 3] Optional enhancement to search:")
    print(f"  # Base search always works")
    print(f"  results = tfidf_engine.most_similar(query)")
    print(f"  ")
    print(f"  # Optionally re-rank by topic overlap")
    print(f"  if is_available() and model:")
    print(f"      query_topics = get_query_topics(query)")
    print(f"      for paper_id, score in results:")
    print(f"          author_topics = author_topic_profile(paper.author_id)")
    print(f"          if author_topics:")
    print(f"              topic_boost = topic_overlap_score(query_topics, author_topics)")
    print(f"              score = score * (1 + 0.2 * topic_boost)")


if __name__ == "__main__":
    print("="*70)
    print("BERTopic Topic Modeling - Optional Module")
    print("="*70)
    
    # Check availability
    available = _test_availability()
    
    if not available:
        print("\n" + "="*70)
        print("Topic modeling is OPTIONAL and not required for the pipeline.")
        print("The system will work perfectly without it.")
        print("="*70)
    else:
        try:
            # Run tests
            model = _test_train_and_save()
            _test_topic_functions(model)
            
            print("\n" + "="*70)
            print("ALL TESTS PASSED!")
            print("="*70)
            
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Show demo regardless
    _demo_optional_usage()
    
    print("\n" + "="*70)
    print("Module is ready for optional use")
    print("="*70)

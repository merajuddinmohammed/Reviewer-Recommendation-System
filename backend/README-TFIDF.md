# TF-IDF Engine Documentation

## Overview

The `TFIDFEngine` class provides **sparse, efficient TF-IDF document similarity search** for academic paper retrieval. It wraps `sklearn.TfidfVectorizer` and uses sparse cosine similarity for fast searches across large corpora.

## Key Features

- **Sparse Matrix Operations**: Uses `scipy.sparse.csr_matrix` for memory-efficient storage
- **Fast Similarity Search**: Cosine similarity computed only on non-zero elements
- **Flexible Configuration**: Customizable n-grams, min/max document frequency, feature limits
- **Persistent Storage**: Serialize/deserialize models with joblib
- **Paper ID Tracking**: Returns (paper_id, score) tuples for easy integration

## Architecture

```
TFIDFEngine
├── TfidfVectorizer      # Text → TF-IDF vectors
├── corpus_matrix        # Sparse CSR matrix (docs × features)
├── paper_ids            # List of paper IDs
└── Methods
    ├── fit()            # Train on corpus
    ├── transform()      # Vectorize new texts
    ├── most_similar()   # Cosine similarity search
    ├── get_top_terms()  # Extract important terms
    ├── save()           # Serialize to disk
    └── load()           # Deserialize from disk
```

## Usage

### Basic Example

```python
from tfidf_engine import TFIDFEngine

# 1. Initialize
engine = TFIDFEngine(
    max_features=50000,      # Vocabulary size limit
    ngram_range=(1, 2),      # Unigrams + bigrams
    min_df=2,                # Ignore terms in < 2 docs
    max_df=0.85              # Ignore terms in > 85% docs
)

# 2. Fit on corpus
corpus_texts = [
    "deep learning neural networks",
    "machine learning algorithms",
    "natural language processing"
]
paper_ids = [101, 102, 103]
engine.fit(corpus_texts, paper_ids)

# 3. Search for similar papers
results = engine.most_similar("deep neural nets", topn=5)
for paper_id, score in results:
    print(f"Paper {paper_id}: {score:.4f}")
```

### Advanced: Save and Load

```python
# Save trained model
engine.save("models/tfidf_v1.joblib")

# Load later
engine = TFIDFEngine.load("models/tfidf_v1.joblib")
results = engine.most_similar("query text")
```

### Extract Top Terms

```python
# Get most important terms per document
top_terms = engine.get_top_terms(topn=10)
for doc_idx, terms_scores in enumerate(top_terms):
    print(f"Doc {doc_idx}:")
    for term, score in terms_scores:
        print(f"  {term}: {score:.4f}")
```

## API Reference

### `__init__(max_features, ngram_range, min_df, max_df)`

**Parameters:**
- `max_features` (int): Maximum vocabulary size (default: 50000)
- `ngram_range` (tuple): (min_n, max_n) for n-grams (default: (1, 2))
- `min_df` (int): Minimum document frequency (default: 2)
- `max_df` (float): Maximum document frequency as fraction (default: 0.85)

**Example:**
```python
# Conservative settings (small corpus)
engine = TFIDFEngine(min_df=1, max_df=1.0)

# Aggressive filtering (large corpus)
engine = TFIDFEngine(min_df=10, max_df=0.5)
```

---

### `fit(corpus_texts, paper_ids)`

Fit the TF-IDF vectorizer on a corpus.

**Parameters:**
- `corpus_texts` (List[str]): List of document texts
- `paper_ids` (List[int]): Corresponding paper IDs

**Returns:** `self`

**Raises:**
- `ValueError`: If corpus_texts is empty
- `ValueError`: If lengths don't match
- `ValueError`: If no terms remain after pruning

**Example:**
```python
corpus = ["doc one", "doc two", "doc three"]
ids = [1, 2, 3]
engine.fit(corpus, ids)
print(f"Vocabulary size: {len(engine.vectorizer.vocabulary_)}")
```

---

### `transform(texts)`

Transform new texts into TF-IDF vectors.

**Parameters:**
- `texts` (List[str]): Texts to transform

**Returns:** `csr_matrix` of shape (n_texts, n_features)

**Raises:**
- `ValueError`: If called before fit()

**Example:**
```python
new_docs = ["new document", "another query"]
vectors = engine.transform(new_docs)
print(vectors.shape)  # (2, n_features)
```

---

### `most_similar(q_text, topn=10)`

Find most similar documents using cosine similarity.

**Parameters:**
- `q_text` (str): Query text
- `topn` (int): Number of results to return (default: 10)

**Returns:** List of `(paper_id, score)` tuples, sorted by score descending

**Raises:**
- `ValueError`: If called before fit()

**Example:**
```python
results = engine.most_similar("neural networks", topn=5)
# [(101, 0.95), (103, 0.87), (105, 0.72), ...]

# Check if any results found
if results:
    best_id, best_score = results[0]
    print(f"Best match: Paper {best_id} (score: {best_score:.4f})")
```

---

### `get_top_terms(topn=10)`

Extract top TF-IDF terms for each document.

**Parameters:**
- `topn` (int): Number of top terms per document (default: 10)

**Returns:** List of lists: `[[(term, score), ...], ...]`

**Raises:**
- `ValueError`: If called before fit()

**Example:**
```python
top_terms = engine.get_top_terms(topn=5)
for i, doc_terms in enumerate(top_terms):
    print(f"Document {i}:")
    for term, score in doc_terms:
        print(f"  {term}: {score:.3f}")
```

---

### `save(path)`

Save engine to disk using joblib.

**Parameters:**
- `path` (str): File path (recommend `.joblib` extension)

**Example:**
```python
engine.save("models/tfidf_20240101.joblib")
```

---

### `load(path)` (class method)

Load engine from disk.

**Parameters:**
- `path` (str): File path to load from

**Returns:** `TFIDFEngine` instance

**Example:**
```python
engine = TFIDFEngine.load("models/tfidf_20240101.joblib")
```

## Performance

### Benchmarks

Tested on 1000 documents:
- **Fit time**: ~10ms
- **Search time**: ~1ms per query
- **Memory**: Sparse matrix uses ~0.1-5% of dense equivalent

### Sparsity

Typical sparsity for academic papers:
- Small corpus (10-100 docs): 60-80% sparse
- Medium corpus (100-1000 docs): 80-95% sparse
- Large corpus (1000+ docs): 95-99% sparse

### Scaling

| Corpus Size | Memory (sparse) | Search Time |
|-------------|-----------------|-------------|
| 100 docs    | ~1 MB           | <1 ms       |
| 1,000 docs  | ~10 MB          | 1-5 ms      |
| 10,000 docs | ~100 MB         | 10-50 ms    |
| 100,000 docs| ~1 GB           | 100-500 ms  |

## Configuration Tips

### Small Corpus (<100 docs)
```python
engine = TFIDFEngine(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,        # Don't filter rare terms
    max_df=1.0       # Keep all terms
)
```

### Medium Corpus (100-10k docs)
```python
engine = TFIDFEngine(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,        # At least 2 documents
    max_df=0.85      # Remove very common terms
)
```

### Large Corpus (>10k docs)
```python
engine = TFIDFEngine(
    max_features=100000,
    ngram_range=(1, 3),  # Add trigrams
    min_df=5,            # More aggressive filtering
    max_df=0.5           # Remove top 50% common
)
```

## Integration with Database

```python
from db_utils import get_connection, get_all_papers
from tfidf_engine import TFIDFEngine

# 1. Load papers from database
db_path = Path("data/papers.db")
papers = get_all_papers(db_path)

# 2. Build corpus
corpus_texts = []
paper_ids = []
for p in papers:
    # Combine title, abstract, fulltext
    text = f"{p['title']} {p['abstract']} {p['fulltext']}"
    corpus_texts.append(text)
    paper_ids.append(p['id'])

# 3. Fit engine
engine = TFIDFEngine()
engine.fit(corpus_texts, paper_ids)
engine.save("models/tfidf_latest.joblib")

# 4. Query
query = "transformer attention mechanisms"
results = engine.most_similar(query, topn=10)

# 5. Retrieve full paper details
with get_connection(db_path) as conn:
    for paper_id, score in results:
        cursor = conn.execute(
            "SELECT title, year FROM papers WHERE id = ?",
            (paper_id,)
        )
        row = cursor.fetchone()
        print(f"{row[0]} ({row[1]}): {score:.4f}")
```

## Error Handling

```python
from tfidf_engine import TFIDFEngine

engine = TFIDFEngine()

# Handle empty corpus
try:
    engine.fit([], [])
except ValueError as e:
    print(f"Error: {e}")  # "corpus_texts cannot be empty"

# Handle mismatched lengths
try:
    engine.fit(["doc1"], [1, 2])
except ValueError as e:
    print(f"Error: {e}")  # "corpus_texts and paper_ids must be same length"

# Handle search before fit
try:
    engine.most_similar("query")
except ValueError as e:
    print(f"Error: {e}")  # "Must call fit() before transform()"

# Handle no terms after pruning
try:
    engine.fit(["a", "b"], [1, 2])  # Too simple
except ValueError as e:
    print(f"Error: {e}")  # "no terms remain after pruning"
```

## Testing

Run the test suite:
```powershell
cd backend
python tfidf_engine.py
```

Quick verification:
```powershell
python test_tfidf_quick.py
```

Expected output:
```
[x] fit() with corpus_texts and paper_ids
[x] transform() returns csr_matrix
[x] most_similar() returns (id, score) pairs
[x] Cosine similarity is sparse & efficient
[x] save()/load() with joblib
[x] Paper IDs stored internally
```

## Logging

The module uses Python's logging module:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Disable info logging
logging.getLogger('tfidf_engine').setLevel(logging.WARNING)
```

Log messages include:
- Fit progress (corpus size, vocabulary size, sparsity)
- Save/load operations (file paths)
- Error conditions (empty corpus, missing fit)

## Dependencies

- `scikit-learn >= 1.3.0`: TfidfVectorizer
- `scipy >= 1.11.0`: Sparse matrices
- `numpy >= 1.24.0`: Array operations
- `joblib >= 1.3.0`: Serialization

Install with:
```powershell
pip install scikit-learn scipy numpy joblib
```

## Related Files

- `backend/db_utils.py`: Database layer for paper storage
- `backend/utils.py`: Text cleaning utilities (use before TF-IDF)
- `backend/parser.py`: PDF ingestion pipeline

## References

- [TF-IDF Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [sklearn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)

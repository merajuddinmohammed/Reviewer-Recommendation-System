# Topic Modeling Documentation (OPTIONAL)

## ⚠️ IMPORTANT: This Module is OPTIONAL

**The entire pipeline works perfectly WITHOUT this module.** Topic modeling with BERTopic is an optional enhancement that provides additional insights but is NOT required for core functionality.

## Overview

The `topic_model.py` module provides **optional** topic modeling capabilities using BERTopic to analyze academic paper abstracts. It identifies common themes and topics across papers, allowing for:

- Topic-based paper analysis
- Author expertise profiling by topics
- Topic-based similarity scoring
- Enhanced search with topic context

## When to Use

**Use topic modeling when:**
- You have 50+ papers with abstracts
- You want to discover research themes
- You need author expertise profiles
- You want topic-based recommendations

**Skip topic modeling when:**
- You have <50 papers
- Installation issues occur (requires C++ compiler on Windows)
- You only need keyword/semantic search
- You want a simpler system

## Installation (Optional)

### Requirements
- Python 3.10+
- C++ Build Tools (Windows) - [Download here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Install

```powershell
# Install BERTopic and dependencies
pip install bertopic umap-learn hdbscan

# Note: hdbscan requires C++ compiler on Windows
# If installation fails, the system will work without it
```

### Verify Installation

```python
from topic_model import is_available

if is_available():
    print("Topic modeling available!")
else:
    print("Topic modeling not available (system works without it)")
```

## API Reference

### `is_available() -> bool`

Check if topic modeling is available.

**Returns:**
- `True` if BERTopic, UMAP, and HDBSCAN are installed
- `False` otherwise (system will use fallbacks)

**Example:**
```python
from topic_model import is_available

if is_available():
    # Use topic modeling
    model = train_bertopic(abstracts)
else:
    # Use TF-IDF or embeddings instead
    print("Skipping topic modeling")
```

---

### `train_bertopic(abstracts, embedding_model, ...) -> Optional[BERTopic]`

Train BERTopic model on paper abstracts.

**Parameters:**
- `abstracts` (List[str]): Paper abstracts
- `embedding_model` (optional): Pre-trained embedding model for speed
- `n_topics` (int): Target number of topics (default: 10)
- `min_topic_size` (int): Minimum papers per topic (default: 10)
- `nr_topics` (int, optional): Reduce to this many topics after training

**Returns:**
- Trained BERTopic model, or `None` if not available

**Example:**
```python
from embedding import Embeddings
from topic_model import train_bertopic

# Use existing embedding model for speed
emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")

# Train on abstracts
model = train_bertopic(
    abstracts=paper_abstracts,
    embedding_model=emb.model,  # Reuse embeddings
    min_topic_size=10,
    nr_topics=20
)

if model:
    # Model trained successfully
    topics, probs = model.transform(new_abstracts)
else:
    # Topic modeling not available
    print("Continuing without topics")
```

---

### `save_bertopic_model(model, path) -> bool`

Save BERTopic model to disk.

**Parameters:**
- `model` (BERTopic): Trained model
- `path` (str): Directory path (default: "models/bertopic_model")

**Returns:**
- `True` if saved successfully, `False` otherwise

**Example:**
```python
model = train_bertopic(abstracts)
if model:
    save_bertopic_model(model, "models/bertopic_v1")
```

---

### `load_bertopic_model(path) -> Optional[BERTopic]`

Load BERTopic model from disk.

**Parameters:**
- `path` (str): Directory path to model

**Returns:**
- Loaded model, or `None` if not available

**Example:**
```python
model = load_bertopic_model("models/bertopic_v1")
if model:
    topics, probs = model.transform(new_abstracts)
```

---

### `author_topic_profile(author_id, db_path, topic_model, topn) -> Optional[List]`

Get author's top topics from their papers.

**Parameters:**
- `author_id` (int): Author's database ID
- `db_path` (Path): Path to database
- `topic_model` (BERTopic, optional): Trained model (loads default if None)
- `topn` (int): Number of top topics (default: 5)

**Returns:**
- List of `(topic_id, topic_name, weight)` tuples, or `None` if not available
- Weight is normalized frequency across author's papers

**Example:**
```python
from pathlib import Path
from topic_model import author_topic_profile

topics = author_topic_profile(
    author_id=42,
    db_path=Path("papers.db"),
    topn=5
)

if topics:
    for topic_id, topic_name, weight in topics:
        print(f"Topic {topic_id}: {topic_name} (weight: {weight:.3f})")
else:
    print("No topics available")
```

---

### `topic_overlap_score(query_topics, author_topics, method) -> float`

Calculate similarity between query topics and author topics.

**Parameters:**
- `query_topics` (List): List of `(topic_id, name, weight)` tuples
- `author_topics` (List): List of `(topic_id, name, weight)` tuples
- `method` (str): "cosine" or "jaccard" (default: "cosine")

**Returns:**
- Overlap score between 0 and 1

**Example:**
```python
from topic_model import topic_overlap_score

query_topics = [(0, "deep learning", 0.8), (1, "neural nets", 0.5)]
author_topics = [(0, "deep learning", 0.7), (2, "vision", 0.6)]

# Cosine similarity (considers weights)
cosine = topic_overlap_score(query_topics, author_topics, method="cosine")
print(f"Cosine: {cosine:.3f}")

# Jaccard similarity (only topic IDs)
jaccard = topic_overlap_score(query_topics, author_topics, method="jaccard")
print(f"Jaccard: {jaccard:.3f}")
```

---

## Usage Patterns

### Pattern 1: Check Before Use

```python
from topic_model import is_available, train_bertopic

if is_available():
    # Use topic modeling
    model = train_bertopic(abstracts)
    # ... use model
else:
    # Fall back to other search methods
    results = tfidf_engine.most_similar(query)
```

### Pattern 2: Graceful Degradation

```python
from topic_model import author_topic_profile

# Always returns None if not available
topics = author_topic_profile(author_id, db_path)

if topics:
    # Enhance search with topics
    for topic_id, name, weight in topics:
        print(f"Expert in: {name}")
else:
    # Continue without topics
    print("Using keyword-based search")
```

### Pattern 3: Optional Re-ranking

```python
from tfidf_engine import TFIDFEngine
from topic_model import is_available, author_topic_profile, topic_overlap_score

# Base search always works
tfidf = TFIDFEngine()
results = tfidf.most_similar(query, topn=20)

# Optionally re-rank by topic overlap
if is_available():
    model = load_bertopic_model()
    if model:
        # Get query topics
        query_topics = model.transform([query])[0]
        
        # Re-rank results
        for paper_id, score in results:
            author_topics = author_topic_profile(paper.author_id, db_path)
            if author_topics:
                topic_boost = topic_overlap_score(query_topics, author_topics)
                score = score * (1 + 0.2 * topic_boost)  # 20% boost

print("Results (with optional topic boost):", results)
```

## Integration Examples

### With Database

```python
from pathlib import Path
from db_utils import get_all_papers
from topic_model import is_available, train_bertopic, save_bertopic_model

db_path = Path("papers.db")

# Load papers
papers = get_all_papers(db_path)

# Extract abstracts
abstracts = [p['abstract'] for p in papers if p['abstract']]

print(f"Found {len(abstracts)} abstracts")

# Train topic model (optional)
if is_available() and len(abstracts) >= 50:
    print("Training topic model...")
    model = train_bertopic(abstracts, min_topic_size=10)
    
    if model:
        save_bertopic_model(model, "models/bertopic_latest")
        print("Topic modeling enabled")
    else:
        print("Topic modeling training failed")
else:
    print("Skipping topic modeling (not required)")
```

### With Embeddings

```python
from embedding import Embeddings
from topic_model import train_bertopic

# Use existing embedding model for speed
emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")

# Train BERTopic with pre-computed embeddings
model = train_bertopic(
    abstracts,
    embedding_model=emb.model,  # Reuse SentenceTransformer
    min_topic_size=10
)

# This is faster than BERTopic computing embeddings itself
```

## Performance

### Training Time

| Corpus Size | Time (CPU) | Time (GPU) |
|-------------|-----------|------------|
| 100 papers | ~30 sec | ~10 sec |
| 500 papers | ~2 min | ~30 sec |
| 1000 papers | ~5 min | ~1 min |

### Memory Usage

- Model: ~500 MB
- Training: 1-2 GB peak
- Inference: ~200 MB

### Recommendations

- **Minimum papers**: 50 (preferably 100+)
- **Minimum topic size**: 10 papers
- **Target topics**: 10-30 for most corpora

## Troubleshooting

### Issue: "BERTopic not available"

**Cause**: Package not installed or installation failed

**Solution**:
```powershell
# Try installing
pip install bertopic umap-learn hdbscan

# If hdbscan fails on Windows, you need C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# OR: Just continue without it
# The system works perfectly without BERTopic!
```

### Issue: "hdbscan installation fails"

**Cause**: Missing C++ compiler on Windows

**Solution**:
1. **Option 1** (Recommended): Skip topic modeling
   - The system works without it
   - Use TF-IDF and embeddings instead

2. **Option 2**: Install C++ Build Tools
   - Download Visual Studio Build Tools
   - Select "C++ build tools" workload
   - Restart and try again

3. **Option 3**: Use pre-built wheels
   ```powershell
   # Try installing from conda
   conda install -c conda-forge hdbscan
   ```

### Issue: "Not enough papers for topics"

**Cause**: Fewer than 50 papers with abstracts

**Solution**:
- Add more papers to the database
- Or skip topic modeling (not required)
- Use TF-IDF/embeddings for search

## Comparison with Other Search Methods

| Method | Setup | Speed | Recall | Use Case |
|--------|-------|-------|--------|----------|
| **TF-IDF** | Easy | Fast | Good | Keyword matching |
| **Embeddings** | Medium | Medium | Excellent | Semantic search |
| **Topics** | Hard | Slow | Good | Theme discovery |

**Recommendation**: Use TF-IDF + Embeddings for search, optionally add Topics for insights

## Architecture

```
topic_model.py (OPTIONAL)
├── Graceful Import Guards
│   ├── try: import bertopic
│   ├── try: import umap
│   └── try: import hdbscan
│
├── Train BERTopic
│   ├── Use provided embeddings
│   ├── UMAP dimensionality reduction
│   └── HDBSCAN clustering
│
├── Author Topic Profile
│   ├── Get author's papers
│   ├── Extract abstracts
│   ├── Compute topic distribution
│   └── Return top-N topics
│
└── Topic Overlap Score
    ├── Cosine similarity (weighted)
    └── Jaccard similarity (binary)
```

## Dependencies

**Required** (already installed):
- numpy
- scipy
- scikit-learn

**Optional** (for topic modeling):
- bertopic
- umap-learn
- hdbscan (requires C++ compiler)

## Related Files

- `backend/embedding.py` - Provides embedding models for BERTopic
- `backend/tfidf_engine.py` - Alternative keyword search
- `backend/db_utils.py` - Database for papers and authors

## Examples

### Full Pipeline with Optional Topics

```python
from pathlib import Path
from db_utils import get_all_papers
from tfidf_engine import TFIDFEngine
from embedding import Embeddings, build_faiss_index
from topic_model import is_available, train_bertopic, author_topic_profile

db_path = Path("papers.db")

# 1. Load papers (always works)
papers = get_all_papers(db_path)
corpus = [p['title'] + ' ' + p['abstract'] for p in papers]
paper_ids = [p['id'] for p in papers]

# 2. Build TF-IDF index (always works)
tfidf = TFIDFEngine()
tfidf.fit(corpus, paper_ids)

# 3. Build semantic index (always works)
emb = Embeddings()
vectors = emb.encode_texts(corpus, normalize=True)
faiss_index = build_faiss_index(vectors, dim=emb.dim)

# 4. Optionally train topics
topic_model = None
if is_available():
    abstracts = [p['abstract'] for p in papers if p['abstract']]
    if len(abstracts) >= 50:
        topic_model = train_bertopic(abstracts, embedding_model=emb.model)

# 5. Search (always works)
query = "deep learning neural networks"
results = tfidf.most_similar(query, topn=10)

# 6. Optionally boost with topics
if topic_model:
    print("Enhancing results with topics...")
    # ... topic-based re-ranking
else:
    print("Using keyword + semantic search")

print(f"Top-10 results: {results}")
```

## Acceptance Criteria

✅ **Can be skipped without breaking the pipeline**
- All imports wrapped in try/except
- Functions return None if not available
- Main search still works without topics

✅ **Provides value when available**
- Discovers topics in corpus
- Creates author expertise profiles
- Enables topic-based similarity

✅ **Graceful degradation**
- Clear logging when not available
- No crashes or errors
- Fallback to other search methods

## Conclusion

Topic modeling with BERTopic is a **powerful but optional** enhancement. The system is designed to work perfectly without it, using TF-IDF and semantic embeddings for search. Only add topic modeling if:

1. You have 50+ papers with abstracts
2. You want to discover research themes
3. You can install the dependencies (including C++ compiler)

Otherwise, skip it and use the robust TF-IDF + embeddings pipeline!

# Sentence Embeddings and FAISS Documentation

## Overview

The `embedding.py` module provides **dense semantic embeddings** using state-of-the-art transformer models (SciBERT, SPECTER) and **FAISS indexing** for fast similarity search. This enables semantic search that understands meaning beyond keyword matching.

## Key Features

- **Scientific Models**: SciBERT and SPECTER pre-trained on academic papers
- **GPU Acceleration**: Automatic device detection via `utils.device()`
- **Batched Encoding**: Process large corpora efficiently with progress bars
- **Float32 Output**: FAISS-compatible embedding format
- **NaN Handling**: Robust error handling for edge cases
- **L2 Normalization**: Enables cosine similarity via inner product
- **FAISS IndexFlatIP**: Fast exact search with inner product metric
- **Persistent Storage**: Save/load indexes and ID mappings

## Architecture

```
Embeddings Module
├── Embeddings Class
│   ├── SentenceTransformer (mean pooling)
│   ├── Device detection (cuda/cpu)
│   └── Batched encoding with progress bar
│
├── FAISS Index Building
│   ├── L2 normalization
│   ├── IndexFlatIP (inner product)
│   └── ID mapping (row → paper_id)
│
└── Persistence
    ├── save_index() → .index file
    ├── load_index() ← .index file
    └── id_map.npy (paper IDs)
```

## Quick Start

### Basic Usage

```python
from embedding import Embeddings, build_faiss_index, save_index, search_index

# 1. Initialize embeddings
emb = Embeddings(
    model_name="allenai/scibert_scivocab_uncased",
    batch_size=8
)

# 2. Encode texts
texts = [
    "Deep learning for natural language processing",
    "Computer vision with convolutional networks",
    "Reinforcement learning agents"
]
vectors = emb.encode_texts(texts, normalize=True)

# 3. Build FAISS index
index = build_faiss_index(vectors, dim=emb.dim, metric="ip")

# 4. Search
query_vec = emb.encode_texts(["neural networks"], normalize=True)
scores, indices = index.search(query_vec, k=3)

print(f"Top-3 similar documents: {indices[0]}")
print(f"Similarity scores: {scores[0]}")
```

### With Database Integration

```python
from pathlib import Path
from db_utils import get_all_papers
from embedding import Embeddings, build_faiss_index, save_index, search_index
import numpy as np

# Load papers from database
papers = get_all_papers(Path("papers.db"))

# Prepare corpus
corpus_texts = []
paper_ids = []

for paper in papers:
    text = f"{paper['title']} {paper['abstract']}"
    corpus_texts.append(text)
    paper_ids.append(paper['id'])

# Initialize embeddings
emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")

# Encode corpus
vectors = emb.encode_texts(corpus_texts, normalize=True)

# Build and save FAISS index
index = build_faiss_index(vectors, dim=emb.dim, metric="ip")
save_index(index, "models/faiss_scibert", paper_ids)

# Later: Search
from embedding import load_index

index, id_map = load_index("models/faiss_scibert")
query_vec = emb.encode_texts(["transformer attention"], normalize=True)
scores, paper_ids_result = search_index(index, query_vec, k=10, id_map=id_map)

print(f"Top-10 papers: {paper_ids_result[0]}")
print(f"Scores: {scores[0]}")
```

## API Reference

### `Embeddings` Class

#### `__init__(model_name, batch_size, device)`

Initialize sentence embeddings model.

**Parameters:**
- `model_name` (str): Hugging Face model identifier
  - `"allenai/scibert_scivocab_uncased"` (768 dim, scientific papers) ✓ Recommended
  - `"allenai/specter"` (768 dim, paper-level embeddings) ✓ Recommended
  - `"sentence-transformers/all-MiniLM-L6-v2"` (384 dim, general purpose, fast)
  - `"sentence-transformers/all-mpnet-base-v2"` (768 dim, general purpose, best quality)
- `batch_size` (int): Batch size for encoding (default: 8)
- `device` (str, optional): Device for computation. If None, uses `utils.device()`

**Example:**
```python
# Use SciBERT (best for scientific papers)
emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")

# Use smaller model for speed
emb = Embeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=16
)

# Force CPU
emb = Embeddings(device="cpu")
```

---

#### `encode_texts(texts, show_progress_bar, normalize)`

Encode texts to embeddings with batching and progress bar.

**Parameters:**
- `texts` (List[str]): List of text strings to encode
- `show_progress_bar` (bool): Show tqdm progress bar (default: True)
- `normalize` (bool): L2-normalize embeddings (default: False, recommended: True)

**Returns:**
- `np.ndarray`: Array of shape (len(texts), dim) with dtype float32

**Raises:**
- `ValueError`: If texts is empty

**Example:**
```python
texts = ["text 1", "text 2", "text 3"]

# Basic encoding
vectors = emb.encode_texts(texts)
print(vectors.shape)  # (3, 768)

# With normalization (for cosine similarity)
vectors_norm = emb.encode_texts(texts, normalize=True)

# Large batch without progress bar
vectors = emb.encode_texts(large_corpus, show_progress_bar=False)
```

**Important Notes:**
- Always use `normalize=True` when building FAISS index with `metric="ip"`
- Normalized vectors have L2 norm = 1.0
- NaN values are automatically replaced with 0.0

---

### `build_faiss_index(embeddings, dim, metric)`

Build FAISS index for fast similarity search.

**Parameters:**
- `embeddings` (np.ndarray): Embeddings array of shape (n, dim)
- `dim` (int): Embedding dimension (must match embeddings.shape[1])
- `metric` (str): Similarity metric
  - `"ip"`: Inner product (requires L2-normalized embeddings) ✓ Recommended
  - `"cosine"`: Alias for "ip" with automatic normalization
  - `"l2"`: Euclidean distance

**Returns:**
- `faiss.Index`: FAISS index ready for search

**Raises:**
- `ImportError`: If faiss not installed
- `ValueError`: If embeddings empty or dimension mismatch

**Example:**
```python
# Build index with inner product (cosine similarity)
index = build_faiss_index(vectors, dim=768, metric="ip")

# Build index with L2 distance
index = build_faiss_index(vectors, dim=768, metric="l2")

# Automatic normalization
index = build_faiss_index(vectors, dim=768, metric="cosine")
```

**Performance:**
- IndexFlatIP: Exact search, no compression
- Search time: O(n) where n = corpus size
- Memory: 4 * n * dim bytes (float32)
- For 10K papers with 768 dim: ~30 MB

---

### `save_index(index, path, paper_ids)`

Save FAISS index and optional ID mapping to disk.

**Parameters:**
- `index` (faiss.Index): FAISS index to save
- `path` (str | Path): Path to save index (will add .index extension)
- `paper_ids` (List[int], optional): Paper IDs corresponding to index rows

**Saves:**
- `{path}.index`: FAISS index file
- `{path}_id_map.npy`: ID mapping (if paper_ids provided)

**Example:**
```python
# Save index and ID mapping
save_index(index, "models/scibert_v1", paper_ids=[101, 102, 103])

# Creates:
#   models/scibert_v1.index
#   models/scibert_v1_id_map.npy

# Save index only
save_index(index, "models/temp_index")
```

---

### `load_index(path, load_id_map)`

Load FAISS index and optional ID mapping from disk.

**Parameters:**
- `path` (str | Path): Path to index file (without .index extension)
- `load_id_map` (bool): Whether to load ID mapping (default: True)

**Returns:**
- If `load_id_map=False`: `faiss.Index`
- If `load_id_map=True`: `Tuple[faiss.Index, np.ndarray]`

**Raises:**
- `FileNotFoundError`: If index file not found

**Example:**
```python
# Load index and ID mapping
index, id_map = load_index("models/scibert_v1")

# Load index only
index = load_index("models/scibert_v1", load_id_map=False)
```

---

### `search_index(index, query_vectors, k, id_map)`

Search FAISS index for top-k similar vectors.

**Parameters:**
- `index` (faiss.Index): FAISS index
- `query_vectors` (np.ndarray): Query embeddings of shape (n_queries, dim)
- `k` (int): Number of results per query (default: 10)
- `id_map` (np.ndarray, optional): ID mapping to convert indices to paper IDs

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`:
  - `scores`: Array of shape (n_queries, k) with similarity scores
  - `ids`: Array of shape (n_queries, k) with paper IDs or indices

**Example:**
```python
# Search with ID mapping
index, id_map = load_index("models/scibert_v1")
query_vec = emb.encode_texts(["transformer"], normalize=True)
scores, paper_ids = search_index(index, query_vec, k=10, id_map=id_map)

# Multiple queries
queries = ["query 1", "query 2"]
query_vecs = emb.encode_texts(queries, normalize=True)
scores, paper_ids = search_index(index, query_vecs, k=5, id_map=id_map)

print(scores.shape)  # (2, 5)
print(paper_ids.shape)  # (2, 5)
```

---

## Model Comparison

### Recommended Models

| Model | Dim | Domain | Speed | Quality | Use Case |
|-------|-----|--------|-------|---------|----------|
| **allenai/scibert_scivocab_uncased** | 768 | Scientific | Medium | ★★★★★ | Academic papers |
| **allenai/specter** | 768 | Scientific | Medium | ★★★★★ | Paper-level embeddings |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | General | Fast | ★★★★ | Speed over quality |
| sentence-transformers/all-mpnet-base-v2 | 768 | General | Medium | ★★★★★ | General documents |

### Model Details

#### SciBERT (`allenai/scibert_scivocab_uncased`)
- **Best for**: Scientific papers, technical documents
- **Pre-trained on**: 1.14M scientific papers
- **Vocabulary**: Scientific terms optimized
- **Use when**: Working with academic papers, preprints, technical reports

#### SPECTER (`allenai/specter`)
- **Best for**: Paper-level embeddings, citation prediction
- **Pre-trained on**: S2ORC corpus with citation links
- **Special**: Trained on paper titles + abstracts
- **Use when**: Need paper similarity, citation recommendation

#### all-MiniLM-L6-v2 (`sentence-transformers/all-MiniLM-L6-v2`)
- **Best for**: Fast prototyping, limited resources
- **Pre-trained on**: General web text
- **Advantage**: 2x faster, 50% smaller embeddings
- **Use when**: Speed is critical, general documents

---

## Performance Benchmarks

### Encoding Speed

| Model | Batch Size | Documents/sec (CPU) | Documents/sec (GPU) |
|-------|------------|---------------------|---------------------|
| SciBERT | 8 | ~20 | ~200 |
| SPECTER | 8 | ~20 | ~200 |
| MiniLM | 16 | ~40 | ~400 |

### FAISS Search Speed

| Corpus Size | Search Time (k=10) | Memory Usage |
|-------------|-------------------|--------------|
| 1K papers | 0.5 ms | 3 MB |
| 10K papers | 2 ms | 30 MB |
| 100K papers | 20 ms | 300 MB |
| 1M papers | 200 ms | 3 GB |

### Memory Requirements

**Model Loading:**
- SciBERT/SPECTER: ~450 MB
- MiniLM: ~90 MB

**Embeddings Storage (float32):**
- 1K papers × 768 dim: 3 MB
- 10K papers × 768 dim: 30 MB
- 100K papers × 768 dim: 300 MB

---

## Integration Examples

### Hybrid Search (TF-IDF + Embeddings)

```python
from tfidf_engine import TFIDFEngine
from embedding import Embeddings, build_faiss_index, search_index

# Load corpus
papers = get_all_papers(db_path)
corpus_texts = [p['title'] + ' ' + p['abstract'] for p in papers]
paper_ids = [p['id'] for p in papers]

# Build TF-IDF index
tfidf_engine = TFIDFEngine()
tfidf_engine.fit(corpus_texts, paper_ids)

# Build semantic index
emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")
vectors = emb.encode_texts(corpus_texts, normalize=True)
faiss_index = build_faiss_index(vectors, dim=emb.dim)

# Hybrid search function
def hybrid_search(query, k=10, alpha=0.5):
    """
    Combine TF-IDF and semantic search.
    
    alpha: Weight for TF-IDF (1-alpha for semantic)
    """
    # TF-IDF results
    tfidf_results = tfidf_engine.most_similar(query, topn=k*2)
    tfidf_scores = {pid: score for pid, score in tfidf_results}
    
    # Semantic results
    query_vec = emb.encode_texts([query], normalize=True)
    sem_scores, sem_ids = search_index(faiss_index, query_vec, k=k*2, id_map=np.array(paper_ids))
    semantic_scores = {pid: score for pid, score in zip(sem_ids[0], sem_scores[0])}
    
    # Combine scores
    all_paper_ids = set(tfidf_scores.keys()) | set(semantic_scores.keys())
    combined = []
    
    for pid in all_paper_ids:
        tfidf_score = tfidf_scores.get(pid, 0)
        sem_score = semantic_scores.get(pid, 0)
        final_score = alpha * tfidf_score + (1 - alpha) * sem_score
        combined.append((pid, final_score))
    
    # Sort and return top-k
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]

# Search
results = hybrid_search("transformer attention mechanisms", k=10, alpha=0.3)
```

### Batch Processing Large Corpus

```python
from embedding import Embeddings
import numpy as np

# Initialize
emb = Embeddings(model_name="allenai/scibert_scivocab_uncased", batch_size=32)

# Load large corpus
papers = get_all_papers(db_path)  # e.g., 100K papers
corpus_texts = [p['title'] + ' ' + p['abstract'] for p in papers]

# Process in chunks to manage memory
chunk_size = 10000
all_vectors = []

for i in range(0, len(corpus_texts), chunk_size):
    chunk = corpus_texts[i:i+chunk_size]
    print(f"Processing chunk {i//chunk_size + 1}/{(len(corpus_texts)-1)//chunk_size + 1}")
    
    vectors = emb.encode_texts(chunk, normalize=True)
    all_vectors.append(vectors)

# Concatenate
final_vectors = np.vstack(all_vectors)
print(f"Total vectors: {final_vectors.shape}")

# Build index
index = build_faiss_index(final_vectors, dim=emb.dim)
save_index(index, "models/large_corpus", paper_ids=[p['id'] for p in papers])
```

---

## Advanced Topics

### GPU Acceleration

```python
from utils import device

# Check GPU availability
dev = device()
print(f"Using device: {dev}")

# Initialize with GPU
emb = Embeddings(
    model_name="allenai/scibert_scivocab_uncased",
    device=dev,
    batch_size=32  # Larger batch for GPU
)

# GPU speedup: ~10x faster than CPU
```

### Error Handling

```python
from embedding import Embeddings, build_faiss_index

try:
    emb = Embeddings(model_name="invalid/model")
except RuntimeError as e:
    print(f"Model loading failed: {e}")

try:
    vectors = emb.encode_texts([])
except ValueError as e:
    print(f"Empty texts: {e}")

try:
    index = build_faiss_index(vectors, dim=512)  # Wrong dim
except ValueError as e:
    print(f"Dimension mismatch: {e}")
```

### Normalization Best Practices

```python
# ALWAYS normalize when using metric="ip"
vectors = emb.encode_texts(texts, normalize=True)
index = build_faiss_index(vectors, dim=emb.dim, metric="ip")

# Verify normalization
norms = np.linalg.norm(vectors, axis=1)
assert np.allclose(norms, 1.0), "Not normalized!"

# Query vectors must also be normalized
query_vec = emb.encode_texts(["query"], normalize=True)
scores, indices = index.search(query_vec, k=10)

# Scores are now cosine similarities in [-1, 1]
# Higher = more similar
```

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size
```python
emb = Embeddings(batch_size=4)  # Smaller batch
```

### Issue: Model download fails

**Solution**: Check internet connection or use cached model
```python
from sentence_transformers import SentenceTransformer

# Download manually first
model = SentenceTransformer("allenai/scibert_scivocab_uncased")
```

### Issue: Slow encoding on CPU

**Solution**: Use smaller model or GPU
```python
# Option 1: Smaller model
emb = Embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Option 2: Use GPU
emb = Embeddings(device="cuda")
```

### Issue: FAISS index search returns wrong results

**Solution**: Ensure query vectors are normalized
```python
# Always normalize both corpus and queries
corpus_vectors = emb.encode_texts(corpus, normalize=True)
query_vec = emb.encode_texts(["query"], normalize=True)
```

---

## Dependencies

- `sentence-transformers >= 2.2.2`: Model loading and encoding
- `faiss-cpu >= 1.7.4` or `faiss-gpu`: Fast similarity search
- `numpy >= 1.24.0`: Array operations
- `tqdm >= 4.65.0`: Progress bars
- `torch >= 2.0.0`: PyTorch backend

Install with:
```powershell
pip install sentence-transformers faiss-cpu tqdm
```

---

## Related Files

- `backend/utils.py`: Device detection (`device()` function)
- `backend/tfidf_engine.py`: TF-IDF similarity (complementary to embeddings)
- `backend/db_utils.py`: Database layer for paper storage

---

## References

- [SentenceTransformers Documentation](https://www.sbert.net/)
- [SciBERT Paper](https://arxiv.org/abs/1903.10676)
- [SPECTER Paper](https://arxiv.org/abs/2004.07180)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

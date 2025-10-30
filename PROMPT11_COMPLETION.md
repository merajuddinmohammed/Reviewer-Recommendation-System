# Prompt 11 Completion - build_tfidf.py

## ✅ Implementation Complete

### Files Created
1. **`backend/build_tfidf.py`** - CLI for building TF-IDF model (528 lines)
2. **Output**: `models/tfidf_vectorizer.pkl` - Trained TF-IDF model with corpus matrix and ID mapping

### Features Implemented

#### 1. **Command-Line Arguments**
```bash
--db            Path to database (default: data/papers.db)
--out           Path to save model (default: models/tfidf_vectorizer.pkl)
--min-df        Minimum document frequency (default: 2)
--max-df        Maximum document frequency (default: 0.85)
--max-features  Maximum vocabulary size (default: 50000)
--ngram-min     Minimum n-gram size (default: 1)
--ngram-max     Maximum n-gram size (default: 2)
--force         Force rebuild
--verbose       Debug logging
```

#### 2. **Text Extraction** (Same as build_vectors.py)
For each paper:
1. **Abstract** (if exists and > 50 chars) ← **Preferred**
2. **Fulltext** (first 2000 chars if exists and > 50 chars)
3. **Title** (fallback)
4. **Skip** (log and skip if no text)

#### 3. **Document Frequency Filters**

**`--min-df` (Minimum Document Frequency)**
- Ignores terms appearing in fewer than N documents
- Default: 2 (must appear in at least 2 documents)
- Effect: Removes rare/typo terms
- Higher value = smaller vocabulary, faster search

**`--max-df` (Maximum Document Frequency)**  
- Ignores terms appearing in more than X% of documents
- Default: 0.85 (must appear in at most 85% of documents)
- Effect: Removes common stopwords
- Lower value = fewer generic terms

#### 4. **Top 30 Terms by IDF**

Shows most discriminative terms (highest IDF):
```
Top 30 terms by IDF (most discriminative):
  1. dharwad                        (IDF: 3.5123)
  2. karnataka                      (IDF: 3.5123)
  3. series evolving                (IDF: 3.5123)
  ...
```

High IDF = appears in fewer documents = more discriminative for search.

#### 5. **Output Statistics**

Prints comprehensive statistics:
- **Documents processed**: 36
- **Documents skipped**: 0
- **Vocabulary size**: 863 unique terms
- **Feature count**: 863 (unigrams + bigrams)
- **Matrix sparsity**: 0.9068 (90.68% zeros - very efficient)
- **Non-zero elements**: 2,897

#### 6. **Saved Model Contents**

The `.pkl` file contains:
```python
{
    'vectorizer': TfidfVectorizer,      # Fitted sklearn vectorizer
    'corpus_matrix': csr_matrix,        # Sparse TF-IDF matrix (36 x 863)
    'paper_ids': List[int],             # Paper ID mapping [1, 2, 3, ..., 36]
    'max_features': 50000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.85
}
```

### Usage Examples

#### Basic Usage
```bash
cd backend
python build_tfidf.py
```

#### Custom Document Frequency Filters
```bash
# More aggressive filtering
python build_tfidf.py --min-df 3 --max-df 0.7

# Less aggressive (keep more terms)
python build_tfidf.py --min-df 1 --max-df 0.95
```

#### Custom Vocabulary Size
```bash
# Smaller vocabulary (faster, less precise)
python build_tfidf.py --max-features 10000

# Larger vocabulary (slower, more precise)
python build_tfidf.py --max-features 100000
```

#### Custom N-gram Range
```bash
# Unigrams only
python build_tfidf.py --ngram-min 1 --ngram-max 1

# Unigrams + bigrams + trigrams
python build_tfidf.py --ngram-min 1 --ngram-max 3
```

#### Force Rebuild
```bash
python build_tfidf.py --force
```

#### All Custom Parameters
```bash
python build_tfidf.py \
  --db data/papers.db \
  --out models/custom_tfidf.pkl \
  --min-df 3 \
  --max-df 0.9 \
  --max-features 30000 \
  --ngram-min 1 \
  --ngram-max 2 \
  --verbose
```

### Test Results

#### Build Output (Actual Run)
```
Database:     data/papers.db
Output file:  models/tfidf_vectorizer.pkl
Min DF:       2
Max DF:       0.85
Max features: 50000
N-gram range: (1, 2)

✓ TF-IDF model built successfully

Documents processed: 36
Vocabulary size:     863
Matrix sparsity:     0.9068
Non-zero elements:   2,897
```

#### Top IDF Terms (Sample)
Most discriminative terms found:
- Geographic: `dharwad`, `karnataka`, `india`
- Technical: `series evolving`, `stable`, `database`, `metric`
- Domain-specific: `minstab`, `ners`, `characteristics`

High IDF indicates these terms are rare and useful for distinguishing papers.

### Testing

After build completes, test with:

```python
from tfidf_engine import TFIDFEngine

# Load model
engine = TFIDFEngine.load('models/tfidf_vectorizer.pkl')

# Check loaded data
print(f"Documents: {engine.corpus_matrix.shape[0]}")
print(f"Features: {engine.corpus_matrix.shape[1]}")
print(f"Paper IDs: {engine.paper_ids[:5]}")

# Test search
results = engine.most_similar("machine learning", topn=5)
print(f"Found {len(results)} similar papers")
```

### Technical Details

#### Sparse Matrix Efficiency
- **Sparsity**: 90.68% (most matrix entries are zero)
- **Memory**: Only stores non-zero values (2,897 instead of 31,068)
- **Speed**: Fast cosine similarity with sparse operations

#### Vocabulary Creation
1. Extract all terms from corpus
2. Filter by `min_df` (remove rare terms)
3. Filter by `max_df` (remove common terms)
4. Limit to `max_features` (keep top terms by document frequency)
5. Build vocabulary: 863 terms (717 unigrams + 146 bigrams)

#### IDF Calculation
```
IDF(term) = log(N / df(term))
where:
  N = total documents (36)
  df(term) = number of documents containing term
```

Higher IDF = more discriminative (appears in fewer documents).

### Error Handling

- ✅ Database not found → Exit with error
- ✅ No papers in database → Raise ValueError
- ✅ No valid texts → Raise ValueError
- ✅ Invalid parameters → Exit with validation error
- ✅ Model exists → Prompt user (unless `--force`)
- ✅ Save failure → Log and raise exception

### Logging

- **Console**: INFO level (default)
- **Log file**: `build_tfidf.log` (all levels)
- **Verbose mode**: DEBUG level with `--verbose` flag

### Acceptance Criteria Met

✅ **CLI with --db and --out**: Both arguments implemented  
✅ **--min-df and --max-df flags**: Configurable document frequency filters  
✅ **Train on cleaned texts**: Same text extraction as build_vectors.py  
✅ **Persist mapping to paper_ids**: Saved in corpus with ID alignment  
✅ **Output counts**: Documents, vocabulary, features, sparsity  
✅ **Top 30 terms by IDF**: Printed to console  
✅ **Works on ingested DB**: Successfully processed 36 papers  
✅ **Creates .pkl file**: `models/tfidf_vectorizer.pkl` created  

---

## 📊 Both Scripts Complete!

### Build Status

| Script | Status | Output | Papers | Features |
|--------|--------|--------|--------|----------|
| **build_vectors.py** | ✅ COMPLETE | `data/faiss_index.faiss` | 36 | 768-dim vectors |
| **build_tfidf.py** | ✅ COMPLETE | `models/tfidf_vectorizer.pkl` | 36 | 863 terms |

### Next Steps

1. **Run full ingestion** to process all 536 PDFs:
   ```bash
   cd backend
   python ingest.py
   ```

2. **Rebuild indices with full corpus**:
   ```bash
   python build_vectors.py --force
   python build_tfidf.py --force
   ```

3. **Implement search API** combining both:
   - TF-IDF for keyword search
   - FAISS for semantic search
   - Hybrid fusion for best results

---

**Status**: ✅ COMPLETE  
**Files**: `backend/build_tfidf.py` (528 lines)  
**Dependencies**: sklearn, scipy, joblib (already installed)  
**Test**: Successfully built TF-IDF model with 36 papers

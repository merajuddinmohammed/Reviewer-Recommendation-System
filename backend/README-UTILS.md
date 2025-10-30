# Text Cleaning & Utility Functions - Quick Reference

## Overview

The `utils.py` module provides pure, documented functions for common text processing, temporal weighting, and device management tasks.

## Installation

All functions are in `backend/utils.py` and can be imported directly:

```python
from utils import clean_text, split_abstract_fulltext, recency_weight, device
```

## Functions

### 1. `clean_text(s: str) -> str`

Clean and normalize text for NLP processing.

**Operations:**
- Convert to lowercase
- Normalize whitespace
- Remove control characters
- Preserve mathematical symbols (Greek letters, operators, subscripts)

**Usage:**
```python
>>> from utils import clean_text

>>> clean_text("Hello   World\n\t!")
'hello world !'

>>> clean_text("Deep\xa0Learning")
'deep learning'

>>> clean_text("α-helix and β-sheet")
'α-helix and β-sheet'

>>> clean_text("x₁ + x₂ = y")
'x₁ + x₂ = y'
```

---

### 2. `split_abstract_fulltext(text: str) -> Tuple[str, str]`

Split paper text into abstract and fulltext sections heuristically.

**Strategy:**
1. Find "Abstract" header
2. Extract 300-1500 words after abstract
3. Find "Introduction" or first numbered section
4. Return (abstract, fulltext)

**Usage:**
```python
>>> from utils import split_abstract_fulltext

>>> paper = """
... Title Here
... 
... Abstract: This is the abstract section with important summary.
... 
... 1. Introduction
... 
... This is the introduction and main content.
... """

>>> abstract, fulltext = split_abstract_fulltext(paper)
>>> print(f"Abstract: {len(abstract)} chars")
Abstract: 54 chars
>>> print(f"Fulltext: {len(fulltext)} chars")
Fulltext: 47 chars
```

---

### 3. `recency_weight(pub_year: int, ref_year: Optional[int] = None, tau: float = 3.0) -> float`

Compute temporal recency weight using exponential decay.

**Formula:** `exp(-age / tau)` where `age = ref_year - pub_year`

**Parameters:**
- `pub_year`: Publication year
- `ref_year`: Reference year (default: current year)
- `tau`: Time constant in years (smaller = faster decay)

**Usage:**
```python
>>> from utils import recency_weight

>>> # Same year = weight 1.0
>>> recency_weight(2023, 2023)
1.0

>>> # 3 years old with tau=3.0 ≈ 0.37
>>> recency_weight(2020, 2023, tau=3.0)
0.36787944117144233

>>> # 6 years old has lower weight
>>> recency_weight(2017, 2023, tau=3.0)
0.1353352832366127

>>> # Smaller tau = faster decay
>>> recency_weight(2020, 2023, tau=1.0)
0.049787068367863944

>>> # Larger tau = slower decay
>>> recency_weight(2020, 2023, tau=5.0)
0.5488116360940264
```

**Use Cases:**
- Weight recent papers higher in recommendations
- Time-based ranking adjustments
- Temporal relevance scoring

---

### 4. `device() -> str`

Get the best available device for PyTorch operations.

**Returns:** `'cuda'` if available, otherwise `'cpu'`

**Usage:**
```python
>>> from utils import device

>>> dev = device()
>>> print(f"Using device: {dev}")
Using device: cpu

>>> # Use in PyTorch
>>> import torch
>>> tensor = torch.randn(10, 10).to(device())
```

---

### 5. `get_device_info() -> dict`

Get detailed information about compute devices.

**Usage:**
```python
>>> from utils import get_device_info

>>> info = get_device_info()
>>> print(info)
{
    'torch_available': True,
    'device': 'cuda',
    'cuda_available': True,
    'cuda_device_count': 1,
    'cuda_device_name': 'NVIDIA GeForce RTX 3080',
    'cuda_version': '11.8'
}
```

---

### 6. `normalize_year(year: Optional[int], min_year: int = 1900, max_year: int = 2030) -> Optional[int]`

Validate and normalize publication year.

**Usage:**
```python
>>> from utils import normalize_year

>>> normalize_year(2023)
2023

>>> normalize_year(1850)  # Too old
None

>>> normalize_year(2100)  # Too far in future
None

>>> normalize_year(None)
None

>>> normalize_year(1985, min_year=1990)
None
```

---

### 7. `truncate_text(text: str, max_words: int = 512, suffix: str = "...") -> str`

Truncate text to maximum word count.

**Usage:**
```python
>>> from utils import truncate_text

>>> long_text = " ".join([f"word{i}" for i in range(1000)])
>>> truncated = truncate_text(long_text, max_words=10)
>>> print(truncated)
word0 word1 word2 word3 word4 word5 word6 word7 word8 word9...

>>> # Short text unchanged
>>> truncate_text("short text", max_words=100)
'short text'

>>> # Custom suffix
>>> truncate_text("one two three four five", max_words=3, suffix="[...]")
'one two three[...]'
```

---

### 8. `word_count(text: str) -> int`

Count words in text.

**Usage:**
```python
>>> from utils import word_count

>>> word_count("Hello world")
2

>>> word_count("one two three four")
4

>>> word_count("")
0
```

---

## Integration Examples

### Example 1: Clean and Process Paper Text

```python
from utils import clean_text, split_abstract_fulltext, word_count

# Load paper text
with open("paper.txt") as f:
    raw_text = f.read()

# Clean text
cleaned = clean_text(raw_text)

# Split into sections
abstract, fulltext = split_abstract_fulltext(cleaned)

print(f"Abstract: {word_count(abstract)} words")
print(f"Fulltext: {word_count(fulltext)} words")
```

### Example 2: Time-Weighted Paper Ranking

```python
from utils import recency_weight
import db_utils

# Get papers from database
papers = db_utils.get_all_papers("papers.db")

# Add recency weights
for paper in papers:
    if paper['year']:
        paper['recency_weight'] = recency_weight(paper['year'], tau=3.0)
    else:
        paper['recency_weight'] = 0.5  # Default for unknown year

# Sort by recency
papers.sort(key=lambda p: p['recency_weight'], reverse=True)

# Display top 10 most recent
for paper in papers[:10]:
    print(f"{paper['year']}: {paper['title']} (weight: {paper['recency_weight']:.3f})")
```

### Example 3: Device-Aware ML Pipeline

```python
from utils import device, get_device_info
import torch

# Check device
info = get_device_info()
print(f"Running on: {info['device']}")

if info['cuda_available']:
    print(f"GPU: {info['cuda_device_name']}")

# Use in model
dev = device()
model = MyModel().to(dev)
data = torch.tensor(data).to(dev)
output = model(data)
```

### Example 4: Text Preprocessing Pipeline

```python
from utils import clean_text, truncate_text, normalize_year

def preprocess_paper(title, abstract, year):
    """Preprocess paper metadata."""
    # Clean text
    title_clean = clean_text(title)
    abstract_clean = clean_text(abstract)
    
    # Truncate if too long
    abstract_truncated = truncate_text(abstract_clean, max_words=300)
    
    # Validate year
    year_valid = normalize_year(year)
    
    return {
        'title': title_clean,
        'abstract': abstract_truncated,
        'year': year_valid
    }

# Use it
paper = preprocess_paper(
    "Deep Learning for NLP",
    "This is a very long abstract...",
    2023
)
```

---

## Testing

All functions include comprehensive doctests:

```python
# Run all tests
python backend/utils.py
```

Expected output:
```
======================================================================
Running Utility Function Tests
======================================================================

[TEST 1] Running doctests...
✓ All 51 doctests passed

[TEST 2] Testing clean_text edge cases...
✓ clean_text handles edge cases

[TEST 3] Testing split_abstract_fulltext...
✓ split_abstract_fulltext works correctly

[TEST 4] Testing recency_weight...
✓ recency_weight behaves correctly

[TEST 5] Testing device detection...
✓ Device detected: cpu

[TEST 6] Testing normalize_year...
✓ normalize_year validates correctly

[TEST 7] Testing truncate_text...
✓ truncate_text works correctly

[TEST 8] Testing word_count...
✓ word_count works correctly

======================================================================
ALL UTILITY TESTS PASSED ✓
======================================================================
```

---

## Design Principles

### Pure Functions
All functions are pure (no side effects):
- Deterministic output for same input
- No global state modification
- Safe to use in parallel

### Safe Imports
Module can be imported even if dependencies are missing:
```python
# torch is optional
from utils import clean_text  # Works even without PyTorch
from utils import device       # Returns 'cpu' if torch unavailable
```

### Documented
Every function includes:
- Type hints
- Docstring with description
- Parameter documentation
- Return type documentation
- Usage examples (doctests)

### Tested
- 51 doctests covering all functions
- Edge case testing
- Integration tests

---

## Performance

| Function | Time Complexity | Notes |
|----------|----------------|-------|
| `clean_text` | O(n) | n = text length |
| `split_abstract_fulltext` | O(n) | n = text length |
| `recency_weight` | O(1) | Simple math |
| `device` | O(1) | Cached result |
| `normalize_year` | O(1) | Simple comparison |
| `truncate_text` | O(n) | n = text length |
| `word_count` | O(n) | n = text length |

All functions are efficient and suitable for batch processing.

---

## See Also

- `db_utils.py` - Database operations
- `parser.py` - PDF ingestion
- Main `README.md` - Project overview

---

**Version**: 1.0.0  
**Last Updated**: October 30, 2025

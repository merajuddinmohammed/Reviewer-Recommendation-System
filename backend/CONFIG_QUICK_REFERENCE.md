# Configuration Quick Reference Card

## Speed & Memory Knobs - Quick Guide

### 🚀 Quick Start

View current configuration:
```bash
cd backend
python -c "import config; config.print_config()"
```

### 📊 Configuration Categories

#### 1. RANKING PARAMETERS
```bash
export TOPK_RETURN=10      # Reviewers to return (5-50)
export N1_FAISS=200        # FAISS candidates (50-500)
export N2_TFIDF=200        # TF-IDF candidates (50-500)
```

#### 2. FEATURE ENGINEERING
```bash
export RECENCY_TAU=3.0     # Time decay years (1.0-5.0)
export W_S=0.55            # Semantic weight (0.0-1.0)
export W_L=0.25            # Lexical weight (0.0-1.0)
export W_R=0.20            # Recency weight (0.0-1.0)
```

#### 3. PERFORMANCE
```bash
export EMB_BATCH=4         # Embedding batch (1-32)
export DB_BATCH=1000       # Database batch (100-10000)
export FAISS_NPROBE=10     # FAISS clusters (1-100)
```

### 🎯 Common Profiles

#### Fast Mode (Resource-Constrained)
```bash
export TOPK_RETURN=5
export N1_FAISS=50
export N2_TFIDF=50
export EMB_BATCH=1
```
**Result**: ~5x faster, less comprehensive

#### Thorough Mode (High Quality)
```bash
export TOPK_RETURN=50
export N1_FAISS=500
export N2_TFIDF=500
export EMB_BATCH=16
```
**Result**: ~3x slower, better recall

#### Recent Work Focus
```bash
export RECENCY_TAU=1.5
export W_R=0.4
export W_S=0.4
export W_L=0.2
```
**Result**: Prefer papers from last 2 years

#### Keyword Matching Focus
```bash
export W_L=0.5
export W_S=0.3
export W_R=0.2
```
**Result**: Emphasize exact term matches

### 📈 Impact Summary

| Parameter | 2x Increase | Speed | Memory |
|-----------|-------------|-------|---------|
| N1_FAISS | 200→400 | -10% | +1.6MB |
| N2_TFIDF | 200→400 | -10% | Negligible |
| EMB_BATCH | 4→8 | +40% | +800MB |
| TOPK_RETURN | 10→20 | -5% | Negligible |

### 🔧 Validation

Check configuration:
```bash
python -c "import config; config.validate_config()"
```

Test imports:
```bash
python -c "import config; print(f'N1={config.N1_FAISS}, N2={config.N2_TFIDF}')"
```

### 📝 Environment Variables

Set in `.env` file:
```env
# .env
TOPK_RETURN=10
N1_FAISS=200
N2_TFIDF=200
W_S=0.55
W_L=0.25
W_R=0.20
EMB_BATCH=4
RECENCY_TAU=3.0
```

Load with python-dotenv:
```python
from dotenv import load_dotenv
load_dotenv()
import config
```

### 🐳 Docker Deployment

In `docker-compose.yml`:
```yaml
services:
  backend:
    environment:
      - TOPK_RETURN=10
      - N1_FAISS=200
      - N2_TFIDF=200
      - W_S=0.55
      - W_L=0.25
      - W_R=0.20
```

### ✅ Testing

Test with different values:
```bash
# Fast mode
export N1_FAISS=50; export N2_TFIDF=50
python app.py

# Thorough mode  
export N1_FAISS=500; export N2_TFIDF=500
python app.py
```

### 📚 Full Documentation

See `PROMPT22_COMPLETION.md` for:
- Detailed parameter descriptions
- Tuning guides with formulas
- Usage examples (5 scenarios)
- Troubleshooting guide
- Production deployment instructions

### 🎯 Key Files Modified

- ✅ `backend/config.py` - Centralized configuration (NEW)
- ✅ `backend/app.py` - Uses config weights
- ✅ `backend/ranker.py` - Uses config retrieval sizes
- ✅ `backend/build_vectors.py` - Uses config batch size

---

**Quick tip**: Start with defaults, then tune based on your needs!

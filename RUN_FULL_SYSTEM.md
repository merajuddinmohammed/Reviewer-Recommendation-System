# Running the Complete System - Full Dataset

This guide explains how to process the entire dataset and run the complete reviewer recommendation system.

## Prerequisites

✅ Virtual environment activated
✅ All dependencies installed (`pip install -r backend/requirements.txt`)
✅ Dataset folder at `../Dataset` with PDF files

## Overview

The complete system requires:
1. **Ingestion**: Extract metadata from PDFs → SQLite database
2. **TF-IDF**: Build TF-IDF vectorizer for lexical search
3. **Embeddings**: Build FAISS index for semantic search
4. **Training Data**: Generate labeled data for ranking
5. **LambdaRank**: Train LambdaRank model
6. **API Server**: FastAPI backend for recommendations
7. **Frontend**: React UI for user interaction

## Quick Start (Automated)

### Option 1: Run Full Pipeline Automatically

```powershell
# Terminal 1: Start ingestion
cd backend
python ingest.py --data_dir "..\Dataset"

# Terminal 2: Monitor and build models
cd backend
python build_pipeline.py
```

The pipeline script will:
- Wait for ingestion to complete
- Automatically build all models
- Report status at each step

## Manual Step-by-Step

If you prefer manual control:

### Step 1: Ingest PDFs

```powershell
cd backend
python ingest.py --data_dir "..\Dataset"
```

**Expected output**:
```
Found 538 PDF files
Ingesting PDFs: 100%|████████| 538/538
✓ Ingested 538 papers, 300+ authors
```

**Time**: ~20-40 minutes (depending on PDF complexity)

---

### Step 2: Build TF-IDF Vectorizer

```powershell
cd backend
python build_tfidf.py
```

**Expected output**:
```
✓ TF-IDF vectorizer built
✓ Saved to: models/tfidf_vectorizer.pkl
Vocabulary size: 15000+ terms
```

**Time**: ~1-2 minutes

---

### Step 3: Build FAISS Embeddings Index

```powershell
cd backend
python build_vectors.py --batch-size 4
```

**Expected output**:
```
Encoding papers: 100%|████████| 538/538
✓ FAISS index built
✓ Saved to: data/faiss_index.faiss
✓ ID map saved to: data/id_map.npy
```

**Time**: ~10-30 minutes (depends on CPU/GPU)

**Memory**: ~2-4 GB RAM

---

### Step 4: Generate Training Data

```powershell
cd backend
python build_training_data.py --n_train 5000 --n_val 1000
```

**Expected output**:
```
✓ Generated 5000 training queries
✓ Generated 1000 validation queries
✓ Saved to: data/train.parquet, data/val.parquet
```

**Time**: ~5-10 minutes

---

### Step 5: Train LambdaRank Model

```powershell
cd backend
python train_ranker.py --n_estimators 100 --max_depth 6
```

**Expected output**:
```
Training LambdaRank...
Validation nDCG@10: 0.65
✓ Model saved to: models/lgbm_ranker.pkl
```

**Time**: ~2-5 minutes

---

## Verify Models

Check that all models are built:

```powershell
cd backend
Get-ChildItem models, data -Include *.pkl, *.faiss, *.npy, *.parquet -Recurse | Select-Object Name, Length
```

**Expected files**:
```
tfidf_vectorizer.pkl    ~10 MB
faiss_index.faiss       ~200 MB
id_map.npy              ~5 KB
train.parquet           ~50 MB
lgbm_ranker.pkl         ~1 MB
```

---

## Run the System

### Start Backend API

```powershell
cd backend
python app.py
```

**Expected output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✓ Database: data\papers.db
✓ TF-IDF vectorizer loaded
✓ FAISS index loaded (538 vectors)
✓ LightGBM model loaded
✓ API ready for recommendations
```

**Access**: http://localhost:8000
**API Docs**: http://localhost:8000/docs

---

### Start Frontend

```powershell
# Terminal 2
cd frontend
npm run dev
```

**Access**: http://localhost:5173

---

## Test the System

### API Health Check

```powershell
curl http://localhost:8000/health
```

**Expected response**:
```json
{
  "status": "ready",
  "models_loaded": {
    "database": true,
    "tfidf_vectorizer": true,
    "faiss_index": true,
    "id_map": true,
    "lgbm_model": true
  }
}
```

---

### Test Recommendation

**Using PowerShell**:
```powershell
$body = @{
  title = "Deep Learning for Computer Vision"
  abstract = "This paper presents a novel approach to image classification using deep neural networks..."
  k = 10
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/recommend" -Method POST -Body $body -ContentType "application/json"
```

**Using Frontend**:
1. Open http://localhost:5173
2. Enter paper title and abstract
3. Click "Get Recommendations"
4. View ranked list of reviewers with evidence papers

---

## Configuration Tuning

Edit `backend/config.py` or set environment variables:

### Fast Mode (Resource-Constrained)

```powershell
$env:TOPK_RETURN = 5
$env:N1_FAISS = 50
$env:N2_TFIDF = 50
$env:EMB_BATCH = 1
python app.py
```

### Thorough Mode (High Quality)

```powershell
$env:TOPK_RETURN = 50
$env:N1_FAISS = 500
$env:N2_TFIDF = 500
$env:EMB_BATCH = 16
python app.py
```

### Prefer Recent Work

```powershell
$env:RECENCY_TAU = 1.5
$env:W_R = 0.4
$env:W_S = 0.4
$env:W_L = 0.2
python app.py
```

See `backend/CONFIG_QUICK_REFERENCE.md` for more profiles.

---

## Monitoring

### Check Database Stats

```powershell
cd backend
python -c "import sqlite3; conn = sqlite3.connect('data/papers.db'); c = conn.cursor(); c.execute('SELECT COUNT(*) FROM papers'); print('Papers:', c.fetchone()[0]); c.execute('SELECT COUNT(*) FROM authors'); print('Authors:', c.fetchone()[0])"
```

### View Logs

```powershell
Get-Content backend\ingest.log -Tail 50
Get-Content backend\build_vectors.log -Tail 50
```

---

## Troubleshooting

### Issue: Ingestion Slow

**Solution**: Some PDFs are complex. This is normal. Average: ~2-3 seconds per PDF.

---

### Issue: Out of Memory During Embedding

**Solution**: Reduce batch size:
```powershell
python build_vectors.py --batch-size 2
```

---

### Issue: Models Not Found

**Solution**: Check models directory:
```powershell
Get-ChildItem models, data -Recurse | Where-Object {!$_.PSIsContainer}
```

If missing, rebuild specific model:
```powershell
python build_tfidf.py        # TF-IDF
python build_vectors.py      # FAISS
python train_ranker.py       # LambdaRank
```

---

### Issue: API Returns 503

**Solution**: Check which models are loaded:
```powershell
curl http://localhost:8000/health
```

If a model is missing, rebuild it.

---

## Performance Expectations

### Dataset Size: 538 PDFs

- **Ingestion**: 20-40 minutes
- **TF-IDF**: 1-2 minutes
- **FAISS**: 10-30 minutes
- **Training Data**: 5-10 minutes
- **LambdaRank**: 2-5 minutes
- **Total**: ~40-90 minutes

### Query Performance

- **Response Time**: 200-500ms per query
- **Throughput**: ~2-5 queries/second
- **Memory Usage**: ~2-3 GB RAM

### Model Sizes

- **TF-IDF**: ~10 MB
- **FAISS**: ~200 MB (538 papers × 768 dims × 4 bytes)
- **Training Data**: ~50 MB
- **LambdaRank**: ~1 MB
- **Total**: ~260 MB

---

## Next Steps

After the system is running:

1. **Evaluate Models**: Run evaluation report
   ```powershell
   cd backend
   python eval_report.py --queries 100
   ```

2. **Deploy**: Follow Docker deployment guides
   - Backend: `backend/DOCKER_GUIDE.md`
   - Frontend: `frontend/DEPLOYMENT.md`

3. **Tune Configuration**: Adjust weights and parameters
   - See: `backend/CONFIG_QUICK_REFERENCE.md`

4. **Monitor Performance**: Track metrics over time
   - P@5, nDCG@10
   - Response times
   - User feedback

---

## Summary

✅ **Complete Dataset Processing**:
1. Ingest 538 PDFs → SQLite database
2. Build TF-IDF vectorizer for lexical search
3. Build FAISS index for semantic search
4. Generate training data with weak labels
5. Train LambdaRank model
6. Start API server
7. Start frontend

🎯 **Total Time**: ~40-90 minutes for full pipeline

🚀 **Result**: Production-ready reviewer recommendation system!

---

**Need help?** Check:
- `PROMPT22_COMPLETION.md` - Configuration tuning
- `PROMPT21_COMPLETION.md` - Evaluation framework
- `backend/DOCKER_GUIDE.md` - Deployment
- `frontend/DEPLOYMENT.md` - Frontend deployment

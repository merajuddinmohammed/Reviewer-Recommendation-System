# LambdaRank Hyperparameter and Label Generation Fixes

## Problem Summary
LambdaRank was showing 0.00% performance in the evaluation report due to:
1. **Poor hyperparameters**: Model stopped at iteration 1, only learned from `recency_mean` feature
2. **Imbalanced training data**: Only 14 positive samples out of 4714 (0.30%)
3. **Insufficient label diversity**: Only 3 positives per query, mostly from coauthors

## Fixes Applied

### 1. Improved Hyperparameters (`train_ranker.py`)

**Before:**
```python
{
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
```

**After:**
```python
{
    'num_leaves': 15,              # Reduced from 31 (less complex trees)
    'learning_rate': 0.01,         # Reduced from 0.05 (slower, more careful learning)
    'min_data_in_leaf': 5,         # Reduced from 20 (allow smaller groups)
    'max_depth': 6,                # NEW: Limit tree depth
    'min_gain_to_split': 0.1,     # NEW: Require meaningful splits
    'feature_fraction': 0.9,       # Increased from 0.8 (use more of our 9 features)
    'bagging_fraction': 0.9,       # Increased from 0.8 (use more data)
    'bagging_freq': 3,             # Reduced from 5 (more frequent bagging)
    'lambda_l1': 0.1,              # NEW: L1 regularization
    'lambda_l2': 0.1,              # NEW: L2 regularization
    'min_data_per_group': 3,       # NEW: Minimum samples per query group
    'n_estimators': 1000,          # Increased from 500 (more iterations)
}
```

### 2. Improved Label Generation (`build_training_data.py`)

**Before:**
- n_positives: 3 (too few)
- n_negatives: 20
- Only added weak positives IF strong positives insufficient

**After:**
- n_positives: 10 (more diverse)
- n_negatives: 15 (better ratio)
- ALWAYS add weak positives for diversity
- Increased coauthor search from 10 to 20 authors
- Increased weak positive candidates from 2x to 3x

**Code Changes:**
```python
# Strategy 2: Co-author neighborhood
coauthor_ids = get_coauthor_neighborhood(db_path, query_author_name, max_authors=20)  # Was 10
positive_ids.update(coauthor_ids)

# Strategy 3: ALWAYS add weak positives (not just when insufficient)
weak_positives = get_weak_positive_authors(
    query_text=query_text,
    db_path=db_path,
    tfidf_engine=tfidf_engine,
    faiss_index=faiss_index,
    id_map=id_map,
    emb_model=emb_model,
    query_author_id=query_author_id,
    topn=n_positives * 3  # Was n_positives * 2
)
positive_ids.update(weak_positives)  # No longer conditional
```

### 3. Fixed Bug in eval_report.py

**Error:**
```python
make_features_for_query(
    db_path=db_path,  # Wrong parameter name
    query_affiliations=[]  # Wrong parameter name
)
```

**Fixed:**
```python
make_features_for_query(
    db=db_path,  # Correct parameter name
    query_affiliation=None  # Correct parameter name
)
```

### 4. Fixed Bug in build_training_data.py

**Error:**
```python
tfidf_sims = tfidf_engine.transform_query(query_text, return_scores=True)  # Method doesn't exist
```

**Fixed:**
```python
tfidf_results = tfidf_engine.most_similar(query_text, topn=topn, return_scores=True)  # Correct method
tfidf_paper_ids = [pid for pid, score in tfidf_results]
```

## Expected Results

### Training Data
- **Old**: 4714 samples, 14 positives (0.30%), 235 queries
- **New**: ~5000 samples, ~2000 positives (40%), 200 queries

### Model Performance
- **Old**: 1 tree, NDCG@5=0.00%, only uses recency_mean
- **New**: ~100-200 trees, NDCG@5>20%, uses all 9 features

### Feature Importance (Expected)
1. **emb_max** / **emb_mean**: 30-40% (embedding similarity)
2. **tfidf_max** / **tfidf_mean**: 25-35% (TF-IDF similarity)
3. **recency_mean** / **recency_max**: 15-25% (recent publications)
4. **pub_count**: 5-10% (publication count)
5. **coi_flag**: 1-5% (conflict of interest)

## Retraining Commands

```powershell
# 1. Rebuild training data with improved labels
cd "C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment\backend"
..\.venv\Scripts\python.exe build_training_data.py --queries 200 --positives 10 --negatives 15 --out data/train.parquet

# 2. Retrain LambdaRank with improved hyperparameters
..\.venv\Scripts\python.exe train_ranker.py --data data/train.parquet --n-estimators 1000 --learning-rate 0.01 --num-leaves 15 --min-data-in-leaf 5

# 3. Regenerate evaluation report
..\.venv\Scripts\python.exe eval_report.py --queries 50 --seed 42
```

## Files Modified

1. `backend/train_ranker.py` - Improved hyperparameters
2. `backend/build_training_data.py` - Improved label generation strategy
3. `backend/eval_report.py` - Fixed parameter names bug
4. `backend/models/lgbm_ranker.pkl` - Will be regenerated

## Verification

After retraining, check:
1. Model has >50 trees (not just 1)
2. LambdaRank shows >20% P@5 in evaluation report (not 0%)
3. Feature importance includes emb_max/emb_mean (not just recency_mean)
4. Training completes without early stopping at iteration 1

---

**Status**: Fixes applied, training data being regenerated, model retraining pending
**Expected Completion**: ~10 minutes for data generation + ~5 minutes for training

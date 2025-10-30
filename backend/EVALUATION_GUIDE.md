# Evaluation Report - Quick Reference

Quick guide for running evaluation and interpreting results.

---

## Quick Start

```bash
cd backend
python eval_report.py
```

This will:
1. Sample 100 query papers
2. Evaluate 4 ranking methods
3. Save `data/eval_metrics_per_query.csv`
4. Save `data/eval_report.md`

---

## Command Line Options

```bash
python eval_report.py \
  --db data/papers.db \           # Database path
  --queries 100 \                  # Number of queries
  --seed 42 \                      # Random seed
  --out-csv data/metrics.csv \     # CSV output
  --out-md data/report.md          # Markdown output
```

---

## Ranking Methods Evaluated

| Method | Description | Features Used |
|--------|-------------|---------------|
| **A: TF-IDF** | Lexical matching | TF-IDF similarity only |
| **B: Embeddings** | Semantic matching | SciBERT embeddings only |
| **C: Hybrid** | Weighted combination | 0.55*emb + 0.25*tfidf + 0.20*recency |
| **D: LambdaRank** | Learned ranker | All 11 features + LightGBM |

---

## Evaluation Metrics

### Precision@5 (P@5)

**What it measures**: Fraction of relevant items in top 5

**Formula**: `(# relevant in top 5) / 5`

**Interpretation**:
- 0.0 = No relevant items in top 5 (bad)
- 0.4 = 2 relevant items in top 5 (fair)
- 0.6 = 3 relevant items in top 5 (good)
- 0.8 = 4 relevant items in top 5 (excellent)
- 1.0 = All 5 items are relevant (perfect)

**Good threshold**: P@5 > 0.4

### nDCG@10

**What it measures**: Quality of ranking in top 10 (position matters)

**Formula**: `DCG@10 / IDCG@10`

**Interpretation**:
- 0.0-0.3 = Poor ranking
- 0.3-0.5 = Fair ranking
- 0.5-0.7 = Good ranking
- 0.7-0.9 = Very good ranking
- 0.9-1.0 = Excellent ranking

**Good threshold**: nDCG@10 > 0.6

**Why use nDCG?**:
- Penalizes relevant items ranked lower
- Rewards relevant items ranked higher
- More informative than P@K

---

## Weak Positive Labels

**How labels are generated** (same as training data):

1. **Coauthors**: Authors who co-authored with query paper's author
2. **TF-IDF neighbors**: Authors from top 10 TF-IDF similar papers
3. **Embedding neighbors**: Authors from top 10 embedding similar papers

**Exclusions**:
- Query paper's own author (conflict of interest)

**Limitations**:
- Not ground truth (weak supervision)
- May miss some relevant reviewers
- May include some irrelevant reviewers
- Best for relative comparison between methods

---

## Output Files

### CSV: `data/eval_metrics_per_query.csv`

**Columns**:
```
query_id          - Paper ID
query_title       - Paper title
n_positives       - Number of positive labels
tfidf_p5          - P@5 for TF-IDF
tfidf_ndcg10      - nDCG@10 for TF-IDF
emb_p5            - P@5 for embeddings
emb_ndcg10        - nDCG@10 for embeddings
hybrid_p5         - P@5 for hybrid
hybrid_ndcg10     - nDCG@10 for hybrid
lambdarank_p5     - P@5 for LambdaRank (if available)
lambdarank_ndcg10 - nDCG@10 for LambdaRank (if available)
```

**Use cases**:
- Detailed analysis per query
- Find queries where methods differ
- Debug poor performance
- Statistical analysis

### Markdown: `data/eval_report.md`

**Sections**:
1. **Summary Statistics**: Mean and std for all metrics
2. **Best Method**: Winner by P@5 and nDCG@10
3. **Analysis**: Lexical vs semantic vs hybrid vs ranker
4. **Conclusions**: Recommendations for deployment

**Use cases**:
- Executive summary
- Method comparison
- Deployment decisions
- Documentation

---

## Expected Results

**Typical performance ranges**:

| Method | P@5 | nDCG@10 |
|--------|-----|---------|
| TF-IDF | 0.25-0.35 | 0.40-0.50 |
| Embeddings | 0.35-0.45 | 0.50-0.60 |
| Hybrid | 0.40-0.50 | 0.55-0.65 |
| LambdaRank | 0.45-0.55 | 0.60-0.70 |

**Expected ordering** (best to worst):
1. LambdaRank (learning helps)
2. Hybrid (combination helps)
3. Embeddings (semantic > lexical)
4. TF-IDF (lexical baseline)

**If your results differ**:
- Check training data quality
- Check positive label coverage
- Try different seeds
- Increase number of queries

---

## Deterministic Evaluation

**Same seed = same results**:
```bash
python eval_report.py --seed 42  # Run 1
python eval_report.py --seed 42  # Run 2 (identical)
```

**Different seed = different queries**:
```bash
python eval_report.py --seed 42  # Query set A
python eval_report.py --seed 99  # Query set B (different)
```

**For reproducibility**:
- Always specify --seed
- Document seed in reports
- Use same seed for comparisons

---

## Troubleshooting

### Models Not Found

```bash
# Build all models first
python build_tfidf.py      # TF-IDF
python build_vectors.py    # FAISS embeddings
python train_ranker.py     # LambdaRank
```

### No Query Papers

```bash
# Check database
python -c "import sqlite3; print(sqlite3.connect('data/papers.db').execute('SELECT COUNT(*) FROM papers').fetchone()[0])"

# Ingest more papers if needed
python ingest.py
```

### Runtime Too Long

```bash
# Reduce queries
python eval_report.py --queries 50

# Or reduce topn in code (edit eval_report.py)
# Change: topn=100 to topn=50
```

### Memory Issues

```bash
# Reduce batch size
python eval_report.py --queries 20

# Or run in chunks
python eval_report.py --queries 50 --out-csv chunk1.csv
python eval_report.py --queries 50 --out-csv chunk2.csv --seed 43
```

---

## Interpretation Examples

### Example 1: Embeddings Win

```
TF-IDF:      P@5 = 0.32, nDCG@10 = 0.45
Embeddings:  P@5 = 0.48, nDCG@10 = 0.62
Hybrid:      P@5 = 0.50, nDCG@10 = 0.65
LambdaRank:  P@5 = 0.54, nDCG@10 = 0.68
```

**Conclusion**: Semantic matching is more important than lexical matching for this dataset.

**Recommendation**: Deploy hybrid or LambdaRank.

### Example 2: TF-IDF Wins

```
TF-IDF:      P@5 = 0.45, nDCG@10 = 0.58
Embeddings:  P@5 = 0.38, nDCG@10 = 0.52
Hybrid:      P@5 = 0.46, nDCG@10 = 0.60
LambdaRank:  P@5 = 0.48, nDCG@10 = 0.62
```

**Conclusion**: Lexical matching is surprisingly effective, possibly due to technical terminology.

**Recommendation**: Deploy hybrid or LambdaRank (slight improvement).

### Example 3: LambdaRank No Improvement

```
TF-IDF:      P@5 = 0.35, nDCG@10 = 0.48
Embeddings:  P@5 = 0.42, nDCG@10 = 0.56
Hybrid:      P@5 = 0.45, nDCG@10 = 0.60
LambdaRank:  P@5 = 0.44, nDCG@10 = 0.59
```

**Conclusion**: LambdaRank not helping, possibly due to:
- Insufficient training data
- Weak positive labels
- Overfitting

**Recommendation**: Deploy hybrid. Collect more training data for LambdaRank.

---

## Next Steps

### Improve Evaluation

1. **Increase queries**: 100 → 500 for more robust estimates
2. **Use strong labels**: Human annotations or actual assignments
3. **Cross-validation**: Train/test split for unbiased estimates
4. **Add metrics**: Recall@K, MAP, MRR

### Improve Models

1. **Tune weights**: Grid search for hybrid weights
2. **More training data**: Generate more synthetic labels
3. **Better features**: Add citation features, topic modeling
4. **Ensemble**: Combine multiple models

### Production Deployment

1. **A/B test**: Deploy multiple methods and compare
2. **User feedback**: Collect explicit feedback
3. **Monitor metrics**: Track P@5 and nDCG@10 over time
4. **Iterate**: Retrain models with production data

---

## Quick Commands

```bash
# Run evaluation
python eval_report.py

# Run with 50 queries
python eval_report.py --queries 50

# Run with custom seed
python eval_report.py --seed 123

# Run with custom output
python eval_report.py \
  --out-csv results/metrics.csv \
  --out-md results/report.md

# View report
cat data/eval_report.md

# View summary statistics
head -20 data/eval_report.md

# Check CSV
head data/eval_metrics_per_query.csv
wc -l data/eval_metrics_per_query.csv
```

---

## Resources

- **Precision@K**: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision
- **nDCG**: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
- **Learning to Rank**: https://en.wikipedia.org/wiki/Learning_to_rank
- **LambdaRank**: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/

---

**Need help?** Check PROMPT21_COMPLETION.md for detailed documentation.

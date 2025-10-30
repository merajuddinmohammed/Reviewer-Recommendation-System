# Evaluation Report - Ranking Method Comparison

**Generated**: 2025-10-31 02:40:49

**Number of Queries**: 30

**Evaluation Metrics**: Precision@5 (P@5), nDCG@10

---

## Summary Statistics

| Method | P@5 Mean | P@5 Std | nDCG@10 Mean | nDCG@10 Std |
|--------|----------|---------|--------------|-------------|
| TF-IDF Only | 0.9267 | 0.1530 | 0.7703 | 0.1398 |
| Embeddings Only | 0.9733 | 0.0691 | 0.7946 | 0.1010 |
| Hybrid Weighted | 0.7133 | 0.1871 | 0.6250 | 0.1332 |
| LambdaRank | 0.5333 | 0.2057 | 0.4959 | 0.1683 |

---

## Best Performing Method

**By P@5**: Embeddings Only (0.9733)

**By nDCG@10**: Embeddings Only (0.7946)

---

## Analysis

### Lexical vs Semantic

**Semantic matching (embeddings)** outperforms lexical matching (TF-IDF) by 0.0467 on P@5. This suggests that semantic understanding is more important than keyword matching for reviewer recommendation.

### Hybrid Approach

**Hybrid method** does not improve over individual methods. This suggests that simple max aggregation may be sufficient, or that weight tuning is needed.

### Learning-to-Rank

**LambdaRank** does not improve over hybrid method. This may be due to limited training data or weak positive labels. More training data or better labels could improve performance.

---

## Conclusions

1. **Semantic vs Lexical**: Semantic (embeddings) is more effective for reviewer recommendation.

2. **Hybrid Approach**: Combining multiple signals can improve performance, but weight tuning is important.

3. **Learning-to-Rank**: Learning optimal feature weights can further improve ranking quality with sufficient training data.

4. **Recommendations**:
   - Deploy **Embeddings Only** for best P@5 performance
   - Consider A/B testing different methods in production
   - Collect user feedback to improve labels and training data
   - Monitor metrics over time to detect degradation


"""
Test script for FastAPI app
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Testing FastAPI App Components")
print("=" * 80)

# Test 1: Import app
print("\n1. Testing app import...")
try:
    from app import app, models, extract_text_from_pdf, compute_weighted_score
    print("   ✓ App imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import app: {e}")
    sys.exit(1)

# Test 2: Check model loading
print("\n2. Checking model status...")
print(f"   Database: {models.db_path}")
print(f"   TF-IDF vectorizer: {'✓ Loaded' if models.tfidf_vectorizer else '✗ Not loaded'}")
print(f"   FAISS index: {'✓ Loaded' if models.faiss_index else '✗ Not loaded'}")
print(f"   ID map: {'✓ Loaded' if models.id_map is not None else '✗ Not loaded'}")
print(f"   BERTopic: {'✓ Loaded' if models.bertopic_model else '⚠ Optional (not loaded)'}")
print(f"   LightGBM: {'✓ Loaded' if models.lgbm_model else '⚠ Optional (not loaded)'}")
print(f"   API Ready: {'✓ Yes' if models.is_ready() else '✗ No'}")

# Test 3: Test weighted scoring function
print("\n3. Testing weighted score function...")
import pandas as pd
test_features = pd.Series({
    'emb_max': 0.8,
    'tfidf_max': 0.6,
    'recency_max': 0.9
})
score = compute_weighted_score(test_features)
expected = 0.55 * 0.8 + 0.25 * 0.6 + 0.20 * 0.9
print(f"   Input: emb_max=0.8, tfidf_max=0.6, recency_max=0.9")
print(f"   Score: {score:.4f}")
print(f"   Expected: {expected:.4f}")
print(f"   ✓ Weighted scoring works" if abs(score - expected) < 0.001 else f"   ✗ Score mismatch")

# Test 4: Test make_features_for_query (if models ready)
if models.is_ready():
    print("\n4. Testing feature generation...")
    try:
        from ranker import make_features_for_query
        
        features_df = make_features_for_query(
            db_path=models.db_path,
            query_title="Deep Learning for Computer Vision",
            query_abstract="This paper presents a novel approach to image classification.",
            query_authors=["Test Author"],
            query_affiliations=["Test University"],
            tfidf_model_path=Path("models/tfidf_vectorizer.pkl"),
            faiss_index_path=Path("data/faiss_index.faiss"),
            id_map_path=Path("data/id_map.npy"),
            topic_model_path=None
        )
        
        print(f"   Generated features for {len(features_df)} candidates")
        print(f"   Feature columns: {list(features_df.columns)}")
        print(f"   ✓ Feature generation works")
        
    except Exception as e:
        print(f"   ✗ Feature generation failed: {e}")
else:
    print("\n4. Skipping feature generation test (models not ready)")

# Test 5: Test API endpoints using TestClient
print("\n5. Testing API endpoints...")
try:
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    print(f"   GET /: {response.status_code} - {response.json()['message']}")
    
    # Test health endpoint
    response = client.get("/health")
    print(f"   GET /health: {response.status_code} - Status: {response.json()['status']}")
    
    # Test recommend endpoint (if models ready)
    if models.is_ready():
        response = client.post("/recommend", json={
            "title": "Test Paper on Machine Learning",
            "abstract": "This paper explores machine learning techniques for classification.",
            "k": 5
        })
        print(f"   POST /recommend: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"      Recommendations: {len(data['recommendations'])}")
            print(f"      Total candidates: {data['total_candidates']}")
            print(f"      Model used: {data['model_used']}")
            print(f"   ✓ Recommendation endpoint works")
        else:
            print(f"   ✗ Recommendation failed: {response.json()}")
    else:
        print("   ⚠ Skipping recommendation test (models not ready)")
    
except Exception as e:
    print(f"   ✗ API endpoint tests failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)

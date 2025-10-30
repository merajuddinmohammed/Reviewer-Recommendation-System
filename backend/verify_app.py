"""
Simple verification that app.py is correctly structured
"""

import sys
from pathlib import Path

print("=" * 80)
print("FastAPI App Verification")
print("=" * 80)

# Test import
print("\n1. Testing imports...")
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    print("   ✓ FastAPI imports work")
except ImportError as e:
    print(f"   ✗ FastAPI import failed: {e}")
    sys.exit(1)

# Test app file syntax
print("\n2. Checking app.py syntax...")
try:
    with open("app.py", 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, "app.py", "exec")
    print("   ✓ app.py syntax is valid")
except SyntaxError as e:
    print(f"   ✗ Syntax error in app.py: {e}")
    sys.exit(1)

# Test that we can import the app module
print("\n3. Importing app module...")
try:
    import app
    print("   ✓ App module imported")
    print(f"   - FastAPI app: {type(app.app)}")
    print(f"   - Model store: {type(app.models)}")
except Exception as e:
    print(f"   ✗ Failed to import app module: {e}")
    sys.exit(1)

# Check endpoints are defined
print("\n4. Checking endpoints...")
try:
    routes = [route.path for route in app.app.routes]
    print(f"   Found {len(routes)} routes:")
    for route in routes:
        print(f"      - {route}")
    
    required_routes = ["/health", "/recommend"]
    for route in required_routes:
        if route in routes:
            print(f"   ✓ {route} endpoint defined")
        else:
            print(f"   ✗ {route} endpoint missing")
except Exception as e:
    print(f"   ✗ Failed to check endpoints: {e}")

# Check model paths
print("\n5. Checking model file paths...")
model_files = {
    "Database": Path("data/papers.db"),
    "TF-IDF": Path("models/tfidf_vectorizer.pkl"),
    "FAISS Index": Path("data/faiss_index.faiss"),
    "ID Map": Path("data/id_map.npy"),
    "LightGBM": Path("models/lgbm_ranker.pkl")
}

for name, path in model_files.items():
    if path.exists():
        size = path.stat().st_size
        print(f"   ✓ {name}: {path} ({size:,} bytes)")
    else:
        print(f"   ✗ {name}: {path} (NOT FOUND)")

print("\n" + "=" * 80)
print("Verification Complete!")
print("=" * 80)
print("\nTo start the server:")
print("  cd backend")
print("  python app.py")
print("\nOr with uvicorn:")
print("  uvicorn app:app --reload --host 0.0.0.0 --port 8000")

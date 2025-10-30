"""
Complete Dataset Processing Pipeline

This script automates the full pipeline:
1. Monitors ingestion completion
2. Builds TF-IDF vectorizer
3. Builds FAISS embeddings index  
4. Generates training data
5. Trains LambdaRank model

Run after starting ingest.py in a separate terminal.
"""

import time
import sqlite3
import subprocess
import sys
from pathlib import Path

def get_db_stats():
    """Get current database statistics."""
    db_path = Path("data/papers.db")
    if not db_path.exists():
        return None, None
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM papers")
        papers = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM authors")
        authors = cursor.fetchone()[0]
        
        conn.close()
        
        return papers, authors
    except Exception as e:
        print(f"Error checking database: {e}")
        return None, None

def wait_for_ingestion(target_papers=500):
    """Wait for ingestion to reach target number of papers."""
    print("=" * 80)
    print("Waiting for ingestion to complete...")
    print("=" * 80)
    print()
    
    prev_papers = 0
    stable_count = 0
    
    while True:
        papers, authors = get_db_stats()
        
        if papers is None:
            print("Waiting for database...")
            time.sleep(5)
            continue
        
        if papers != prev_papers:
            print(f"[{time.strftime('%H:%M:%S')}] Papers: {papers:4d}, Authors: {authors:4d}")
            prev_papers = papers
            stable_count = 0
        else:
            stable_count += 1
        
        # If no change for 30 seconds and we have reasonable number of papers, assume complete
        if stable_count >= 15 and papers >= 50:
            print()
            print(f"✓ Ingestion appears complete: {papers} papers, {authors} authors")
            print()
            return papers, authors
        
        time.sleep(2)

def run_command(cmd, description):
    """Run a command and show output."""
    print("=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print()
        print(f"✓ {description} completed successfully")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ {description} failed with error code {e.returncode}")
        print()
        return False

def build_all_models():
    """Build all required models."""
    print("=" * 80)
    print("Building All Models")
    print("=" * 80)
    print()
    
    # Step 1: Build TF-IDF vectorizer
    success = run_command(
        r"..\.venv\Scripts\python.exe build_tfidf.py",
        "Build TF-IDF Vectorizer"
    )
    if not success:
        print("Warning: TF-IDF build failed, continuing...")
    
    # Step 2: Build FAISS embeddings index (may take a while)
    print("Note: Building embeddings may take 10-30 minutes depending on dataset size...")
    success = run_command(
        r"..\.venv\Scripts\python.exe build_vectors.py --batch-size 4",
        "Build FAISS Embeddings Index"
    )
    if not success:
        print("Warning: FAISS build failed, continuing...")
    
    # Step 3: Generate training data
    success = run_command(
        r"..\.venv\Scripts\python.exe build_training_data.py --n_train 5000 --n_val 1000",
        "Generate Training Data"
    )
    if not success:
        print("Warning: Training data generation failed, continuing...")
    
    # Step 4: Train LambdaRank model
    success = run_command(
        r"..\.venv\Scripts\python.exe train_ranker.py --n_estimators 100 --max_depth 6",
        "Train LambdaRank Model"
    )
    if not success:
        print("Warning: LambdaRank training failed, continuing...")
    
    print("=" * 80)
    print("Model Building Complete!")
    print("=" * 80)
    print()

def check_models():
    """Check which models exist."""
    print("=" * 80)
    print("Checking Model Files")
    print("=" * 80)
    print()
    
    models = {
        "TF-IDF Vectorizer": Path("models/tfidf_vectorizer.pkl"),
        "FAISS Index": Path("data/faiss_index.faiss"),
        "FAISS ID Map": Path("data/id_map.npy"),
        "Training Data": Path("data/train.parquet"),
        "LambdaRank Model": Path("models/lgbm_ranker.pkl")
    }
    
    for name, path in models.items():
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {name:25s} {size:8.2f} MB")
        else:
            print(f"✗ {name:25s} NOT FOUND")
    
    print()

def main():
    """Main pipeline."""
    print("=" * 80)
    print("Complete Dataset Processing Pipeline")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Wait for ingestion to complete")
    print("  2. Build TF-IDF vectorizer")
    print("  3. Build FAISS embeddings index")
    print("  4. Generate training data")
    print("  5. Train LambdaRank model")
    print()
    print("Make sure ingest.py is running in another terminal!")
    print()
    
    input("Press Enter to start monitoring...")
    print()
    
    # Wait for ingestion
    papers, authors = wait_for_ingestion()
    
    # Build all models
    build_all_models()
    
    # Check final status
    check_models()
    
    print("=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print()
    print(f"Final Statistics:")
    print(f"  Papers:  {papers}")
    print(f"  Authors: {authors}")
    print()
    print("You can now start the backend server:")
    print("  cd backend")
    print("  python app.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Pipeline interrupted by user")
        sys.exit(1)

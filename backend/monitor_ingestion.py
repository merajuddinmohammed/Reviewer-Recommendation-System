"""
Monitor Ingestion and Build Models

This script monitors the ingestion process and then builds all required models.

Steps:
1. Monitor database growth during ingestion
2. Build TF-IDF vectorizer
3. Build FAISS embeddings index
4. Build training data for LambdaRank
5. Train LambdaRank model
"""

import time
import sqlite3
from pathlib import Path

def get_db_stats():
    """Get current database statistics."""
    db_path = Path("data/papers.db")
    if not db_path.exists():
        return None, None
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM papers")
    papers = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM authors")
    authors = cursor.fetchone()[0]
    
    conn.close()
    
    return papers, authors

if __name__ == "__main__":
    print("=" * 80)
    print("Ingestion Monitor")
    print("=" * 80)
    print()
    
    prev_papers = 0
    prev_authors = 0
    
    # Monitor for changes
    while True:
        papers, authors = get_db_stats()
        
        if papers is None:
            print("Database not found yet...")
            time.sleep(5)
            continue
        
        if papers != prev_papers or authors != prev_authors:
            print(f"[{time.strftime('%H:%M:%S')}] Papers: {papers}, Authors: {authors}")
            prev_papers = papers
            prev_authors = authors
        
        time.sleep(2)

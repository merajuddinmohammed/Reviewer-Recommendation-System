"""
Complete Integration Demo - PDF Ingestion to TF-IDF Search

This script demonstrates the full pipeline:
1. Initialize database
2. Ingest PDFs
3. Build TF-IDF index
4. Search for similar papers
"""

from pathlib import Path
from db_utils import init_db, get_all_papers
from parser import walk_and_ingest
from tfidf_engine import TFIDFEngine
from utils import clean_text

def full_pipeline_demo():
    """
    Demonstrates complete pipeline from PDF ingestion to similarity search.
    
    Expected directory structure:
        papers/
            Author One/
                paper1.pdf
                paper2.pdf
            Author Two/
                paper3.pdf
    """
    print("=" * 70)
    print("FULL PIPELINE DEMO: PDF → Database → TF-IDF → Search")
    print("=" * 70)
    
    # Configuration
    papers_dir = Path("papers")
    db_path = Path("papers.db")
    model_path = Path("models/tfidf_latest.joblib")
    
    # Step 1: Initialize Database
    print("\n[STEP 1] Initializing database...")
    init_db(db_path)
    print("  Database initialized: papers.db")
    
    # Step 2: Ingest PDFs (if papers directory exists)
    if papers_dir.exists():
        print(f"\n[STEP 2] Ingesting PDFs from {papers_dir}...")
        results = walk_and_ingest(papers_dir, db_path)
        
        print(f"  Total PDFs found: {results['total_pdfs']}")
        print(f"  Successfully processed: {results['successful_pdfs']}")
        print(f"  Failed: {results['failed_pdfs']}")
        print(f"  Authors created: {results['authors_created']}")
        print(f"  Papers created: {results['papers_created']}")
        print(f"  Papers with abstracts: {results['papers_with_abstract']} "
              f"({results['abstract_percentage']:.1f}%)")
        print(f"  Papers with years: {results['papers_with_year']} "
              f"({results['year_percentage']:.1f}%)")
        
        if results['errors']:
            print(f"  Errors encountered: {results['error_count']}")
            for error in results['errors'][:3]:
                print(f"    - {error}")
    else:
        print(f"\n[STEP 2] No papers directory found at {papers_dir}")
        print("  Skipping ingestion. Using existing database.")
    
    # Step 3: Load papers from database
    print("\n[STEP 3] Loading papers from database...")
    papers = get_all_papers(db_path)
    
    if not papers:
        print("  No papers found in database!")
        print("  Please add PDFs to papers/ directory and re-run.")
        return
    
    print(f"  Loaded {len(papers)} papers from database")
    
    # Step 4: Build TF-IDF index
    print("\n[STEP 4] Building TF-IDF index...")
    
    # Prepare corpus
    corpus_texts = []
    paper_ids = []
    
    for paper in papers:
        # Combine title, abstract, fulltext for maximum context
        text_parts = []
        
        if paper['title']:
            text_parts.append(clean_text(paper['title']))
        
        if paper['abstract']:
            text_parts.append(clean_text(paper['abstract']))
        
        if paper['fulltext']:
            # Use first 5000 words to avoid memory issues
            fulltext_clean = clean_text(paper['fulltext'])
            words = fulltext_clean.split()[:5000]
            text_parts.append(' '.join(words))
        
        combined_text = ' '.join(text_parts)
        
        if combined_text.strip():  # Only add non-empty texts
            corpus_texts.append(combined_text)
            paper_ids.append(paper['id'])
    
    print(f"  Prepared {len(corpus_texts)} documents for indexing")
    
    # Configure and fit TF-IDF
    if len(corpus_texts) < 2:
        print("  Need at least 2 documents for TF-IDF. Please add more papers.")
        return
    
    # Adjust min_df based on corpus size
    min_df = max(1, min(2, len(corpus_texts) // 10))
    
    engine = TFIDFEngine(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.85
    )
    
    engine.fit(corpus_texts, paper_ids)
    
    print(f"  TF-IDF fitted:")
    print(f"    Vocabulary size: {len(engine.vectorizer.vocabulary_)}")
    print(f"    Matrix shape: {engine.corpus_matrix.shape}")
    sparsity = 1 - engine.corpus_matrix.nnz / (engine.corpus_matrix.shape[0] * engine.corpus_matrix.shape[1])
    print(f"    Matrix sparsity: {sparsity:.4f}")
    
    # Step 5: Save model
    print("\n[STEP 5] Saving TF-IDF model...")
    model_path.parent.mkdir(exist_ok=True)
    engine.save(str(model_path))
    print(f"  Model saved to: {model_path}")
    
    # Step 6: Demo search queries
    print("\n[STEP 6] Running demo search queries...")
    
    demo_queries = [
        "deep learning neural networks",
        "machine learning algorithms",
        "natural language processing",
        "computer vision image recognition"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n  Query {i}: '{query}'")
        results = engine.most_similar(query, topn=5)
        
        if results:
            print(f"  Found {len(results)} similar papers:")
            for rank, (pid, score) in enumerate(results, 1):
                # Get paper details
                paper = next((p for p in papers if p['id'] == pid), None)
                if paper:
                    title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
                    year = paper['year'] or "N/A"
                    print(f"    {rank}. [{year}] {title}")
                    print(f"       Score: {score:.4f}")
        else:
            print("  No similar papers found.")
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Papers in database: {len(papers)}")
    print(f"  Papers indexed: {len(corpus_texts)}")
    print(f"  Model location: {model_path}")
    print(f"\nNext steps:")
    print("  1. Load model: engine = TFIDFEngine.load('models/tfidf_latest.joblib')")
    print("  2. Search: results = engine.most_similar('your query', topn=10)")
    print("  3. Get paper details from database using paper IDs")


if __name__ == "__main__":
    try:
        full_pipeline_demo()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

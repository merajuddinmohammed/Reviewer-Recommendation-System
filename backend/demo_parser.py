"""
Demo script for PDF ingestion system.

Creates a sample directory structure with mock PDFs and demonstrates
the ingestion pipeline.
"""

import tempfile
import shutil
from pathlib import Path
from parser import walk_and_ingest


def create_sample_structure():
    """
    Create a sample directory structure with test data.
    
    Returns:
        Tuple of (root_dir, db_path)
    """
    # Create temporary directories
    temp_root = Path(tempfile.mkdtemp(prefix="pdf_ingest_demo_"))
    
    print(f"Created sample directory at: {temp_root}")
    print("\nDirectory structure:")
    
    # Create author directories
    authors = [
        "Alice Smith",
        "Bob Johnson",
        "Carol Williams"
    ]
    
    for author in authors:
        author_dir = temp_root / author
        author_dir.mkdir()
        print(f"  {author}/")
        
        # Note: In real usage, you would copy actual PDFs here
        # For now, we just create the structure
    
    # Database path
    db_path = temp_root / "papers.db"
    
    print(f"\nDatabase will be created at: {db_path}")
    print("\nTo use this demo:")
    print("1. Copy your PDF files into the author folders above")
    print("2. Run the ingestion:")
    print(f"   from parser import walk_and_ingest")
    print(f"   walk_and_ingest('{temp_root}', '{db_path}')")
    
    return temp_root, db_path


def demo_ingestion():
    """
    Demonstrate the ingestion pipeline.
    """
    print("=" * 70)
    print("PDF Ingestion Demo")
    print("=" * 70)
    print("\nThis demo shows how to use the PDF ingestion system.")
    print("\nRequired directory structure:")
    print("  root_dir/")
    print("    Author Name 1/")
    print("      paper1.pdf")
    print("      paper2.pdf")
    print("    Author Name 2/")
    print("      paper3.pdf")
    print("\n" + "=" * 70)
    
    root_dir, db_path = create_sample_structure()
    
    print(f"\n{'=' * 70}")
    print("Sample structure created successfully!")
    print(f"{'=' * 70}")
    print(f"\nRoot directory: {root_dir}")
    print(f"Database path: {db_path}")
    print("\nWhen you're ready to ingest PDFs, use:")
    print(f"\n  python")
    print(f"  >>> from parser import walk_and_ingest")
    print(f"  >>> results = walk_and_ingest(r'{root_dir}', r'{db_path}')")
    print(f"  >>> print(results)")
    
    return root_dir, db_path


if __name__ == "__main__":
    demo_ingestion()

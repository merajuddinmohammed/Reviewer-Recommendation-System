#!/usr/bin/env python3
"""
Offline PDF Ingestion Script

This script walks through a directory structure of academic papers, extracts metadata
and text from PDFs, and stores everything in a SQLite database. After ingestion, it
generates a summary CSV report with statistics about authors and papers.

Directory Structure Expected:
    data/papers/
    ├── Author1/
    │   ├── paper1.pdf
    │   ├── paper2.pdf
    │   └── ...
    ├── Author2/
    │   ├── paper1.pdf
    │   └── ...
    └── ...

Features:
- Walks directory tree to find all PDFs
- Extracts metadata (title, year, authors, abstract)
- Deduplicates by MD5 hash
- Builds co-author network edges
- Generates summary statistics CSV
- Safe to rerun (idempotent)
- Progress bars for all operations

Usage:
    python ingest.py --data_dir ../data/papers --db data/authors.db
    python ingest.py --data_dir /path/to/papers --db mydb.db --force

Author: Applied AI Assignment
Date: December 2024
"""

import argparse
import logging
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")

# Import local modules
from db_utils import (
    init_db, upsert_author, upsert_paper, insert_paper_author,
    refresh_coauthor_edges, list_authors, get_author_papers,
    find_author_by_name, get_connection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PDF Discovery and Ingestion
# ============================================================================

def find_all_pdfs(data_dir: Path) -> List[Tuple[Path, str]]:
    """
    Recursively find all PDF files in directory structure.
    
    Args:
        data_dir: Root directory to search
        
    Returns:
        List of (pdf_path, author_folder_name) tuples
        
    Examples:
        >>> pdfs = find_all_pdfs(Path("data/papers"))
        >>> len(pdfs) >= 0
        True
    """
    pdfs = []
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return []
    
    logger.info(f"Scanning for PDFs in: {data_dir}")
    
    # Walk directory tree
    for author_dir in data_dir.iterdir():
        if not author_dir.is_dir():
            continue
        
        author_name = author_dir.name
        
        # Find all PDFs in author's directory
        for pdf_file in author_dir.rglob("*.pdf"):
            if pdf_file.is_file():
                pdfs.append((pdf_file, author_name))
    
    logger.info(f"Found {len(pdfs)} PDF files in {data_dir}")
    return pdfs


def ingest_pdf(
    pdf_path: Path,
    author_name: str,
    db_path: Path,
    force: bool = False
) -> Dict[str, any]:
    """
    Ingest a single PDF file into the database.
    
    Args:
        pdf_path: Path to PDF file
        author_name: Name of primary author (from folder name)
        db_path: Path to database
        force: If True, re-process even if MD5 exists
        
    Returns:
        Dictionary with ingestion result:
        {
            "success": bool,
            "paper_id": int or None,
            "is_new": bool,
            "error": str or None
        }
    """
    result = {
        "success": False,
        "paper_id": None,
        "is_new": False,
        "error": None,
        "skipped": False
    }
    
    try:
        # Import helper functions from parser
        from parser import (
            extract_text_with_fallback, extract_title, extract_year,
            extract_abstract, parse_author_names, compute_md5
        )
        
        # Compute MD5
        md5 = compute_md5(pdf_path)
        
        # Check if already exists (unless force mode)
        if not force:
            with get_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM papers WHERE md5 = ?", (md5,))
                existing = cursor.fetchone()
                
                if existing:
                    result["success"] = True
                    result["paper_id"] = existing[0]
                    result["is_new"] = False
                    result["skipped"] = True
                    logger.debug(f"Skipping duplicate: {pdf_path.name} (MD5: {md5[:8]}...)")
                    return result
        
        # Extract text and metadata
        text, metadata = extract_text_with_fallback(pdf_path)
        
        if not text:
            result["error"] = "No text extracted from PDF"
            logger.warning(f"Failed to extract text: {pdf_path}")
            return result
        
        # Extract components
        title = extract_title(text, pdf_path.name, metadata)
        year = extract_year(text, metadata)
        abstract = extract_abstract(text)
        coauthors = parse_author_names(text, metadata)
        
        # Upsert primary author
        author_id = upsert_author(
            db_path=str(db_path),
            name=author_name,
            affiliation=None
        )
        
        # Upsert paper
        paper_id, is_new = upsert_paper(
            db_path=str(db_path),
            author_id=author_id,
            title=title,
            md5=md5,
            year=year,
            path=str(pdf_path),
            abstract=abstract,
            fulltext=text[:5000] if text else None  # Store first 5000 chars
        )
        
        # Add primary author to paper_authors
        insert_paper_author(
            db_path=str(db_path),
            paper_id=paper_id,
            person_name=author_name,
            author_order=0
        )
        
        # Add co-authors if available
        for idx, coauthor_name in enumerate(coauthors, start=1):
            if coauthor_name and coauthor_name != author_name:
                # Upsert co-author
                upsert_author(db_path=str(db_path), name=coauthor_name)
                
                # Link to paper
                insert_paper_author(
                    db_path=str(db_path),
                    paper_id=paper_id,
                    person_name=coauthor_name,
                    author_order=idx
                )
        
        result["success"] = True
        result["paper_id"] = paper_id
        result["is_new"] = is_new
        
        if is_new:
            logger.info(f"Ingested: {title[:60]}... (ID: {paper_id})")
        else:
            logger.debug(f"Updated: {title[:60]}... (ID: {paper_id})")
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error ingesting {pdf_path}: {e}", exc_info=True)
        return result


def walk_and_ingest(
    data_dir: Path,
    db_path: Path,
    force: bool = False
) -> Dict[str, int]:
    """
    Walk directory tree and ingest all PDFs.
    
    Args:
        data_dir: Root directory containing author folders
        db_path: Path to SQLite database
        force: If True, re-process existing papers
        
    Returns:
        Dictionary with statistics:
        {
            "total_found": int,
            "successful": int,
            "failed": int,
            "skipped": int,
            "new_papers": int,
            "updated_papers": int
        }
    """
    logger.info("=" * 80)
    logger.info("Starting PDF ingestion")
    logger.info("=" * 80)
    
    # Find all PDFs
    pdfs = find_all_pdfs(data_dir)
    
    stats = {
        "total_found": len(pdfs),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "new_papers": 0,
        "updated_papers": 0
    }
    
    if not pdfs:
        logger.warning("No PDF files found!")
        return stats
    
    # Process each PDF with progress bar
    if TQDM_AVAILABLE:
        pdf_iter = tqdm(pdfs, desc="Ingesting PDFs", unit="file")
    else:
        pdf_iter = pdfs
        logger.info(f"Processing {len(pdfs)} PDFs...")
    
    for pdf_path, author_name in pdf_iter:
        result = ingest_pdf(pdf_path, author_name, db_path, force)
        
        if result["success"]:
            stats["successful"] += 1
            if result["skipped"]:
                stats["skipped"] += 1
            elif result["is_new"]:
                stats["new_papers"] += 1
            else:
                stats["updated_papers"] += 1
        else:
            stats["failed"] += 1
            logger.error(f"Failed: {pdf_path.name} - {result['error']}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total PDFs found:    {stats['total_found']}")
    logger.info(f"  Successfully processed: {stats['successful']}")
    logger.info(f"  Failed:              {stats['failed']}")
    logger.info(f"  Skipped (duplicates): {stats['skipped']}")
    logger.info(f"  New papers added:    {stats['new_papers']}")
    logger.info(f"  Papers updated:      {stats['updated_papers']}")
    logger.info("=" * 80)
    
    return stats


# ============================================================================
# Co-author Network Building
# ============================================================================

def build_coauthor_network(db_path: Path) -> int:
    """
    Build co-author edges for all authors in database.
    
    Args:
        db_path: Path to database
        
    Returns:
        Number of authors processed
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Building co-author network")
    logger.info("=" * 80)
    
    # Get all authors
    authors = list_authors(str(db_path))
    
    if not authors:
        logger.warning("No authors found in database")
        return 0
    
    logger.info(f"Processing {len(authors)} authors...")
    
    # Refresh edges for each author with progress bar
    if TQDM_AVAILABLE:
        author_iter = tqdm(authors, desc="Building coauthor edges", unit="author")
    else:
        author_iter = authors
    
    for author in author_iter:
        try:
            edge_count = refresh_coauthor_edges(str(db_path), author["id"])
            logger.debug(f"Author {author['name']}: {edge_count} coauthor edges")
        except Exception as e:
            logger.error(f"Error refreshing edges for author {author['id']}: {e}")
    
    logger.info(f"✓ Co-author network built for {len(authors)} authors")
    logger.info("=" * 80)
    
    return len(authors)


# ============================================================================
# Summary Report Generation
# ============================================================================

def calculate_author_stats(db_path: Path, author_id: int) -> Dict[str, any]:
    """
    Calculate statistics for a single author.
    
    Args:
        db_path: Path to database
        author_id: Author's database ID
        
    Returns:
        Dictionary with statistics:
        {
            "paper_count": int,
            "avg_year": float or None,
            "has_abstract": int (count),
            "has_fulltext": int (count),
            "metadata_percentage": float
        }
    """
    papers = get_author_papers(str(db_path), author_id)
    
    stats = {
        "paper_count": len(papers),
        "avg_year": None,
        "has_abstract": 0,
        "has_fulltext": 0,
        "metadata_percentage": 0.0
    }
    
    if not papers:
        return stats
    
    # Calculate year average
    years = [p["year"] for p in papers if p.get("year")]
    if years:
        stats["avg_year"] = sum(years) / len(years)
    
    # Count metadata availability
    for paper in papers:
        if paper.get("abstract"):
            stats["has_abstract"] += 1
        if paper.get("fulltext"):
            stats["has_fulltext"] += 1
    
    # Calculate metadata percentage (papers with abstract OR fulltext)
    papers_with_metadata = len([
        p for p in papers 
        if p.get("abstract") or p.get("fulltext")
    ])
    stats["metadata_percentage"] = (papers_with_metadata / len(papers)) * 100
    
    return stats


def generate_summary_csv(db_path: Path, output_path: Path) -> None:
    """
    Generate CSV summary report with author statistics.
    
    Args:
        db_path: Path to database
        output_path: Path to output CSV file
        
    CSV Columns:
        - author_name: Name of author
        - paper_count: Number of papers
        - avg_year: Average publication year
        - has_abstract: Count of papers with abstracts
        - has_fulltext: Count of papers with fulltext
        - metadata_percentage: Percentage of papers with metadata
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Generating summary report")
    logger.info("=" * 80)
    
    # Get all authors
    authors = list_authors(str(db_path))
    
    if not authors:
        logger.warning("No authors found in database")
        return
    
    # Calculate stats for each author
    rows = []
    
    if TQDM_AVAILABLE:
        author_iter = tqdm(authors, desc="Calculating statistics", unit="author")
    else:
        author_iter = authors
        logger.info(f"Processing {len(authors)} authors...")
    
    for author in author_iter:
        stats = calculate_author_stats(db_path, author["id"])
        
        rows.append({
            "author_name": author["name"],
            "affiliation": author.get("affiliation", ""),
            "paper_count": stats["paper_count"],
            "avg_year": f"{stats['avg_year']:.1f}" if stats["avg_year"] else "",
            "has_abstract": stats["has_abstract"],
            "has_fulltext": stats["has_fulltext"],
            "metadata_percentage": f"{stats['metadata_percentage']:.1f}"
        })
    
    # Sort by paper count (descending)
    rows.sort(key=lambda x: x["paper_count"], reverse=True)
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "author_name", "affiliation", "paper_count", "avg_year",
            "has_abstract", "has_fulltext", "metadata_percentage"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"✓ Summary report saved to: {output_path}")
    logger.info(f"  Total authors: {len(rows)}")
    logger.info(f"  Total papers: {sum(r['paper_count'] for r in rows)}")
    logger.info("=" * 80)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest academic papers from directory structure into SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses ../Dataset by default)
  python ingest.py

  # Force re-processing of all papers
  python ingest.py --force

  # Custom data directory
  python ingest.py --data_dir /path/to/papers --db mydb.db

Directory Structure Expected:
  Dataset/
  ├── Amit Saxena/
  │   ├── paper1.pdf
  │   └── paper2.pdf
  ├── Amita Jain/
  │   └── paper1.pdf
  └── ...

Output:
  - SQLite database with papers, authors, and co-author network
  - CSV summary report with statistics
  - Log file (ingest.log)
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../Dataset",
        help="Root directory containing author folders with PDFs (default: ../Dataset)"
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default="data/papers.db",
        help="Path to SQLite database (default: data/papers.db)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/ingest_summary.csv",
        help="Path to output CSV summary (default: data/ingest_summary.csv)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of existing papers (default: skip duplicates)"
    )
    
    parser.add_argument(
        "--skip-coauthors",
        action="store_true",
        help="Skip building co-author network (faster, but no COI detection)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert paths
    data_dir = Path(args.data_dir)
    db_path = Path(args.db)
    output_path = Path(args.output)
    
    # Print configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info("PDF Ingestion Script")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Database:       {db_path}")
    logger.info(f"Output CSV:     {output_path}")
    logger.info(f"Force mode:     {args.force}")
    logger.info(f"Skip coauthors: {args.skip_coauthors}")
    logger.info("=" * 80)
    
    # Validate data directory
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    try:
        # Step 1: Initialize database
        logger.info("")
        logger.info("Step 1: Initializing database...")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        init_db(str(db_path))
        logger.info("✓ Database initialized")
        
        # Step 2: Ingest all PDFs
        logger.info("")
        logger.info("Step 2: Ingesting PDFs...")
        stats = walk_and_ingest(data_dir, db_path, force=args.force)
        
        if stats["failed"] > 0:
            logger.warning(f"⚠ {stats['failed']} PDFs failed to process")
        
        # Step 3: Build co-author network (optional)
        if not args.skip_coauthors:
            logger.info("")
            logger.info("Step 3: Building co-author network...")
            author_count = build_coauthor_network(db_path)
            logger.info(f"✓ Co-author network built for {author_count} authors")
        else:
            logger.info("")
            logger.info("Step 3: Skipping co-author network (--skip-coauthors)")
        
        # Step 4: Generate summary report
        logger.info("")
        logger.info("Step 4: Generating summary report...")
        generate_summary_csv(db_path, output_path)
        
        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("INGESTION COMPLETE ✓")
        logger.info("=" * 80)
        logger.info(f"Database:       {db_path}")
        logger.info(f"Summary CSV:    {output_path}")
        logger.info(f"Log file:       ingest.log")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review summary CSV for statistics")
        logger.info("  2. Check ingest.log for any errors")
        logger.info("  3. Run again with --force to re-process papers")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

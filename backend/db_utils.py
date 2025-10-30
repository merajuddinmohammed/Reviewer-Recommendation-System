"""
SQLite Database Utilities for Paper Management System

Provides schema initialization, upsert operations, and query helpers
for managing academic papers, authors, and their relationships.

Features:
- SQLite with WAL mode for better concurrency
- Safe upsert operations with duplicate prevention
- Author and co-author relationship management
- Vector metadata tracking for FAISS integration
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def get_connection(db_path: str):
    """
    Context manager for SQLite connections with WAL mode enabled.
    
    Args:
        db_path: Path to SQLite database file
        
    Yields:
        sqlite3.Connection: Database connection with row factory
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    
    # Enable Write-Ahead Logging for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")  # Enforce foreign key constraints
    
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """
    Initialize database schema with all required tables.
    
    Creates tables if they don't exist:
    - authors: Main author entities
    - papers: Paper metadata and content
    - paper_authors: Many-to-many relationship for co-authors
    - coauthors: Derived co-author network edges
    - paper_vectors: Metadata for FAISS vector storage
    
    Args:
        db_path: Path to SQLite database file
    """
    logger.info(f"Initializing database at: {db_path}")
    
    # Ensure directory exists
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Authors table - primary entities
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                affiliation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Papers table - main content storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                author_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                year INTEGER,
                path TEXT,
                abstract TEXT,
                fulltext TEXT,
                md5 TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE
            )
        """)
        
        # Paper authors junction table - co-author relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                person_name TEXT NOT NULL,
                author_order INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                UNIQUE(paper_id, person_name)
            )
        """)
        
        # Co-authors table - derived network edges
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coauthors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                author_id INTEGER NOT NULL,
                coauthor_name TEXT NOT NULL,
                collaboration_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE,
                UNIQUE(author_id, coauthor_name)
            )
        """)
        
        # Paper vectors metadata - FAISS integration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL UNIQUE,
                dim INTEGER NOT NULL,
                norm REAL DEFAULT 0.0,
                faiss_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_author_id 
            ON papers(author_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_md5 
            ON papers(md5)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_year 
            ON papers(year)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_authors_paper_id 
            ON paper_authors(paper_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_authors_person_name 
            ON paper_authors(person_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_coauthors_author_id 
            ON coauthors(author_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_coauthors_name 
            ON coauthors(coauthor_name)
        """)
        
        conn.commit()
        logger.info("Database schema initialized successfully")


def upsert_author(db_path: str, name: str, affiliation: Optional[str] = None) -> int:
    """
    Insert or update an author record.
    
    Args:
        db_path: Path to SQLite database
        name: Author name (unique identifier)
        affiliation: Author's institutional affiliation
        
    Returns:
        int: Author ID (existing or newly created)
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Try to get existing author
        cursor.execute("SELECT id FROM authors WHERE name = ?", (name,))
        result = cursor.fetchone()
        
        if result:
            author_id = result[0]
            # Update affiliation if provided
            if affiliation:
                cursor.execute("""
                    UPDATE authors 
                    SET affiliation = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (affiliation, author_id))
            logger.debug(f"Updated existing author: {name} (ID: {author_id})")
        else:
            # Insert new author
            cursor.execute("""
                INSERT INTO authors (name, affiliation) 
                VALUES (?, ?)
            """, (name, affiliation))
            author_id = cursor.lastrowid
            logger.info(f"Created new author: {name} (ID: {author_id})")
        
        conn.commit()
        return author_id


def upsert_paper(
    db_path: str,
    author_id: int,
    title: str,
    md5: str,
    year: Optional[int] = None,
    path: Optional[str] = None,
    abstract: Optional[str] = None,
    fulltext: Optional[str] = None
) -> Tuple[int, bool]:
    """
    Insert or update a paper record.
    
    Papers are uniquely identified by MD5 hash to prevent duplicates.
    
    Args:
        db_path: Path to SQLite database
        author_id: Primary author's ID
        title: Paper title
        md5: MD5 hash of paper content (unique identifier)
        year: Publication year
        path: File path to PDF/source
        abstract: Paper abstract
        fulltext: Extracted full text
        
    Returns:
        Tuple[int, bool]: (paper_id, is_new)
            - paper_id: Database ID of paper
            - is_new: True if newly created, False if updated
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if paper exists by MD5
        cursor.execute("SELECT id FROM papers WHERE md5 = ?", (md5,))
        result = cursor.fetchone()
        
        if result:
            paper_id = result[0]
            # Update existing paper
            cursor.execute("""
                UPDATE papers 
                SET title = ?, 
                    year = ?, 
                    path = ?, 
                    abstract = ?, 
                    fulltext = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (title, year, path, abstract, fulltext, paper_id))
            logger.debug(f"Updated existing paper: {title} (ID: {paper_id})")
            is_new = False
        else:
            # Insert new paper
            cursor.execute("""
                INSERT INTO papers (author_id, title, year, path, abstract, fulltext, md5)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (author_id, title, year, path, abstract, fulltext, md5))
            paper_id = cursor.lastrowid
            logger.info(f"Created new paper: {title} (ID: {paper_id})")
            is_new = True
        
        conn.commit()
        return paper_id, is_new


def insert_paper_author(
    db_path: str,
    paper_id: int,
    person_name: str,
    author_order: int = 0
) -> None:
    """
    Add a co-author to a paper.
    
    Args:
        db_path: Path to SQLite database
        paper_id: Paper ID
        person_name: Co-author's name
        author_order: Position in author list (0-indexed)
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Use INSERT OR IGNORE to handle duplicates gracefully
        cursor.execute("""
            INSERT OR IGNORE INTO paper_authors (paper_id, person_name, author_order)
            VALUES (?, ?, ?)
        """, (paper_id, person_name, author_order))
        
        if cursor.rowcount > 0:
            logger.debug(f"Added co-author '{person_name}' to paper {paper_id}")
        
        conn.commit()


def refresh_coauthor_edges(db_path: str, author_id: int) -> int:
    """
    Rebuild co-author network edges for a given author.
    
    Analyzes all papers by the author and creates/updates co-author relationships.
    
    Args:
        db_path: Path to SQLite database
        author_id: Author ID to refresh edges for
        
    Returns:
        int: Number of co-author edges created/updated
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Get author's name
        cursor.execute("SELECT name FROM authors WHERE id = ?", (author_id,))
        author_result = cursor.fetchone()
        if not author_result:
            logger.warning(f"Author ID {author_id} not found")
            return 0
        
        author_name = author_result[0]
        
        # Delete existing co-author edges
        cursor.execute("DELETE FROM coauthors WHERE author_id = ?", (author_id,))
        
        # Find all co-authors from papers
        cursor.execute("""
            SELECT DISTINCT pa.person_name, COUNT(*) as collab_count
            FROM papers p
            JOIN paper_authors pa ON p.id = pa.paper_id
            WHERE p.author_id = ?
              AND pa.person_name != ?
            GROUP BY pa.person_name
        """, (author_id, author_name))
        
        coauthors_data = cursor.fetchall()
        edge_count = 0
        
        for row in coauthors_data:
            coauthor_name = row[0]
            collab_count = row[1]
            
            cursor.execute("""
                INSERT INTO coauthors (author_id, coauthor_name, collaboration_count)
                VALUES (?, ?, ?)
            """, (author_id, coauthor_name, collab_count))
            edge_count += 1
        
        conn.commit()
        logger.info(f"Refreshed {edge_count} co-author edges for author {author_id}")
        return edge_count


def get_all_papers(db_path: str) -> List[Dict[str, Any]]:
    """
    Retrieve all papers from the database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        List of paper dictionaries with all fields
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                p.*,
                a.name as author_name,
                a.affiliation as author_affiliation
            FROM papers p
            JOIN authors a ON p.author_id = a.id
            ORDER BY p.year DESC, p.title ASC
        """)
        
        rows = cursor.fetchall()
        papers = [dict(row) for row in rows]
        logger.debug(f"Retrieved {len(papers)} papers")
        return papers


def get_author_papers(db_path: str, author_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve all papers by a specific author.
    
    Args:
        db_path: Path to SQLite database
        author_id: Author ID
        
    Returns:
        List of paper dictionaries
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                p.*,
                a.name as author_name,
                a.affiliation as author_affiliation
            FROM papers p
            JOIN authors a ON p.author_id = a.id
            WHERE p.author_id = ?
            ORDER BY p.year DESC, p.title ASC
        """, (author_id,))
        
        rows = cursor.fetchall()
        papers = [dict(row) for row in rows]
        logger.debug(f"Retrieved {len(papers)} papers for author {author_id}")
        return papers


def find_author_by_name(db_path: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Find an author by exact name match.
    
    Args:
        db_path: Path to SQLite database
        name: Author name to search for
        
    Returns:
        Author dictionary or None if not found
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM authors WHERE name = ?
        """, (name,))
        
        row = cursor.fetchone()
        if row:
            author = dict(row)
            logger.debug(f"Found author: {name} (ID: {author['id']})")
            return author
        else:
            logger.debug(f"Author not found: {name}")
            return None


def list_authors(db_path: str) -> List[Dict[str, Any]]:
    """
    Retrieve all authors from the database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        List of author dictionaries
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                a.*,
                COUNT(p.id) as paper_count
            FROM authors a
            LEFT JOIN papers p ON a.id = p.author_id
            GROUP BY a.id
            ORDER BY a.name ASC
        """)
        
        rows = cursor.fetchall()
        authors = [dict(row) for row in rows]
        logger.debug(f"Retrieved {len(authors)} authors")
        return authors


def get_coauthors(db_path: str, author_id: int) -> List[Dict[str, Any]]:
    """
    Get all co-authors for a given author.
    
    Args:
        db_path: Path to SQLite database
        author_id: Author ID
        
    Returns:
        List of co-author dictionaries with collaboration counts
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM coauthors 
            WHERE author_id = ?
            ORDER BY collaboration_count DESC, coauthor_name ASC
        """, (author_id,))
        
        rows = cursor.fetchall()
        coauthors = [dict(row) for row in rows]
        return coauthors


def insert_paper_vector(
    db_path: str,
    paper_id: int,
    dim: int,
    norm: float = 0.0,
    faiss_index: Optional[int] = None
) -> None:
    """
    Insert or update vector metadata for a paper.
    
    Args:
        db_path: Path to SQLite database
        paper_id: Paper ID
        dim: Vector dimensionality
        norm: Vector L2 norm
        faiss_index: Index position in FAISS
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO paper_vectors (paper_id, dim, norm, faiss_index)
            VALUES (?, ?, ?, ?)
        """, (paper_id, dim, norm, faiss_index))
        
        conn.commit()
        logger.debug(f"Inserted vector metadata for paper {paper_id}")


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """
    Self-contained test suite for database utilities.
    
    Creates a temporary database and performs basic CRUD operations.
    """
    import tempfile
    import os
    
    print("=" * 70)
    print("Running Database Utility Tests")
    print("=" * 70)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as tmp:
        test_db_path = tmp.name
    
    try:
        # Test 1: Initialize database
        print("\n[TEST 1] Initializing database...")
        init_db(test_db_path)
        print("✓ Database initialized successfully")
        
        # Test 2: Upsert authors
        print("\n[TEST 2] Creating authors...")
        author1_id = upsert_author(test_db_path, "Alice Smith", "MIT")
        author2_id = upsert_author(test_db_path, "Bob Johnson", "Stanford")
        author3_id = upsert_author(test_db_path, "Carol Williams", "Berkeley")
        print(f"✓ Created 3 authors (IDs: {author1_id}, {author2_id}, {author3_id})")
        
        # Test 3: Upsert duplicate author (should return existing ID)
        print("\n[TEST 3] Testing duplicate author handling...")
        duplicate_id = upsert_author(test_db_path, "Alice Smith", "MIT CSAIL")
        assert duplicate_id == author1_id, "Should return existing author ID"
        print(f"✓ Duplicate handled correctly (ID: {duplicate_id})")
        
        # Test 4: Insert papers
        print("\n[TEST 4] Creating papers...")
        paper1_id, is_new1 = upsert_paper(
            test_db_path,
            author_id=author1_id,
            title="Deep Learning Fundamentals",
            md5="abc123def456",
            year=2023,
            abstract="A comprehensive study on deep learning.",
            fulltext="Full text content here..."
        )
        
        paper2_id, is_new2 = upsert_paper(
            test_db_path,
            author_id=author1_id,
            title="Neural Networks in Practice",
            md5="xyz789uvw012",
            year=2024,
            abstract="Practical applications of neural networks."
        )
        
        paper3_id, is_new3 = upsert_paper(
            test_db_path,
            author_id=author2_id,
            title="Transformers Explained",
            md5="mno345pqr678",
            year=2024,
            abstract="Understanding transformer architectures."
        )
        
        assert is_new1 and is_new2 and is_new3, "All papers should be new"
        print(f"✓ Created 3 papers (IDs: {paper1_id}, {paper2_id}, {paper3_id})")
        
        # Test 5: Duplicate paper (same MD5)
        print("\n[TEST 5] Testing duplicate paper handling...")
        paper1_dup_id, is_new_dup = upsert_paper(
            test_db_path,
            author_id=author1_id,
            title="Deep Learning Fundamentals (Updated)",
            md5="abc123def456",  # Same MD5
            year=2023
        )
        assert paper1_dup_id == paper1_id, "Should return existing paper ID"
        assert not is_new_dup, "Should not be flagged as new"
        print(f"✓ Duplicate paper handled correctly (ID: {paper1_dup_id})")
        
        # Test 6: Add co-authors
        print("\n[TEST 6] Adding co-authors...")
        insert_paper_author(test_db_path, paper1_id, "Alice Smith", 0)
        insert_paper_author(test_db_path, paper1_id, "Bob Johnson", 1)
        insert_paper_author(test_db_path, paper1_id, "Carol Williams", 2)
        
        insert_paper_author(test_db_path, paper2_id, "Alice Smith", 0)
        insert_paper_author(test_db_path, paper2_id, "Carol Williams", 1)
        
        insert_paper_author(test_db_path, paper3_id, "Bob Johnson", 0)
        insert_paper_author(test_db_path, paper3_id, "Alice Smith", 1)
        print("✓ Co-authors added successfully")
        
        # Test 7: Refresh co-author edges
        print("\n[TEST 7] Refreshing co-author network...")
        edge_count1 = refresh_coauthor_edges(test_db_path, author1_id)
        edge_count2 = refresh_coauthor_edges(test_db_path, author2_id)
        print(f"✓ Alice has {edge_count1} co-authors")
        print(f"✓ Bob has {edge_count2} co-authors")
        
        # Test 8: Query all papers
        print("\n[TEST 8] Querying all papers...")
        all_papers = get_all_papers(test_db_path)
        assert len(all_papers) == 3, f"Expected 3 papers, got {len(all_papers)}"
        print(f"✓ Retrieved {len(all_papers)} papers")
        
        # Test 9: Query author papers
        print("\n[TEST 9] Querying papers by author...")
        alice_papers = get_author_papers(test_db_path, author1_id)
        assert len(alice_papers) == 2, f"Expected 2 papers, got {len(alice_papers)}"
        print(f"✓ Alice has {len(alice_papers)} papers")
        
        # Test 10: Find author by name
        print("\n[TEST 10] Finding author by name...")
        found_author = find_author_by_name(test_db_path, "Bob Johnson")
        assert found_author is not None, "Author should be found"
        assert found_author['id'] == author2_id, "Should return correct author"
        print(f"✓ Found author: {found_author['name']} (ID: {found_author['id']})")
        
        # Test 11: List all authors
        print("\n[TEST 11] Listing all authors...")
        all_authors = list_authors(test_db_path)
        assert len(all_authors) == 3, f"Expected 3 authors, got {len(all_authors)}"
        print(f"✓ Retrieved {len(all_authors)} authors")
        for author in all_authors:
            print(f"  - {author['name']}: {author['paper_count']} papers")
        
        # Test 12: Get co-authors
        print("\n[TEST 12] Getting co-authors...")
        alice_coauthors = get_coauthors(test_db_path, author1_id)
        print(f"✓ Alice's co-authors ({len(alice_coauthors)}):")
        for coauthor in alice_coauthors:
            print(f"  - {coauthor['coauthor_name']}: {coauthor['collaboration_count']} collaborations")
        
        # Test 13: Vector metadata
        print("\n[TEST 13] Adding vector metadata...")
        insert_paper_vector(test_db_path, paper1_id, dim=768, norm=1.0, faiss_index=0)
        insert_paper_vector(test_db_path, paper2_id, dim=768, norm=0.95, faiss_index=1)
        print("✓ Vector metadata added successfully")
        
        # Test 14: Verify data integrity
        print("\n[TEST 14] Verifying data integrity...")
        with get_connection(test_db_path) as conn:
            cursor = conn.cursor()
            
            # Check foreign key constraints
            cursor.execute("SELECT COUNT(*) FROM papers WHERE author_id NOT IN (SELECT id FROM authors)")
            orphaned_papers = cursor.fetchone()[0]
            assert orphaned_papers == 0, "No orphaned papers should exist"
            
            # Check unique constraints
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT md5) FROM papers")
            unique_md5 = cursor.fetchone()[0]
            assert total_papers == unique_md5, "All papers should have unique MD5"
            
            print("✓ Data integrity verified")
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
            print(f"\nCleaned up test database: {test_db_path}")


if __name__ == "__main__":
    run_tests()

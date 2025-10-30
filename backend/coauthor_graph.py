"""
Co-author Graph and Conflict-of-Interest Detection

This module builds co-author relationship graphs from the paper database and
detects potential conflicts of interest for peer review assignments.

Conflict-of-Interest (COI) Rules:
1. Same Person: Candidate name matches query author name (case-insensitive)
2. Is Coauthor: Candidate has co-authored papers with query author
3. Same Affiliation: Candidate shares affiliation with query author (case-insensitive)

These rules help identify reviewers who should be excluded from reviewing a paper.

Features:
- Build co-author edges from database
- Normalize names and affiliations for matching
- Detect conflicts with multiple criteria
- Safe handling of name collisions
- Comprehensive test suite

Author: Applied AI Assignment
Date: December 2024
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

# Import database utilities
from db_utils import get_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions for Name and Affiliation Normalization
# ============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize author name for matching.
    
    Applies:
    - Case folding (lowercase)
    - Whitespace normalization
    - Strip leading/trailing whitespace
    
    Args:
        name: Author name to normalize
        
    Returns:
        Normalized name string
        
    Examples:
        >>> normalize_name("John Smith")
        'john smith'
        >>> normalize_name("  JANE  DOE  ")
        'jane doe'
        >>> normalize_name("Mary-Jane Watson")
        'mary-jane watson'
    """
    if not name:
        return ""
    
    # Case fold and normalize whitespace
    normalized = name.strip().lower()
    # Replace multiple spaces with single space
    normalized = " ".join(normalized.split())
    
    return normalized


def normalize_affiliation(affiliation: str) -> str:
    """
    Normalize affiliation string for matching.
    
    Applies:
    - Case folding (lowercase)
    - Whitespace normalization
    - Strip leading/trailing whitespace
    
    Args:
        affiliation: Affiliation string to normalize
        
    Returns:
        Normalized affiliation string
        
    Examples:
        >>> normalize_affiliation("Stanford University")
        'stanford university'
        >>> normalize_affiliation("MIT  CSAIL")
        'mit csail'
        >>> normalize_affiliation("  UC Berkeley  ")
        'uc berkeley'
    """
    if not affiliation:
        return ""
    
    # Case fold and normalize whitespace
    normalized = affiliation.strip().lower()
    # Replace multiple spaces with single space
    normalized = " ".join(normalized.split())
    
    return normalized


def names_match(name1: str, name2: str) -> bool:
    """
    Check if two names match (case-insensitive).
    
    Args:
        name1: First name
        name2: Second name
        
    Returns:
        True if names match after normalization
        
    Examples:
        >>> names_match("John Smith", "john smith")
        True
        >>> names_match("Jane Doe", "Jane  DOE")
        True
        >>> names_match("Alice", "Bob")
        False
    """
    return normalize_name(name1) == normalize_name(name2)


def affiliation_match(affiliation1: str, affiliation2: str) -> bool:
    """
    Check if two affiliations match (case-insensitive).
    
    Args:
        affiliation1: First affiliation
        affiliation2: Second affiliation
        
    Returns:
        True if affiliations match after normalization
        
    Examples:
        >>> affiliation_match("Stanford University", "stanford university")
        True
        >>> affiliation_match("MIT", "mit")
        True
        >>> affiliation_match("Stanford", "MIT")
        False
    """
    norm1 = normalize_affiliation(affiliation1)
    norm2 = normalize_affiliation(affiliation2)
    
    # Both empty means no match (no affiliation information)
    if not norm1 or not norm2:
        return False
    
    return norm1 == norm2


# ============================================================================
# Co-author Graph Building
# ============================================================================

def build_edges_for_author(
    db_path: Path,
    author_id: int
) -> List[Tuple[int, int, int]]:
    """
    Build co-author edges for a given author.
    
    Queries the papers table and paper_authors table to find all authors who have co-authored
    papers with the specified author. Returns edges as (author_id, coauthor_id, count)
    tuples where count is the number of papers they've co-authored together.
    
    Args:
        db_path: Path to SQLite database
        author_id: Database ID of the author
        
    Returns:
        List of (author_id, coauthor_id, paper_count) tuples
        Empty list if author has no papers or no coauthors
        
    Examples:
        >>> # Assuming author 1 has co-authored with authors 2 and 3
        >>> edges = build_edges_for_author(Path("test.db"), 1)
        >>> len(edges) >= 0  # May have coauthors or not
        True
    """
    edges = []
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Step 1: Get author name
            cursor.execute("SELECT name FROM authors WHERE id = ?", (author_id,))
            author_row = cursor.fetchone()
            
            if not author_row:
                logger.warning(f"Author {author_id} not found")
                return []
            
            author_name = author_row[0]
            
            # Step 2: Find all papers where this author is listed
            cursor.execute("""
                SELECT DISTINCT paper_id 
                FROM paper_authors 
                WHERE person_name = ?
            """, (author_name,))
            
            paper_ids = [row[0] for row in cursor.fetchall()]
            
            if not paper_ids:
                logger.info(f"Author {author_id} has no papers")
                return []
            
            # Step 3: For each paper, find coauthor names
            coauthor_name_counts = Counter()
            
            for paper_id in paper_ids:
                cursor.execute("""
                    SELECT person_name 
                    FROM paper_authors 
                    WHERE paper_id = ? AND person_name != ?
                """, (paper_id, author_name))
                
                coauthor_names = [row[0] for row in cursor.fetchall()]
                
                # Count co-authorship instances
                for coauthor_name in coauthor_names:
                    coauthor_name_counts[coauthor_name] += 1
            
            # Step 4: Convert coauthor names to IDs and build edges
            for coauthor_name, count in coauthor_name_counts.items():
                cursor.execute("""
                    SELECT id FROM authors WHERE name = ?
                """, (coauthor_name,))
                
                coauthor_row = cursor.fetchone()
                
                if coauthor_row:
                    coauthor_id = coauthor_row[0]
                    edges.append((author_id, coauthor_id, count))
            
            logger.info(f"Author {author_id} has {len(edges)} coauthor edges")
            return edges
            
    except sqlite3.Error as e:
        logger.error(f"Database error building edges for author {author_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error building edges for author {author_id}: {e}")
        return []


def build_coauthor_graph(db_path: Path) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build complete co-author graph for all authors in database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary mapping author_id to list of (coauthor_id, paper_count) tuples
        
    Examples:
        >>> graph = build_coauthor_graph(Path("test.db"))
        >>> isinstance(graph, dict)
        True
    """
    graph = defaultdict(list)
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all author IDs
            cursor.execute("SELECT id FROM authors")
            author_ids = [row[0] for row in cursor.fetchall()]
            
            # Build edges for each author
            for author_id in author_ids:
                edges = build_edges_for_author(db_path, author_id)
                graph[author_id] = [(coauthor_id, count) for _, coauthor_id, count in edges]
            
            logger.info(f"Built coauthor graph with {len(graph)} authors")
            return dict(graph)
            
    except Exception as e:
        logger.error(f"Error building coauthor graph: {e}")
        return {}


# ============================================================================
# Conflict-of-Interest Detection
# ============================================================================

def conflicts_for_candidate(
    candidate_name: str,
    query_author_names: List[str],
    query_affiliations: List[str],
    db_path: Optional[Path] = None,
    candidate_affiliation: Optional[str] = None
) -> Dict[str, bool]:
    """
    Detect conflicts of interest for a candidate reviewer.
    
    Checks three types of conflicts:
    1. is_same_person: Candidate name matches any query author name (case-insensitive)
    2. is_coauthor: Candidate has co-authored with any query author (requires db_path)
    3. same_affiliation: Candidate shares affiliation with any query author (case-insensitive)
    
    Safe handling of name collisions:
    - Uses normalized (case-folded) string matching
    - Exact match required after normalization
    - No partial matching (avoids false positives)
    
    Args:
        candidate_name: Name of candidate reviewer
        query_author_names: List of author names to check against
        query_affiliations: List of author affiliations to check against
        db_path: Optional path to database (required for coauthor check)
        candidate_affiliation: Optional affiliation of candidate
        
    Returns:
        Dictionary with boolean flags:
        {
            "is_same_person": bool,    # Name matches query author
            "is_coauthor": bool,        # Has co-authored with query author
            "same_affiliation": bool    # Shares affiliation with query author
        }
        
    Examples:
        >>> # Same person check
        >>> conflicts_for_candidate("John Smith", ["john smith"], [])
        {'is_same_person': True, 'is_coauthor': False, 'same_affiliation': False}
        
        >>> # Same affiliation check
        >>> conflicts_for_candidate("Alice", ["Bob"], ["MIT"], candidate_affiliation="MIT")
        {'is_same_person': False, 'is_coauthor': False, 'same_affiliation': True}
        
        >>> # No conflicts
        >>> conflicts_for_candidate("Alice", ["Bob"], ["Stanford"], candidate_affiliation="MIT")
        {'is_same_person': False, 'is_coauthor': False, 'same_affiliation': False}
    """
    # Initialize result with all conflicts False
    result = {
        "is_same_person": False,
        "is_coauthor": False,
        "same_affiliation": False
    }
    
    # Validate inputs
    if not candidate_name:
        logger.warning("Empty candidate_name provided")
        return result
    
    if not query_author_names:
        logger.warning("Empty query_author_names list provided")
        return result
    
    # Normalize candidate name once
    candidate_norm = normalize_name(candidate_name)
    
    # -------------------------------------------------------------------------
    # Check 1: Same Person (name matching)
    # -------------------------------------------------------------------------
    for author_name in query_author_names:
        if names_match(candidate_name, author_name):
            result["is_same_person"] = True
            logger.info(f"Same person detected: '{candidate_name}' matches '{author_name}'")
            break
    
    # -------------------------------------------------------------------------
    # Check 2: Is Coauthor (database lookup)
    # -------------------------------------------------------------------------
    if db_path and db_path.exists():
        try:
            with get_connection(db_path) as conn:
                cursor = conn.cursor()
                
                # Check if candidate has co-authored with any query author
                for query_name in query_author_names:
                    query_norm = normalize_name(query_name)
                    
                    # Check if they share any papers using person_name
                    cursor.execute("""
                        SELECT COUNT(DISTINCT pa1.paper_id) 
                        FROM paper_authors pa1
                        JOIN paper_authors pa2 ON pa1.paper_id = pa2.paper_id
                        WHERE LOWER(pa1.person_name) = ? 
                        AND LOWER(pa2.person_name) = ?
                        AND pa1.person_name != pa2.person_name
                    """, (candidate_norm, query_norm))
                    
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        result["is_coauthor"] = True
                        logger.info(f"Coauthor detected: '{candidate_name}' has "
                                  f"co-authored {count} papers with '{query_name}'")
                        break
                
        except sqlite3.Error as e:
            logger.error(f"Database error checking coauthor status: {e}")
        except Exception as e:
            logger.error(f"Unexpected error checking coauthor status: {e}")
    
    # -------------------------------------------------------------------------
    # Check 3: Same Affiliation
    # -------------------------------------------------------------------------
    if candidate_affiliation:
        candidate_aff_norm = normalize_affiliation(candidate_affiliation)
        
        if candidate_aff_norm:  # Only check if not empty after normalization
            for query_aff in query_affiliations:
                if affiliation_match(candidate_affiliation, query_aff):
                    result["same_affiliation"] = True
                    logger.info(f"Same affiliation detected: '{candidate_affiliation}' "
                              f"matches '{query_aff}'")
                    break
    
    return result


def get_coauthors_for_author(
    db_path: Path,
    author_name: str
) -> List[Dict[str, any]]:
    """
    Get list of coauthors for a given author name.
    
    Args:
        db_path: Path to SQLite database
        author_name: Name of the author
        
    Returns:
        List of dictionaries with coauthor information:
        [
            {
                "author_id": int,
                "name": str,
                "paper_count": int
            },
            ...
        ]
        
    Examples:
        >>> coauthors = get_coauthors_for_author(Path("test.db"), "John Smith")
        >>> isinstance(coauthors, list)
        True
    """
    coauthors = []
    
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get author ID
            author_norm = normalize_name(author_name)
            cursor.execute("""
                SELECT id FROM authors WHERE LOWER(name) = ?
            """, (author_norm,))
            
            author_row = cursor.fetchone()
            
            if not author_row:
                logger.warning(f"Author '{author_name}' not found in database")
                return []
            
            author_id = author_row[0]
            
            # Build edges for this author
            edges = build_edges_for_author(db_path, author_id)
            
            # Get coauthor details
            for _, coauthor_id, paper_count in edges:
                cursor.execute("""
                    SELECT name FROM authors WHERE id = ?
                """, (coauthor_id,))
                
                name_row = cursor.fetchone()
                
                if name_row:
                    coauthors.append({
                        "author_id": coauthor_id,
                        "name": name_row[0],
                        "paper_count": paper_count
                    })
            
            # Sort by paper count (most collaborations first)
            coauthors.sort(key=lambda x: x["paper_count"], reverse=True)
            
            return coauthors
            
    except Exception as e:
        logger.error(f"Error getting coauthors for '{author_name}': {e}")
        return []


def has_conflict(
    candidate_name: str,
    query_author_names: List[str],
    query_affiliations: List[str],
    db_path: Optional[Path] = None,
    candidate_affiliation: Optional[str] = None
) -> bool:
    """
    Check if candidate has ANY conflict of interest.
    
    Convenience function that returns True if any COI flag is True.
    
    Args:
        candidate_name: Name of candidate reviewer
        query_author_names: List of author names to check against
        query_affiliations: List of author affiliations to check against
        db_path: Optional path to database (required for coauthor check)
        candidate_affiliation: Optional affiliation of candidate
        
    Returns:
        True if any conflict detected, False otherwise
        
    Examples:
        >>> has_conflict("John Smith", ["john smith"], [])
        True
        >>> has_conflict("Alice", ["Bob"], [])
        False
    """
    conflicts = conflicts_for_candidate(
        candidate_name,
        query_author_names,
        query_affiliations,
        db_path,
        candidate_affiliation
    )
    
    return any(conflicts.values())


# ============================================================================
# Main Demo and Tests
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    from db_utils import (
        init_db, upsert_paper, upsert_author,
        insert_paper_author, list_authors
    )
    
    print("=" * 80)
    print("Co-author Graph & COI Detection - Demo and Tests")
    print("=" * 80)
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
        tmp_db_path = Path(tmp.name)
    
    try:
        print("\n1. Setting up test database...")
        init_db(tmp_db_path)
        
        # Create test authors
        alice_id = upsert_author(tmp_db_path, "Alice Smith", "MIT")
        bob_id = upsert_author(tmp_db_path, "Bob Jones", "Stanford")
        charlie_id = upsert_author(tmp_db_path, "Charlie Brown", "MIT")
        dave_id = upsert_author(tmp_db_path, "Dave Wilson", "Berkeley")
        eve_id = upsert_author(tmp_db_path, "Eve Davis", "Stanford")
        
        print(f"   Created 5 authors: Alice, Bob, Charlie, Dave, Eve")
        
        # Create test papers
        paper1_id, _ = upsert_paper(
            db_path=tmp_db_path,
            author_id=alice_id,
            title="Deep Learning Fundamentals",
            md5="hash1",
            abstract="A paper on deep learning.",
            year=2023
        )
        
        paper2_id, _ = upsert_paper(
            db_path=tmp_db_path,
            author_id=alice_id,
            title="Neural Networks",
            md5="hash2",
            abstract="A paper on neural networks.",
            year=2023
        )
        
        paper3_id, _ = upsert_paper(
            db_path=tmp_db_path,
            author_id=bob_id,
            title="Computer Vision",
            md5="hash3",
            abstract="A paper on computer vision.",
            year=2024
        )
        
        print(f"   Created 3 papers")
        
        # Link authors to papers (create collaborations)
        # Paper 1: Alice + Bob (MIT + Stanford)
        insert_paper_author(tmp_db_path, paper1_id, "Alice Smith")
        insert_paper_author(tmp_db_path, paper1_id, "Bob Jones")
        
        # Paper 2: Alice + Charlie (MIT + MIT)
        insert_paper_author(tmp_db_path, paper2_id, "Alice Smith")
        insert_paper_author(tmp_db_path, paper2_id, "Charlie Brown")
        
        # Paper 3: Bob + Eve (Stanford + Stanford)
        insert_paper_author(tmp_db_path, paper3_id, "Bob Jones")
        insert_paper_author(tmp_db_path, paper3_id, "Eve Davis")
        
        print(f"   Linked authors to papers (created collaborations)")
        print(f"   - Paper 1: Alice + Bob")
        print(f"   - Paper 2: Alice + Charlie")
        print(f"   - Paper 3: Bob + Eve")
        
        # -----------------------------------------------------------------------
        # Test 1: Build co-author edges
        # -----------------------------------------------------------------------
        print("\n2. Testing build_edges_for_author()...")
        
        alice_edges = build_edges_for_author(tmp_db_path, alice_id)
        print(f"   Alice's coauthor edges: {alice_edges}")
        assert len(alice_edges) == 2, "Alice should have 2 coauthors"
        assert (alice_id, bob_id, 1) in alice_edges, "Alice-Bob edge missing"
        assert (alice_id, charlie_id, 1) in alice_edges, "Alice-Charlie edge missing"
        print(f"   ✓ Alice has 2 coauthors (Bob, Charlie)")
        
        bob_edges = build_edges_for_author(tmp_db_path, bob_id)
        print(f"   Bob's coauthor edges: {bob_edges}")
        assert len(bob_edges) == 2, "Bob should have 2 coauthors"
        assert (bob_id, alice_id, 1) in bob_edges, "Bob-Alice edge missing"
        assert (bob_id, eve_id, 1) in bob_edges, "Bob-Eve edge missing"
        print(f"   ✓ Bob has 2 coauthors (Alice, Eve)")
        
        dave_edges = build_edges_for_author(tmp_db_path, dave_id)
        print(f"   Dave's coauthor edges: {dave_edges}")
        assert len(dave_edges) == 0, "Dave should have no coauthors"
        print(f"   ✓ Dave has 0 coauthors (no papers)")
        
        # -----------------------------------------------------------------------
        # Test 2: Conflict detection - Same person
        # -----------------------------------------------------------------------
        print("\n3. Testing conflicts_for_candidate() - Same person...")
        
        conflicts = conflicts_for_candidate(
            candidate_name="alice smith",  # Lowercase
            query_author_names=["Alice Smith"],  # Mixed case
            query_affiliations=[],
            db_path=tmp_db_path
        )
        print(f"   Conflicts for 'alice smith' vs ['Alice Smith']: {conflicts}")
        assert conflicts["is_same_person"] == True, "Should detect same person"
        print(f"   ✓ Same person detected (case-insensitive)")
        
        conflicts = conflicts_for_candidate(
            candidate_name="ALICE SMITH",  # Uppercase
            query_author_names=["alice smith"],  # Lowercase
            query_affiliations=[],
            db_path=tmp_db_path
        )
        assert conflicts["is_same_person"] == True, "Should detect same person"
        print(f"   ✓ Same person detected (case variations)")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Dave Wilson",
            query_author_names=["Alice Smith"],
            query_affiliations=[],
            db_path=tmp_db_path
        )
        assert conflicts["is_same_person"] == False, "Should NOT detect same person"
        print(f"   ✓ Different people correctly identified")
        
        # -----------------------------------------------------------------------
        # Test 3: Conflict detection - Coauthor
        # -----------------------------------------------------------------------
        print("\n4. Testing conflicts_for_candidate() - Coauthor...")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Bob Jones",
            query_author_names=["Alice Smith"],
            query_affiliations=[],
            db_path=tmp_db_path
        )
        print(f"   Conflicts for Bob vs Alice: {conflicts}")
        assert conflicts["is_coauthor"] == True, "Bob and Alice are coauthors"
        print(f"   ✓ Coauthor relationship detected")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Charlie Brown",
            query_author_names=["Alice Smith"],
            query_affiliations=[],
            db_path=tmp_db_path
        )
        assert conflicts["is_coauthor"] == True, "Charlie and Alice are coauthors"
        print(f"   ✓ Coauthor relationship detected")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Dave Wilson",
            query_author_names=["Alice Smith"],
            query_affiliations=[],
            db_path=tmp_db_path
        )
        assert conflicts["is_coauthor"] == False, "Dave and Alice are NOT coauthors"
        print(f"   ✓ Non-coauthor correctly identified")
        
        # -----------------------------------------------------------------------
        # Test 4: Conflict detection - Same affiliation
        # -----------------------------------------------------------------------
        print("\n5. Testing conflicts_for_candidate() - Same affiliation...")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Dave Wilson",
            query_author_names=["Alice Smith"],
            query_affiliations=["MIT"],
            db_path=tmp_db_path,
            candidate_affiliation="MIT"
        )
        print(f"   Conflicts for Dave (MIT) vs Alice (MIT): {conflicts}")
        assert conflicts["same_affiliation"] == True, "Both at MIT"
        print(f"   ✓ Same affiliation detected")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Dave Wilson",
            query_author_names=["Bob Jones"],
            query_affiliations=["Stanford"],
            db_path=tmp_db_path,
            candidate_affiliation="mit"  # Lowercase
        )
        print(f"   Conflicts for Dave (MIT) vs Bob (Stanford): {conflicts}")
        assert conflicts["same_affiliation"] == False, "Different affiliations"
        print(f"   ✓ Different affiliations correctly identified")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Dave Wilson",
            query_author_names=["Alice Smith"],
            query_affiliations=["MIT  "],  # Extra whitespace
            db_path=tmp_db_path,
            candidate_affiliation="  mit"  # Lowercase with whitespace
        )
        assert conflicts["same_affiliation"] == True, "Should normalize whitespace"
        print(f"   ✓ Affiliation matching with whitespace normalization")
        
        # -----------------------------------------------------------------------
        # Test 5: Multiple conflicts
        # -----------------------------------------------------------------------
        print("\n6. Testing multiple conflicts...")
        
        conflicts = conflicts_for_candidate(
            candidate_name="Charlie Brown",
            query_author_names=["Alice Smith", "Charlie Brown"],
            query_affiliations=["MIT"],
            db_path=tmp_db_path,
            candidate_affiliation="MIT"
        )
        print(f"   Conflicts for Charlie vs [Alice, Charlie] at MIT: {conflicts}")
        assert conflicts["is_same_person"] == True, "Charlie is in query list"
        assert conflicts["is_coauthor"] == True, "Charlie coauthored with Alice"
        assert conflicts["same_affiliation"] == True, "Both at MIT"
        print(f"   ✓ All three conflicts detected correctly")
        
        # -----------------------------------------------------------------------
        # Test 6: has_conflict() convenience function
        # -----------------------------------------------------------------------
        print("\n7. Testing has_conflict() convenience function...")
        
        assert has_conflict("Alice Smith", ["alice smith"], [], tmp_db_path) == True
        print(f"   ✓ has_conflict() detects same person")
        
        assert has_conflict("Bob Jones", ["Alice Smith"], [], tmp_db_path) == True
        print(f"   ✓ has_conflict() detects coauthor")
        
        assert has_conflict("Dave", ["Alice"], ["MIT"], tmp_db_path, "MIT") == True
        print(f"   ✓ has_conflict() detects same affiliation")
        
        assert has_conflict("Dave", ["Bob"], ["Stanford"], tmp_db_path, "MIT") == False
        print(f"   ✓ has_conflict() returns False for no conflicts")
        
        # -----------------------------------------------------------------------
        # Test 7: get_coauthors_for_author()
        # -----------------------------------------------------------------------
        print("\n8. Testing get_coauthors_for_author()...")
        
        coauthors = get_coauthors_for_author(tmp_db_path, "Alice Smith")
        print(f"   Alice's coauthors: {coauthors}")
        assert len(coauthors) == 2, "Alice has 2 coauthors"
        names = [c["name"] for c in coauthors]
        assert "Bob Jones" in names and "Charlie Brown" in names
        print(f"   ✓ Coauthor list retrieved correctly")
        
        # -----------------------------------------------------------------------
        # Test 8: Edge cases
        # -----------------------------------------------------------------------
        print("\n9. Testing edge cases...")
        
        # Empty candidate name
        conflicts = conflicts_for_candidate("", ["Alice"], [])
        assert all(not v for v in conflicts.values()), "Empty name should have no conflicts"
        print(f"   ✓ Empty candidate name handled")
        
        # Empty query list
        conflicts = conflicts_for_candidate("Alice", [], [])
        assert all(not v for v in conflicts.values()), "Empty query should have no conflicts"
        print(f"   ✓ Empty query list handled")
        
        # Missing database
        conflicts = conflicts_for_candidate("Alice", ["Bob"], [], Path("nonexistent.db"))
        assert conflicts["is_coauthor"] == False, "Missing DB should not crash"
        print(f"   ✓ Missing database handled gracefully")
        
        # Name with special characters
        conflicts = conflicts_for_candidate(
            "Jean-Luc Picard",
            ["jean-luc picard"],
            []
        )
        assert conflicts["is_same_person"] == True, "Should handle hyphens"
        print(f"   ✓ Names with hyphens handled correctly")
        
        # -----------------------------------------------------------------------
        # Test 9: Build full coauthor graph
        # -----------------------------------------------------------------------
        print("\n10. Testing build_coauthor_graph()...")
        
        graph = build_coauthor_graph(tmp_db_path)
        print(f"   Full coauthor graph: {len(graph)} authors")
        assert len(graph) == 5, "Should have all 5 authors"
        assert len(graph[alice_id]) == 2, "Alice has 2 coauthors"
        assert len(graph[bob_id]) == 2, "Bob has 2 coauthors"
        assert len(graph[dave_id]) == 0, "Dave has no coauthors"
        print(f"   ✓ Full graph built correctly")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nFeatures verified:")
        print("  ✓ build_edges_for_author() builds coauthor edges")
        print("  ✓ conflicts_for_candidate() detects same person (case-insensitive)")
        print("  ✓ conflicts_for_candidate() detects coauthors via database")
        print("  ✓ conflicts_for_candidate() detects same affiliation")
        print("  ✓ Safe handling of name collisions")
        print("  ✓ Whitespace and case normalization")
        print("  ✓ Edge cases handled gracefully")
        print("  ✓ has_conflict() convenience function")
        print("  ✓ get_coauthors_for_author() retrieves coauthor list")
        print("  ✓ build_coauthor_graph() builds complete graph")
        print("\nAll acceptance criteria met!")
        
    finally:
        # Clean up temporary database
        if tmp_db_path.exists():
            os.unlink(tmp_db_path)
            print(f"\nCleaned up temporary database: {tmp_db_path}")

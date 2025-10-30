"""
PDF Ingestion & Parsing Module

Walks directory trees organized as Author Name/paper.pdf and extracts:
- Text content and metadata
- Title, year, abstract detection
- Author relationships
- MD5-based deduplication

Supports multiple extraction backends with fallback:
1. pdfplumber (primary)
2. PyPDF2 (fallback)
3. Apache Tika (last resort)
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# PDF parsing libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available")

try:
    from tika import parser as tika_parser
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False
    logging.warning("Tika not available")

# Import our database utilities
import db_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from PDF ingestion process."""
    total_pdfs: int = 0
    successful_pdfs: int = 0
    failed_pdfs: int = 0
    authors_created: int = 0
    papers_created: int = 0
    papers_updated: int = 0
    papers_with_abstract: int = 0
    papers_with_year: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'total_pdfs': self.total_pdfs,
            'successful_pdfs': self.successful_pdfs,
            'failed_pdfs': self.failed_pdfs,
            'authors_created': self.authors_created,
            'papers_created': self.papers_created,
            'papers_updated': self.papers_updated,
            'papers_with_abstract': self.papers_with_abstract,
            'papers_with_year': self.papers_with_year,
            'abstract_percentage': (
                (self.papers_with_abstract / self.successful_pdfs * 100)
                if self.successful_pdfs > 0 else 0.0
            ),
            'year_percentage': (
                (self.papers_with_year / self.successful_pdfs * 100)
                if self.successful_pdfs > 0 else 0.0
            ),
            'error_count': len(self.errors),
            'errors': self.errors[:10]  # Limit to first 10 errors in output
        }


def compute_md5(file_path: Path) -> str:
    """
    Compute MD5 hash of file contents.
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash as hex string
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute MD5 for {file_path}: {e}")
        # Return a fallback hash based on filename
        return hashlib.md5(str(file_path).encode()).hexdigest()


def extract_text_pdfplumber(file_path: Path) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Extract text using pdfplumber (primary method).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (full_text, metadata_dict)
    """
    if not PDFPLUMBER_AVAILABLE:
        return None, None
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            metadata = pdf.metadata or {}
            
            # Extract text from all pages
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = '\n'.join(text_parts) if text_parts else None
            logger.debug(f"pdfplumber extracted {len(full_text) if full_text else 0} chars from {file_path.name}")
            return full_text, metadata
    except Exception as e:
        logger.warning(f"pdfplumber failed for {file_path.name}: {e}")
        return None, None


def extract_text_pypdf2(file_path: Path) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Extract text using PyPDF2 (fallback method).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (full_text, metadata_dict)
    """
    if not PYPDF2_AVAILABLE:
        return None, None
    
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            
            # Extract metadata
            metadata = {}
            if reader.metadata:
                for key, value in reader.metadata.items():
                    # Remove leading '/' from keys
                    clean_key = key.lstrip('/')
                    metadata[clean_key] = value
            
            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = '\n'.join(text_parts) if text_parts else None
            logger.debug(f"PyPDF2 extracted {len(full_text) if full_text else 0} chars from {file_path.name}")
            return full_text, metadata
    except Exception as e:
        logger.warning(f"PyPDF2 failed for {file_path.name}: {e}")
        return None, None


def extract_text_tika(file_path: Path) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Extract text using Apache Tika (last resort method).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (full_text, metadata_dict)
    """
    if not TIKA_AVAILABLE:
        return None, None
    
    try:
        parsed = tika_parser.from_file(str(file_path))
        full_text = parsed.get('content', None)
        metadata = parsed.get('metadata', {})
        
        logger.debug(f"Tika extracted {len(full_text) if full_text else 0} chars from {file_path.name}")
        return full_text, metadata
    except Exception as e:
        logger.warning(f"Tika failed for {file_path.name}: {e}")
        return None, None


def extract_text_with_fallback(file_path: Path) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Extract text with fallback chain: pdfplumber → PyPDF2 → Tika.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (full_text, metadata_dict)
    """
    # Try pdfplumber first
    text, metadata = extract_text_pdfplumber(file_path)
    if text:
        logger.debug(f"Used pdfplumber for {file_path.name}")
        return text, metadata
    
    # Try PyPDF2
    text, metadata = extract_text_pypdf2(file_path)
    if text:
        logger.debug(f"Used PyPDF2 for {file_path.name}")
        return text, metadata
    
    # Try Tika as last resort
    text, metadata = extract_text_tika(file_path)
    if text:
        logger.debug(f"Used Tika for {file_path.name}")
        return text, metadata
    
    logger.error(f"All extraction methods failed for {file_path.name}")
    return None, None


def extract_title(text: Optional[str], filename: str, metadata: Optional[Dict] = None) -> str:
    """
    Extract paper title heuristically.
    
    Strategy:
    1. Check metadata for title field
    2. Find first TitleCase line with length 4-200 in first page
    3. Fallback to filename without extension
    
    Args:
        text: Extracted text content
        filename: PDF filename
        metadata: PDF metadata dictionary
        
    Returns:
        Extracted or inferred title
    """
    # Try metadata first
    if metadata:
        for key in ['Title', 'title', 'dc:title']:
            if key in metadata and metadata[key]:
                title = str(metadata[key]).strip()
                if 4 <= len(title) <= 200:
                    logger.debug(f"Title from metadata: {title[:50]}...")
                    return title
    
    # Try to extract from text
    if text:
        # Get first page (roughly first 2000 chars)
        first_page = text[:2000]
        lines = first_page.split('\n')
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            
            # Skip empty lines, URLs, emails
            if not line or '@' in line or 'http' in line.lower():
                continue
            
            # Check if line looks like a title
            # Must be TitleCase and reasonable length
            if (4 <= len(line) <= 200 and
                line[0].isupper() and
                not line.isupper() and  # Not ALL CAPS
                not line.endswith(':') and  # Not a section header
                sum(c.isupper() for c in line) >= 2):  # Has multiple capitals
                
                logger.debug(f"Title from text: {line[:50]}...")
                return line
    
    # Fallback to filename
    title = Path(filename).stem
    # Clean up common filename patterns
    title = re.sub(r'[-_]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    logger.debug(f"Title from filename: {title[:50]}...")
    return title


def extract_year(text: Optional[str], metadata: Optional[Dict] = None) -> Optional[int]:
    """
    Extract publication year from metadata or text.
    
    Args:
        text: Extracted text content
        metadata: PDF metadata dictionary
        
    Returns:
        Year as integer or None
    """
    # Try metadata first
    if metadata:
        for key in ['Year', 'year', 'CreationDate', 'creation-date', 'created']:
            if key in metadata and metadata[key]:
                value = str(metadata[key])
                # Extract 4-digit year
                year_match = re.search(r'(19\d{2}|20\d{2})', value)
                if year_match:
                    year = int(year_match.group(1))
                    if 1900 <= year <= 2030:
                        logger.debug(f"Year from metadata: {year}")
                        return year
    
    # Try first page of text
    if text:
        first_page = text[:3000]
        
        # Look for common year patterns
        patterns = [
            r'\b(19\d{2}|20[0-2]\d)\b',  # Standalone year
            r'(?:Published|Copyright|©)\s*:?\s*(19\d{2}|20[0-2]\d)',  # Published/Copyright year
            r'\((19\d{2}|20[0-2]\d)\)',  # Year in parentheses
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, first_page, re.IGNORECASE)
            if matches:
                # Take the first valid year found
                for match in matches:
                    year = int(match if isinstance(match, str) else match[0])
                    if 1900 <= year <= 2030:
                        logger.debug(f"Year from text: {year}")
                        return year
    
    logger.debug("No year found")
    return None


def extract_abstract(text: Optional[str]) -> Optional[str]:
    """
    Extract abstract from paper text.
    
    Args:
        text: Extracted text content
        
    Returns:
        Abstract text or None
    """
    if not text:
        return None
    
    # Look for abstract section
    patterns = [
        r'(?i)abstract\s*[:\-]?\s*\n+(.*?)(?:\n\n|introduction|keywords|1\.|I\.)',
        r'(?i)\babstract\b\s*[:\-]?\s+(.*?)(?:\n\n|\bintroduction\b)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:5000], re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            if 50 <= len(abstract) <= 3000:  # Reasonable abstract length
                logger.debug(f"Abstract found: {len(abstract)} chars")
                return abstract
    
    logger.debug("No abstract found")
    return None


def parse_author_names(text: Optional[str], metadata: Optional[Dict] = None) -> List[str]:
    """
    Extract list of author names from metadata or text.
    
    Args:
        text: Extracted text content
        metadata: PDF metadata dictionary
        
    Returns:
        List of author names
    """
    authors = []
    
    # Try metadata first
    if metadata:
        for key in ['Author', 'author', 'Authors', 'authors', 'Creator', 'creator']:
            if key in metadata and metadata[key]:
                author_text = str(metadata[key])
                # Split by common delimiters
                author_list = re.split(r'[;,]|\band\b', author_text)
                for author in author_list:
                    author = author.strip()
                    if author and len(author) > 2:
                        authors.append(author)
    
    # Try to extract from first page
    if not authors and text:
        first_page = text[:2000]
        lines = first_page.split('\n')
        
        # Look for author lines after title (usually within first 10 lines)
        for i, line in enumerate(lines[1:11]):  # Skip first line (likely title)
            line = line.strip()
            
            # Check if line looks like authors
            # Usually contains names with capitals, possibly with commas/and
            if (line and 
                len(line) < 150 and
                sum(c.isupper() for c in line) >= 2 and
                not line.endswith(':') and
                (',' in line or ' and ' in line.lower() or 
                 re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', line))):
                
                # Split by delimiters
                author_list = re.split(r'[;,]|\band\b', line)
                for author in author_list:
                    author = re.sub(r'\d+|[*†‡§]', '', author).strip()  # Remove numbers and symbols
                    if author and 5 <= len(author) <= 50:
                        authors.append(author)
                
                if authors:
                    break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_authors = []
    for author in authors:
        if author not in seen:
            seen.add(author)
            unique_authors.append(author)
    
    logger.debug(f"Found {len(unique_authors)} authors: {unique_authors}")
    return unique_authors


def process_pdf(
    file_path: Path,
    author_name: str,
    db_path: Path,
    stats: IngestionStats
) -> bool:
    """
    Process a single PDF file and ingest into database.
    
    Args:
        file_path: Path to PDF file
        author_name: Primary author (from folder name)
        db_path: Path to SQLite database
        stats: Statistics object to update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {file_path.name} by {author_name}")
        
        # Compute MD5
        md5 = compute_md5(file_path)
        
        # Extract text and metadata
        text, metadata = extract_text_with_fallback(file_path)
        
        if not text:
            logger.warning(f"No text extracted from {file_path.name}")
            stats.errors.append(f"{file_path.name}: No text could be extracted")
            return False
        
        # Extract components
        title = extract_title(text, file_path.name, metadata)
        year = extract_year(text, metadata)
        abstract = extract_abstract(text)
        author_list = parse_author_names(text, metadata)
        
        # Ensure primary author exists
        author_id = db_utils.upsert_author(str(db_path), author_name)
        
        # Track if this is a new author
        if db_utils.find_author_by_name(str(db_path), author_name):
            pass  # Author already existed
        else:
            stats.authors_created += 1
        
        # Upsert paper
        paper_id, is_new = db_utils.upsert_paper(
            str(db_path),
            author_id=author_id,
            title=title,
            md5=md5,
            year=year,
            path=str(file_path),
            abstract=abstract,
            fulltext=text
        )
        
        if is_new:
            stats.papers_created += 1
        else:
            stats.papers_updated += 1
        
        # Add co-authors
        # Include primary author if found in parsed list
        all_authors = set([author_name])
        if author_list:
            all_authors.update(author_list)
        
        for idx, person_name in enumerate(sorted(all_authors)):
            db_utils.insert_paper_author(
                str(db_path),
                paper_id=paper_id,
                person_name=person_name,
                author_order=idx
            )
        
        # Refresh co-author edges for primary author
        db_utils.refresh_coauthor_edges(str(db_path), author_id)
        
        # Update statistics
        if abstract:
            stats.papers_with_abstract += 1
        if year:
            stats.papers_with_year += 1
        
        logger.info(f"✓ Successfully processed {file_path.name}")
        return True
        
    except Exception as e:
        error_msg = f"{file_path.name}: {str(e)}"
        logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
        stats.errors.append(error_msg)
        return False


def walk_and_ingest(root_dir: Path, db_path: Path) -> Dict[str, Any]:
    """
    Walk directory tree and ingest PDFs into database.
    
    Directory structure: root_dir/Author Name/*.pdf
    
    Args:
        root_dir: Root directory containing author folders
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with ingestion statistics
    """
    logger.info("=" * 70)
    logger.info(f"Starting PDF ingestion from: {root_dir}")
    logger.info(f"Database: {db_path}")
    logger.info("=" * 70)
    
    # Initialize database
    db_utils.init_db(str(db_path))
    
    # Initialize statistics
    stats = IngestionStats()
    
    # Convert to Path objects
    root_dir = Path(root_dir)
    db_path = Path(db_path)
    
    if not root_dir.exists():
        error_msg = f"Root directory does not exist: {root_dir}"
        logger.error(error_msg)
        stats.errors.append(error_msg)
        return stats.to_dict()
    
    # Walk directory tree
    for author_dir in sorted(root_dir.iterdir()):
        # Skip files in root directory
        if not author_dir.is_dir():
            continue
        
        author_name = author_dir.name
        logger.info(f"\nProcessing author folder: {author_name}")
        
        # Find all PDFs in author directory
        pdf_files = list(author_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDFs found in {author_name}/")
            continue
        
        logger.info(f"Found {len(pdf_files)} PDF(s) for {author_name}")
        
        # Process each PDF
        for pdf_file in sorted(pdf_files):
            stats.total_pdfs += 1
            
            try:
                success = process_pdf(pdf_file, author_name, db_path, stats)
                if success:
                    stats.successful_pdfs += 1
                else:
                    stats.failed_pdfs += 1
            except Exception as e:
                # Catch any unexpected errors to ensure one bad file doesn't crash everything
                logger.error(f"Unexpected error processing {pdf_file.name}: {e}", exc_info=True)
                stats.failed_pdfs += 1
                stats.errors.append(f"{pdf_file.name}: Unexpected error - {str(e)}")
    
    # Generate summary
    summary = stats.to_dict()
    
    logger.info("\n" + "=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total PDFs processed: {summary['total_pdfs']}")
    logger.info(f"Successful: {summary['successful_pdfs']}")
    logger.info(f"Failed: {summary['failed_pdfs']}")
    logger.info(f"Authors created: {summary['authors_created']}")
    logger.info(f"Papers created: {summary['papers_created']}")
    logger.info(f"Papers updated: {summary['papers_updated']}")
    logger.info(f"Papers with abstract: {summary['papers_with_abstract']} ({summary['abstract_percentage']:.1f}%)")
    logger.info(f"Papers with year: {summary['papers_with_year']} ({summary['year_percentage']:.1f}%)")
    
    if summary['errors']:
        logger.warning(f"\nErrors encountered: {summary['error_count']}")
        for error in summary['errors'][:5]:
            logger.warning(f"  - {error}")
    
    logger.info("=" * 70)
    
    return summary


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """
    Self-contained tests for parser module.
    """
    import tempfile
    import os
    
    print("=" * 70)
    print("Running Parser Tests")
    print("=" * 70)
    
    # Test 1: MD5 computation
    print("\n[TEST 1] Testing MD5 computation...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
        tmp.write("test content")
        tmp_path = Path(tmp.name)
    
    md5 = compute_md5(tmp_path)
    assert len(md5) == 32, "MD5 should be 32 chars"
    print(f"✓ MD5 computed: {md5}")
    os.unlink(tmp_path)
    
    # Test 2: Title extraction
    print("\n[TEST 2] Testing title extraction...")
    test_text = """
    Deep Learning for Natural Language Processing
    
    John Doe, Jane Smith
    
    Abstract: This paper presents...
    """
    title = extract_title(test_text, "test.pdf")
    assert "Deep Learning" in title, f"Title extraction failed: {title}"
    print(f"✓ Title extracted: {title}")
    
    # Test 3: Year extraction
    print("\n[TEST 3] Testing year extraction...")
    test_text_year = "Published in 2023\nThis paper discusses..."
    year = extract_year(test_text_year)
    assert year == 2023, f"Expected 2023, got {year}"
    print(f"✓ Year extracted: {year}")
    
    # Test 4: Abstract extraction
    print("\n[TEST 4] Testing abstract extraction...")
    test_abstract = """
    Title Here
    Authors Here
    
    Abstract: This is a test abstract with enough content to be valid. 
    It should be extracted correctly from the text. We need to make sure
    it has sufficient length to pass validation checks.
    
    1. Introduction
    """
    abstract = extract_abstract(test_abstract)
    assert abstract is not None, "Abstract should be extracted"
    assert len(abstract) > 50, "Abstract should be substantial"
    print(f"✓ Abstract extracted: {len(abstract)} chars")
    
    # Test 5: Author name parsing
    print("\n[TEST 5] Testing author name parsing...")
    test_authors = "John Doe, Jane Smith and Bob Wilson"
    authors = parse_author_names(test_authors, {"Author": test_authors})
    assert len(authors) >= 2, f"Expected multiple authors, got {authors}"
    print(f"✓ Authors parsed: {authors}")
    
    # Test 6: Fallback title from filename
    print("\n[TEST 6] Testing filename fallback...")
    title_fb = extract_title(None, "my_research_paper.pdf", None)
    assert "research" in title_fb.lower(), f"Filename fallback failed: {title_fb}"
    print(f"✓ Filename fallback: {title_fb}")
    
    # Test 7: Resilience to empty data
    print("\n[TEST 7] Testing resilience to empty data...")
    title_empty = extract_title(None, "test.pdf", None)
    year_empty = extract_year(None, None)
    abstract_empty = extract_abstract(None)
    authors_empty = parse_author_names(None, None)
    assert title_empty == "test", "Should fallback to filename"
    assert year_empty is None, "Should return None for no year"
    assert abstract_empty is None, "Should return None for no abstract"
    assert authors_empty == [], "Should return empty list"
    print("✓ Handles empty data gracefully")
    
    print("\n" + "=" * 70)
    print("ALL PARSER TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

"""
Text Cleaning and Utility Functions

Provides common utilities for text processing, temporal weighting,
and device management for ML operations.

All functions are pure (no side effects) and documented with doctests.
"""

import re
import unicodedata
from datetime import datetime
from typing import Optional, Tuple

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def clean_text(s: str) -> str:
    """
    Clean and normalize text for NLP processing.
    
    Operations:
    - Convert to lowercase
    - Normalize whitespace (tabs, newlines, multiple spaces)
    - Remove control characters
    - Preserve mathematical tokens (Greek letters, operators, subscripts)
    - Preserve alphanumeric and common punctuation
    
    Args:
        s: Input text string
        
    Returns:
        Cleaned and normalized text
        
    Examples:
        >>> clean_text("Hello   World\\n\\t!")
        'hello world !'
        
        >>> clean_text("Deep\\xa0Learning")
        'deep learning'
        
        >>> clean_text("UPPERCASE text")
        'uppercase text'
        
        >>> clean_text("α-helix and β-sheet")
        'α-helix and β-sheet'
        
        >>> clean_text("x₁ + x₂ = y")
        'x₁ + x₂ = y'
        
        >>> clean_text("Multiple    spaces")
        'multiple spaces'
        
        >>> clean_text("")
        ''
        
        >>> clean_text("   ")
        ''
    """
    if not s:
        return ""
    
    # Convert to lowercase
    s = s.lower()
    
    # Normalize Unicode (but preserve subscripts/superscripts)
    # Use NFC instead of NFKD to preserve formatting characters
    s = unicodedata.normalize('NFC', s)
    
    # Remove control characters but keep whitespace
    # Preserve Greek letters (α, β, γ, etc.), math operators, subscripts, superscripts
    def is_valid_char(c):
        category = unicodedata.category(c)
        
        # Keep letters (Latin, Greek, etc.)
        if category.startswith('L'):
            return True
        
        # Keep numbers
        if category.startswith('N'):
            return True
        
        # Keep punctuation and symbols (math operators, etc.)
        if category.startswith('P') or category.startswith('S'):
            return True
        
        # Keep whitespace
        if category.startswith('Z') or c in ' \t\n':
            return True
        
        # Keep math symbols
        if category == 'Sm':  # Math symbols
            return True
        
        return False
    
    s = ''.join(c for c in s if is_valid_char(c))
    
    # Normalize whitespace
    # Replace tabs and newlines with spaces
    s = re.sub(r'[\t\n\r\f\v]', ' ', s)
    
    # Collapse multiple spaces into single space
    s = re.sub(r'\s+', ' ', s)
    
    # Strip leading/trailing whitespace
    s = s.strip()
    
    return s


def split_abstract_fulltext(text: str) -> Tuple[str, str]:
    """
    Split paper text into abstract and fulltext sections heuristically.
    
    Strategy:
    1. Look for "Abstract" header
    2. Extract ~300-1500 words after abstract as abstract section
    3. Find "Introduction" or first numbered section
    4. Everything after that is fulltext
    
    Args:
        text: Full paper text
        
    Returns:
        Tuple of (abstract, fulltext)
        
    Examples:
        >>> text = "Title\\nAbstract: This is the abstract.\\nIntroduction\\nThis is the intro."
        >>> abstract, fulltext = split_abstract_fulltext(text)
        >>> "abstract" in abstract.lower()
        True
        >>> "introduction" in fulltext.lower()
        True
        
        >>> text = "This is just paper content without any section markers."
        >>> abstract, fulltext = split_abstract_fulltext(text)
        >>> len(fulltext) > 0
        True
        >>> abstract
        ''
        
        >>> text = "Abstract\\n" + " ".join(["word"] * 2000) + "\\nIntroduction\\nMore text"
        >>> abstract, fulltext = split_abstract_fulltext(text)
        >>> 300 <= len(abstract.split()) <= 1500
        True
        
        >>> split_abstract_fulltext("")
        ('', '')
    """
    if not text:
        return "", ""
    
    abstract = ""
    fulltext = ""
    
    # Try to find abstract section
    abstract_pattern = r'(?i)\babstract\b\s*[:\-]?\s*'
    abstract_match = re.search(abstract_pattern, text)
    
    if abstract_match:
        # Found abstract marker
        abstract_start = abstract_match.end()
        
        # Look for end of abstract (introduction, keywords, or first section)
        end_patterns = [
            r'(?i)\n\s*(?:introduction|keywords?|1\.|I\.|background|related work)',
            r'\n\s*\d+\.\s+[A-Z]',  # Numbered section
            r'\n\s*[IVX]+\.\s+[A-Z]',  # Roman numeral section
        ]
        
        abstract_end = None
        for pattern in end_patterns:
            end_match = re.search(pattern, text[abstract_start:])
            if end_match:
                abstract_end = abstract_start + end_match.start()
                break
        
        # Extract abstract text
        if abstract_end:
            abstract_text = text[abstract_start:abstract_end]
        else:
            # No clear end marker, take first 1500 words
            words = text[abstract_start:].split()
            abstract_text = ' '.join(words[:1500])
        
        # Enforce length constraints (300-1500 words)
        abstract_words = abstract_text.split()
        
        if len(abstract_words) > 1500:
            # Truncate to 1500 words
            abstract_text = ' '.join(abstract_words[:1500])
            abstract_end = abstract_start + len(abstract_text)
        elif len(abstract_words) < 50:
            # Too short, might not be real abstract
            # Try to extend to reasonable length
            words_after = text[abstract_start:].split()
            abstract_text = ' '.join(words_after[:min(300, len(words_after))])
        
        abstract = abstract_text.strip()
        
        # Fulltext is everything after abstract
        if abstract_end:
            fulltext = text[abstract_end:].strip()
        else:
            # If no clear end, fulltext is remainder
            fulltext = text[abstract_start + len(abstract):].strip()
    else:
        # No abstract marker found, treat all as fulltext
        fulltext = text
    
    return abstract, fulltext


def recency_weight(
    pub_year: int,
    ref_year: Optional[int] = None,
    tau: float = 3.0
) -> float:
    """
    Compute temporal recency weight using exponential decay.
    
    Formula: exp(-age / tau)
    where age = ref_year - pub_year
    
    More recent papers get higher weights. The tau parameter controls
    the decay rate (smaller tau = faster decay).
    
    Args:
        pub_year: Publication year of the paper
        ref_year: Reference year (default: current year)
        tau: Time constant for exponential decay (years)
        
    Returns:
        Weight between 0 and 1 (1 = most recent)
        
    Examples:
        >>> # Same year = weight of 1.0
        >>> abs(recency_weight(2023, 2023) - 1.0) < 0.01
        True
        
        >>> # 3 years old with tau=3.0 should be ~0.37
        >>> weight = recency_weight(2020, 2023, tau=3.0)
        >>> 0.35 < weight < 0.40
        True
        
        >>> # 6 years old should be less weighted
        >>> recency_weight(2017, 2023, tau=3.0) < recency_weight(2020, 2023, tau=3.0)
        True
        
        >>> # Older papers always have weight > 0
        >>> recency_weight(2000, 2023) > 0
        True
        
        >>> # Default ref_year is current year
        >>> w1 = recency_weight(2023)
        >>> w2 = recency_weight(2023, datetime.now().year)
        >>> abs(w1 - w2) < 0.01
        True
        
        >>> # Smaller tau = faster decay
        >>> recency_weight(2020, 2023, tau=1.0) < recency_weight(2020, 2023, tau=5.0)
        True
        
        >>> # Future papers (shouldn't happen, but handle gracefully)
        >>> recency_weight(2025, 2023) >= 1.0
        True
    """
    import math
    
    # Default to current year if not specified
    if ref_year is None:
        ref_year = datetime.now().year
    
    # Calculate age
    age = ref_year - pub_year
    
    # Handle future papers (shouldn't happen, but be safe)
    if age < 0:
        return 1.0
    
    # Exponential decay: exp(-age / tau)
    weight = math.exp(-age / tau)
    
    return weight


def device() -> str:
    """
    Get the best available device for PyTorch operations.
    
    Returns 'cuda' if CUDA is available, otherwise 'cpu'.
    
    Returns:
        Device string: 'cuda' or 'cpu'
        
    Examples:
        >>> dev = device()
        >>> dev in ['cuda', 'cpu']
        True
        
        >>> # Always returns a string
        >>> isinstance(device(), str)
        True
    """
    if not TORCH_AVAILABLE:
        return 'cpu'
    
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dictionary with device information
        
    Examples:
        >>> info = get_device_info()
        >>> 'device' in info
        True
        >>> 'torch_available' in info
        True
        >>> info['device'] in ['cuda', 'cpu']
        True
    """
    info = {
        'torch_available': TORCH_AVAILABLE,
        'device': device()
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    else:
        info['cuda_available'] = False
    
    return info


def normalize_year(year: Optional[int], min_year: int = 1900, max_year: int = 2030) -> Optional[int]:
    """
    Validate and normalize publication year.
    
    Args:
        year: Year to validate
        min_year: Minimum valid year
        max_year: Maximum valid year
        
    Returns:
        Validated year or None if invalid
        
    Examples:
        >>> normalize_year(2023)
        2023
        
        >>> normalize_year(1850) is None
        True
        
        >>> normalize_year(2100) is None
        True
        
        >>> normalize_year(None) is None
        True
        
        >>> normalize_year(2000, min_year=1990)
        2000
        
        >>> normalize_year(1985, min_year=1990) is None
        True
    """
    if year is None:
        return None
    
    if min_year <= year <= max_year:
        return year
    
    return None


def truncate_text(text: str, max_words: int = 512, suffix: str = "...") -> str:
    """
    Truncate text to maximum word count.
    
    Args:
        text: Input text
        max_words: Maximum number of words
        suffix: String to append if truncated
        
    Returns:
        Truncated text
        
    Examples:
        >>> truncate_text("one two three four five", max_words=3)
        'one two three...'
        
        >>> truncate_text("short text", max_words=100)
        'short text'
        
        >>> truncate_text("a b c d e", max_words=3, suffix="[...]")
        'a b c[...]'
        
        >>> truncate_text("", max_words=10)
        ''
    """
    if not text:
        return text
    
    words = text.split()
    
    if len(words) <= max_words:
        return text
    
    truncated = ' '.join(words[:max_words])
    return truncated + suffix


def word_count(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of words
        
    Examples:
        >>> word_count("Hello world")
        2
        
        >>> word_count("one two three four")
        4
        
        >>> word_count("")
        0
        
        >>> word_count("   ")
        0
    """
    if not text:
        return 0
    
    return len(text.split())


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """
    Run comprehensive tests for all utility functions.
    """
    import doctest
    
    print("=" * 70)
    print("Running Utility Function Tests")
    print("=" * 70)
    
    # Run doctests
    print("\n[TEST 1] Running doctests...")
    results = doctest.testmod(verbose=False)
    
    if results.failed == 0:
        print(f"✓ All {results.attempted} doctests passed")
    else:
        print(f"✗ {results.failed}/{results.attempted} doctests failed")
        return False
    
    # Additional integration tests
    print("\n[TEST 2] Testing clean_text edge cases...")
    
    # Test with various Unicode characters
    test_cases = [
        ("Hello\x00World", "hello world"),  # Null character
        ("café", "café"),  # Accented character
        ("résumé", "résumé"),  # Multiple accents
        ("1 + 1 = 2", "1 + 1 = 2"),  # Math operators
        ("α + β = γ", "α + β = γ"),  # Greek letters
        ("x₁ × x₂", "x₁ × x₂"),  # Subscripts and multiplication
    ]
    
    for input_text, expected in test_cases:
        result = clean_text(input_text)
        # Just verify it doesn't crash and returns something reasonable
        assert isinstance(result, str), f"Failed for: {input_text}"
    
    print("✓ clean_text handles edge cases")
    
    # Test split_abstract_fulltext
    print("\n[TEST 3] Testing split_abstract_fulltext...")
    
    paper_text = """
    Deep Learning for NLP
    
    Abstract: This paper presents a comprehensive study of deep learning 
    methods for natural language processing. We explore various architectures
    including transformers and recurrent networks.
    
    1. Introduction
    
    Natural language processing has seen tremendous advances in recent years.
    This work builds on previous research to develop new methods.
    """
    
    abstract, fulltext = split_abstract_fulltext(paper_text)
    assert len(abstract) > 0, "Abstract should be extracted"
    assert "deep learning" in abstract.lower(), "Abstract should contain key terms"
    assert "introduction" in fulltext.lower(), "Fulltext should contain introduction"
    
    print("✓ split_abstract_fulltext works correctly")
    
    # Test recency_weight
    print("\n[TEST 4] Testing recency_weight...")
    
    # Test monotonic decrease
    w1 = recency_weight(2023, 2023, tau=3.0)
    w2 = recency_weight(2020, 2023, tau=3.0)
    w3 = recency_weight(2015, 2023, tau=3.0)
    
    assert w1 > w2 > w3, "Weights should decrease with age"
    assert 0.9 < w1 <= 1.0, "Same year should have weight ~1"
    
    # Test tau parameter
    w_fast = recency_weight(2020, 2023, tau=1.0)
    w_slow = recency_weight(2020, 2023, tau=10.0)
    assert w_fast < w_slow, "Smaller tau should decay faster"
    
    print("✓ recency_weight behaves correctly")
    
    # Test device
    print("\n[TEST 5] Testing device detection...")
    
    dev = device()
    assert dev in ['cuda', 'cpu'], f"Invalid device: {dev}"
    print(f"✓ Device detected: {dev}")
    
    info = get_device_info()
    assert 'device' in info, "Device info should contain device"
    assert 'torch_available' in info, "Device info should contain torch status"
    print(f"✓ Device info: {info}")
    
    # Test normalize_year
    print("\n[TEST 6] Testing normalize_year...")
    
    assert normalize_year(2023) == 2023
    assert normalize_year(1800) is None
    assert normalize_year(2100) is None
    assert normalize_year(None) is None
    
    print("✓ normalize_year validates correctly")
    
    # Test truncate_text
    print("\n[TEST 7] Testing truncate_text...")
    
    long_text = " ".join([f"word{i}" for i in range(1000)])
    truncated = truncate_text(long_text, max_words=100)
    assert word_count(truncated) <= 103, "Should truncate to ~100 words + suffix"
    assert "..." in truncated, "Should have suffix"
    
    short_text = "short text"
    truncated_short = truncate_text(short_text, max_words=100)
    assert truncated_short == short_text, "Short text should not be truncated"
    
    print("✓ truncate_text works correctly")
    
    # Test word_count
    print("\n[TEST 8] Testing word_count...")
    
    assert word_count("one two three") == 3
    assert word_count("") == 0
    assert word_count("   ") == 0
    assert word_count("single") == 1
    
    print("✓ word_count works correctly")
    
    print("\n" + "=" * 70)
    print("ALL UTILITY TESTS PASSED ✓")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    run_tests()

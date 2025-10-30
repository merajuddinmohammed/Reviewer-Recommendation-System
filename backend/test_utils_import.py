"""
Test script to verify utils.py can be imported safely by other modules.
"""

from utils import (
    clean_text,
    split_abstract_fulltext,
    recency_weight,
    device,
    get_device_info,
    normalize_year,
    truncate_text,
    word_count
)

print("=" * 70)
print("Testing Safe Import from utils.py")
print("=" * 70)

# Test 1: All imports successful
print("\n[TEST 1] All functions imported successfully ✓")

# Test 2: Clean text
print("\n[TEST 2] Testing clean_text()...")
result = clean_text("  HELLO   WORLD  \n\t")
assert result == "hello world"
print(f"  Input: '  HELLO   WORLD  \\n\\t'")
print(f"  Output: '{result}'")
print("  ✓ clean_text works")

# Test 3: Split abstract
print("\n[TEST 3] Testing split_abstract_fulltext()...")
text = "Abstract: This is abstract.\nIntroduction\nThis is intro."
abstract, fulltext = split_abstract_fulltext(text)
print(f"  Abstract extracted: {len(abstract)} chars")
print(f"  Fulltext extracted: {len(fulltext)} chars")
print("  ✓ split_abstract_fulltext works")

# Test 4: Recency weight
print("\n[TEST 4] Testing recency_weight()...")
w = recency_weight(2023, 2023)
print(f"  Weight for same year: {w:.4f}")
assert w > 0.99
w_old = recency_weight(2020, 2023)
print(f"  Weight for 3 years old: {w_old:.4f}")
assert w > w_old
print("  ✓ recency_weight works")

# Test 5: Device
print("\n[TEST 5] Testing device()...")
dev = device()
print(f"  Device: {dev}")
assert dev in ['cuda', 'cpu']
print("  ✓ device works")

# Test 6: Device info
print("\n[TEST 6] Testing get_device_info()...")
info = get_device_info()
print(f"  Torch available: {info['torch_available']}")
print(f"  Device: {info['device']}")
print(f"  CUDA available: {info['cuda_available']}")
print("  ✓ get_device_info works")

# Test 7: Normalize year
print("\n[TEST 7] Testing normalize_year()...")
y = normalize_year(2023)
assert y == 2023
y_invalid = normalize_year(1800)
assert y_invalid is None
print("  ✓ normalize_year works")

# Test 8: Truncate text
print("\n[TEST 8] Testing truncate_text()...")
long_text = " ".join([f"word{i}" for i in range(1000)])
truncated = truncate_text(long_text, max_words=50)
words = truncated.split()
print(f"  Truncated to ~{len(words)} words")
assert len(words) <= 51  # 50 + "..."
print("  ✓ truncate_text works")

# Test 9: Word count
print("\n[TEST 9] Testing word_count()...")
count = word_count("one two three four five")
assert count == 5
print(f"  Word count: {count}")
print("  ✓ word_count works")

print("\n" + "=" * 70)
print("ALL IMPORT TESTS PASSED ✓")
print("utils.py is safe to import from other modules")
print("=" * 70)

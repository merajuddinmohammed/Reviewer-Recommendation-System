"""Test pdfplumber import and extraction."""
import sys
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import pdfplumber
    print(f"✓ pdfplumber imported: {pdfplumber.__version__}")
    PDFPLUMBER_AVAILABLE = True
except ImportError as e:
    print(f"❌ pdfplumber import failed: {e}")
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    print(f"✓ PyPDF2 imported")
    PYPDF2_AVAILABLE = True
except ImportError as e:
    print(f"❌ PyPDF2 import failed: {e}")
    PYPDF2_AVAILABLE = False

print(f"\nPDFPLUMBER_AVAILABLE: {PDFPLUMBER_AVAILABLE}")
print(f"PYPDF2_AVAILABLE: {PYPDF2_AVAILABLE}")

if PDFPLUMBER_AVAILABLE:
    from pathlib import Path
    pdf_path = Path("../Dataset/Amit Saxena/A Review of Clustering Techniques.pdf")
    
    print(f"\nTesting extraction on: {pdf_path.name}")
    print(f"File exists: {pdf_path.exists()}")
    
    if pdf_path.exists():
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"Pages: {len(pdf.pages)}")
                text = pdf.pages[0].extract_text()
                print(f"Text length: {len(text) if text else 0}")
                print(f"Text snippet: {text[:200] if text else 'NONE'}")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")

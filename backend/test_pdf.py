"""Quick test to check if PDFs have extractable text."""
import pdfplumber
from pathlib import Path

# Test a few PDFs
test_pdfs = [
    "../Dataset/Amit Saxena/A Review of Clustering Techniques.pdf",
    "../Dataset/Amit Saxena/Cardioprotection from ischemia.pdf",
    "../Dataset/Prakash Chandra Sharma/A Review on Swarm Intelligence.pdf",
]

for pdf_path in test_pdfs:
    path = Path(pdf_path)
    if not path.exists():
        print(f"❌ Not found: {path.name}")
        continue
    
    try:
        with pdfplumber.open(path) as pdf:
            print(f"\n{'='*60}")
            print(f"📄 {path.name}")
            print(f"   Pages: {len(pdf.pages)}")
            
            # Check first page
            if pdf.pages:
                text = pdf.pages[0].extract_text()
                has_text = bool(text and text.strip())
                print(f"   Has text: {has_text}")
                if has_text:
                    print(f"   Text length: {len(text)} chars")
                    print(f"   First 150 chars: {text[:150]}")
                else:
                    print(f"   ⚠️ NO TEXT - Likely image-based/scanned PDF")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'='*60}")
print("Done!")

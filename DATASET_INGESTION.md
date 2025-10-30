# Dataset Ingestion Guide

## 📊 Dataset Overview

**Location:** `C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment\Dataset`

**Structure:**
- 70+ author folders
- Each folder contains PDFs of that author's papers
- Example: `Dataset/Amit Saxena/*.pdf`

**Total Papers:** ~500-1000 PDFs (estimated)

## 🚀 Quick Start

### 1. Basic Ingestion

```powershell
# Navigate to backend
cd "C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment\backend"

# Activate environment
.\.venv\Scripts\Activate.ps1

# Run ingestion (uses Dataset/ by default)
python ingest.py
```

**Expected Time:** 15-30 minutes for full dataset (depends on PDF count and complexity)

### 2. Monitor Progress

The script will show:
- Progress bars for each step (requires `tqdm`)
- Real-time logging to console
- Detailed logging to `ingest.log`

```
Step 1: Initializing database...
✓ Database initialized

Step 2: Ingesting PDFs...
Ingesting PDFs: 100%|██████████| 542/542 [12:34<00:00, 1.39s/file]

Step 3: Building co-author network...
Building coauthor edges: 100%|██████████| 68/68 [00:15<00:00, 4.53author/s]

Step 4: Generating summary report...
✓ Summary report saved to: data/ingest_summary.csv
```

### 3. Check Output

After ingestion, you'll have:

**1. Database (`data/papers.db`):**
```sql
-- Tables created:
-- authors: 70+ authors
-- papers: ~500-1000 papers
-- paper_authors: author-paper relationships
-- coauthors: co-author network edges
```

**2. Summary CSV (`data/ingest_summary.csv`):**
```csv
author_name,affiliation,paper_count,avg_year,has_abstract,has_fulltext,metadata_percentage
Amit Saxena,,16,2021.5,12,16,100.0
Amita Jain,,8,2022.3,6,8,87.5
...
```

**3. Log File (`ingest.log`):**
- Detailed processing log
- Error messages (if any)
- Statistics

## 🔧 Advanced Options

### Force Re-processing

If you need to re-ingest papers (e.g., after fixing parser):

```powershell
python ingest.py --force
```

**Note:** This will re-process all PDFs, even if already in database.

### Skip Co-author Network

To speed up ingestion (skip co-author network building):

```powershell
python ingest.py --skip-coauthors
```

**Use case:** Quick testing or when you don't need COI detection.

### Custom Paths

```powershell
# Custom data directory
python ingest.py --data_dir C:\path\to\papers --db mydb.db

# Custom output CSV
python ingest.py --output results.csv

# Verbose logging
python ingest.py --verbose
```

## 📈 Expected Results

### Database Statistics

After ingesting the full dataset:

| Metric | Expected Value |
|--------|---------------|
| **Authors** | 70+ |
| **Papers** | 500-1000 |
| **Co-author edges** | 200-500 |
| **Papers with abstracts** | 60-80% |
| **Papers with metadata** | 85-95% |

### Processing Performance

| Operation | Time (estimated) |
|-----------|-----------------|
| PDF text extraction | ~1-2 sec/file |
| Metadata extraction | ~0.5 sec/file |
| Database insertion | ~0.1 sec/file |
| Co-author network | ~0.2 sec/author |
| **Total** | **15-30 minutes** |

## 🐛 Troubleshooting

### Issue: "No PDF files found"

**Cause:** Wrong data directory path

**Solution:**
```powershell
# Check if Dataset exists
ls ..\Dataset

# Use absolute path
python ingest.py --data_dir "C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment\Dataset"
```

### Issue: "Failed to extract text"

**Cause:** Corrupted or image-only PDF

**Solution:**
- Script automatically skips failed PDFs
- Check `ingest.log` for failed files
- Review `ingest_summary.csv` for authors with low metadata %

### Issue: Slow processing

**Solutions:**
1. **Skip co-authors:** `python ingest.py --skip-coauthors` (30% faster)
2. **Install tqdm:** `pip install tqdm` (shows progress bars)
3. **Check disk I/O:** Use SSD if available

### Issue: Database locked

**Cause:** Another process accessing database

**Solution:**
```powershell
# Close any open connections
# Then rerun - script uses WAL mode for concurrency
python ingest.py
```

## 🔍 Verifying Ingestion

### 1. Check Database

```powershell
# Count papers
python -c "from db_utils import get_connection; import sqlite3; conn = sqlite3.connect('data/papers.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM papers'); print(f'Papers: {cursor.fetchone()[0]}'); cursor.execute('SELECT COUNT(*) FROM authors'); print(f'Authors: {cursor.fetchone()[0]}')"
```

### 2. Review Summary CSV

```powershell
# Open in Excel or view in terminal
type data\ingest_summary.csv | more
```

### 3. Check for Errors

```powershell
# View last 20 errors
Select-String -Path ingest.log -Pattern "ERROR" | Select-Object -Last 20
```

### 4. Test Database Queries

```python
from pathlib import Path
from db_utils import get_all_papers, list_authors

# List all authors
authors = list_authors("data/papers.db")
print(f"Total authors: {len(authors)}")
print(f"First 5: {[a['name'] for a in authors[:5]]}")

# Get papers
papers = get_all_papers("data/papers.db")
print(f"Total papers: {len(papers)}")
print(f"With abstracts: {len([p for p in papers if p['abstract']])}")
```

## 📊 Summary Report Fields

The `ingest_summary.csv` contains:

| Column | Description | Example |
|--------|-------------|---------|
| `author_name` | Author's name (from folder) | "Amit Saxena" |
| `affiliation` | Affiliation (if extracted) | "IIT Delhi" or "" |
| `paper_count` | Number of papers | 16 |
| `avg_year` | Average publication year | 2021.5 |
| `has_abstract` | Papers with abstracts | 12 |
| `has_fulltext` | Papers with fulltext | 16 |
| `metadata_percentage` | % with metadata | 87.5 |

**Sorted by:** `paper_count` (descending)

## 🎯 Next Steps

After successful ingestion:

1. **Build TF-IDF index:**
   ```python
   from tfidf_engine import TFIDFEngine
   from db_utils import get_all_papers
   
   papers = get_all_papers("data/papers.db")
   corpus = [f"{p['title']} {p['abstract']}" for p in papers]
   paper_ids = [p['id'] for p in papers]
   
   tfidf = TFIDFEngine()
   tfidf.fit(corpus, paper_ids)
   tfidf.save("models/tfidf_latest.joblib")
   ```

2. **Build FAISS index:**
   ```python
   from embedding import Embeddings, build_faiss_index, save_index
   
   emb = Embeddings("allenai/scibert_scivocab_uncased")
   vectors = emb.encode_texts(corpus, normalize=True)
   faiss_index = build_faiss_index(vectors, dim=768)
   save_index(faiss_index, paper_ids, "models/faiss_scibert")
   ```

3. **Query the system:**
   - Use TF-IDF for keyword search
   - Use FAISS for semantic search
   - Use coauthor_graph for COI detection

## 📝 Notes

- **Idempotent:** Safe to rerun - skips existing papers by default
- **MD5 deduplication:** Same PDF won't be ingested twice
- **Graceful errors:** Failed PDFs logged but don't stop process
- **WAL mode:** Database supports concurrent reads during ingestion
- **Memory efficient:** Processes PDFs one at a time (no batch loading)

## 🔗 Related Documentation

- `README-PARSER.md` - PDF parsing details
- `README.md` - Main project documentation
- `SUMMARY.md` - Technical summary
- Parser module docstrings - In-code documentation

---

**Last Updated:** December 2024  
**Dataset Path:** `C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment\Dataset`  
**Default Database:** `backend/data/papers.db`

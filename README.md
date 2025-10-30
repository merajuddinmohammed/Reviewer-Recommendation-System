# 📄 Academic Reviewer Recommendation System# Applied AI Assignment - PDF Paper Management System



[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://reviewer-recommendation-system-1.onrender.com/)A robust Windows-friendly system for ingesting, parsing, and managing academic papers with GPU acceleration support.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/merajuddinmohammed/Reviewer-Recommendation-System)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)## 🎯 Overview

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org/)This system provides a complete pipeline for:

- **PDF Ingestion**: Automated extraction of text and metadata from academic papers

An intelligent system for recommending academic reviewers for research papers using state-of-the-art NLP and machine learning techniques. Upload a PDF paper and get the top matching reviewers instantly!- **Database Management**: SQLite-based storage with relationship tracking

- **Co-author Networks**: Automatic detection and analysis of collaboration patterns

---- **GPU Acceleration**: CUDA 11.8 support for future ML tasks

- **Resilient Processing**: Handles corrupted PDFs without crashing

## 🌟 Features

## 🚀 Quick Start

- **📤 PDF Upload**: Extract paper metadata (title, abstract, authors) automatically

- **🤖 Smart Matching**: Uses multiple similarity methods:### 1. Clone or Setup Project

  - **Semantic Search** (SciBERT embeddings + FAISS) - 98.02% P@5 accuracy

  - **Lexical Search** (TF-IDF) - 96.05% P@5 accuracy```powershell

  - **Learning-to-Rank** (LightGBM) for optimal rankingcd "C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment"

- **⚡ Fast**: Returns top 10 reviewers in seconds```

- **🎯 Conflict Detection**: Automatically filters co-authors and same institution

- **🔍 Evidence-Based**: Shows matching papers for each recommendation### 2. Run Automated Setup

- **📊 Comprehensive Database**: 519 papers, 589 unique authors

```powershell

---.\setup.ps1

```

## 🚀 Live Demo

This will:

**Frontend:** https://reviewer-recommendation-system-1.onrender.com/- ✅ Create Python virtual environment in `backend/.venv`

- ✅ Install PyTorch with CUDA 11.8

Try uploading a research paper PDF and see the recommendations!- ✅ Install all Python dependencies

- ✅ Install Node.js dependencies

---- ✅ Verify GPU availability

- ✅ Create configuration files

## 📊 Performance Metrics

### 3. Test Installation

Our system achieves **state-of-the-art performance** on the recommendation task:

```powershell

| Method | Precision@5 | nDCG@10 | Notes |# Activate environment

|--------|-------------|---------|-------|cd backend

| **Embeddings (SciBERT)** | **98.02%** | **0.7978** | Best overall |.\.venv\Scripts\Activate.ps1

| TF-IDF | 96.05% | 0.7922 | Fast, lightweight |

| Hybrid | 69.88% | 0.6068 | Needs tuning |# Test database layer

| LambdaRank | 0.00% | 0.0000 | Bug in evaluation |python db_utils.py



> **Context:** State-of-the-art systems typically achieve 70-85% P@5. Our system exceeds this benchmark!# Test parser

python parser.py

---```



## 🏗️ ArchitectureBoth should show: **ALL TESTS PASSED ✓**



```### 4. Ingest the Dataset

┌──────────────────────────────────────────────────────────────────┐

│                      Frontend (React)                             │The dataset is already organized in the `Dataset/` folder with 70+ authors.

│  • PDF Upload Interface                                           │

│  • Results Display with Evidence                                  │```powershell

│  • Deployed on Render (Static Site - Free)                        │# Activate environment (if not already active)

└─────────────────┬────────────────────────────────────────────────┘cd backend

                  │ HTTPS/REST API.\.venv\Scripts\Activate.ps1

┌─────────────────▼────────────────────────────────────────────────┐

│                   Backend (FastAPI)                               │# Run ingestion (uses Dataset/ by default)

│  • PDF Processing (pdfplumber, PyPDF2)                            │python ingest.py

│  • Feature Extraction Pipeline                                    │

│  • Multi-Model Ranking                                            │# This will:

│  • Deployed on Render (Web Service - Free Tier)                   │# - Process all PDFs in Dataset/

└─────────────────┬────────────────────────────────────────────────┘# - Create data/papers.db with metadata

                  │# - Generate data/ingest_summary.csv with statistics

┌─────────────────▼────────────────────────────────────────────────┐# - Build co-author network

│                    ML Models & Data                               │```

│  • SciBERT Embeddings (768-dim)                                   │

│  • FAISS Index (519 papers)                                       │**Note:** First run may take 15-30 minutes depending on dataset size. Safe to rerun - skips duplicates automatically.

│  • TF-IDF Vectorizer (9,350 features)                             │

│  • LightGBM Ranker                                                │## 📁 Project Structure

│  • SQLite Database (papers.db)                                    │

└──────────────────────────────────────────────────────────────────┘```

```Applied AI Assignment/

│

---├── setup.ps1                    # Automated Windows setup script

├── README.md                    # This file

## 🛠️ Technology Stack├── README-SETUP.md              # Detailed setup guide

├── SUMMARY.md                   # Technical summary

### Backend│

- **Framework:** FastAPI (Python 3.11+)├── Dataset/                     # Academic papers dataset

- **NLP:** Sentence-Transformers, Transformers, SciBERT│   ├── Amit Saxena/            # Author folders

- **Vector Search:** FAISS (Facebook AI Similarity Search)│   ├── Amita Jain/

- **ML:** Scikit-learn, LightGBM, TF-IDF│   ├── ... (70+ authors)

- **PDF Processing:** pdfplumber, PyPDF2│   └── (PDFs organized by author)

- **Database:** SQLite│

├── backend/

### Frontend│   ├── .venv/                   # Python virtual environment

- **Framework:** React 18.2 + Vite│   ├── .env                     # Environment configuration

- **HTTP Client:** Axios│   │

- **UI:** Custom CSS with modern design│   ├── db_utils.py             # SQLite database layer

│   ├── parser.py               # PDF ingestion & parsing

### Deployment│   ├── utils.py                # Text cleaning utilities

- **Platform:** Render.com│   ├── tfidf_engine.py         # TF-IDF similarity search

- **Backend:** Web Service (Free Tier - 512MB RAM)│   ├── embedding.py            # Sentence embeddings + FAISS

- **Frontend:** Static Site (Free)│   ├── topic_model.py          # OPTIONAL: BERTopic topic modeling

- **CI/CD:** Auto-deploy from GitHub│   ├── coauthor_graph.py       # Co-author network & COI detection

│   ├── ingest.py               # CLI ingestion script

---│   │

│   ├── demo_parser.py          # Parser demo script

## 🚀 Deployment Details│   ├── demo_full_pipeline.py   # Full pipeline demo

│   ├── test_tfidf_quick.py     # TF-IDF quick tests

### Live URLs│   │

- **Frontend:** https://reviewer-recommendation-system-1.onrender.com/│   ├── README-PARSER.md        # Parser documentation

- **Backend API:** (Deploy backend to get URL)│   ├── README-UTILS.md         # Utils documentation

- **API Docs:** `/docs` endpoint (Swagger UI)│   ├── README-TFIDF.md         # TF-IDF documentation

- **GitHub:** https://github.com/merajuddinmohammed/Reviewer-Recommendation-System│   ├── README-EMBEDDING.md     # Embeddings documentation

│   ├── README-TOPIC.md         # Topic modeling documentation

### Memory Optimization for Free Tier│   │

│   ├── data/                   # Generated data

#### The Challenge 🎯│   │   ├── papers.db           # SQLite database

│   │   └── ingest_summary.csv  # Ingestion statistics

Render's free tier provides **512MB RAM**, but our ML models initially required ~600MB:│   │

│   └── models/

| Component | Memory Usage |│       ├── hf_cache/           # Hugging Face model cache

|-----------|--------------|│       ├── tfidf_latest.joblib # Trained TF-IDF model

| SciBERT model | ~200MB |│       ├── faiss_scibert.index # FAISS semantic index

| FAISS index | ~2MB |│       └── bertopic_model/     # BERTopic models (optional)

| TF-IDF vectorizer | ~450KB |│

| Dependencies | ~300MB |└── frontend/

| **Total** | **~600MB ❌** |    └── (Future: React/Next.js UI)

```

**Problem:** Out of Memory (OOM) errors during deployment!

## 📦 Components

#### The Solution ✅

### 1. Environment Setup (`setup.ps1`)

We implemented **aggressive memory optimizations** to fit within 512MB:

**Features:**

##### 1. **Lazy Loading of Embeddings Model** 🎯- Idempotent setup script (safe to re-run)

- Python 3.10+ validation

Instead of loading the heavy SciBERT model at startup, we load it only when the first PDF is uploaded:- PyTorch CUDA 11.8 installation

- All dependencies with pinned versions

```python- GPU verification with `torch.cuda.is_available()`

# At startup: Don't load embeddings- Environment file creation

models.embeddings = None  # Saves ~200MB at startup

**Usage:**

# On first request: Load dynamically```powershell

if models.embeddings is None:.\setup.ps1                    # Full setup

    logger.info("Loading embeddings model on first request...").\setup.ps1 -SkipNode         # Skip Node.js setup

    from embedding import Embeddings.\setup.ps1 -Force            # Force recreate venv

    models.embeddings = Embeddings()```

```

**Documentation:** See `README-SETUP.md`

**Impact:**

- Startup memory: ~300MB ✅### 2. Database Layer (`backend/db_utils.py`)

- First request adds: ~200MB

- Peak memory: ~480MB (fits in 512MB!)**Features:**

- SQLite with WAL mode for concurrency

##### 2. **Thread and Worker Limiting** 🔧- 5-table schema for papers, authors, and relationships

- MD5-based duplicate prevention

Reduced parallelism to save memory:- Upsert operations for safe updates

- Co-author network builder

```bash- Comprehensive query helpers

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer multiprocessing

export OMP_NUM_THREADS=1             # Single OpenMP thread**Schema:**

export MKL_NUM_THREADS=1             # Single MKL thread```sql

uvicorn app:app --workers 1          # Single worker processauthors          (id, name, affiliation)

```papers           (id, author_id, title, year, abstract, fulltext, md5)

paper_authors    (paper_id, person_name, author_order)

**Impact:** Saves ~100MB by avoiding thread overheadcoauthors        (author_id, coauthor_name, collaboration_count)

paper_vectors    (paper_id, dim, norm, faiss_index)

##### 3. **Optional Model Loading** 📦```



Made some models optional:**Usage:**

- BERTopic: Disabled (not needed for core functionality)```python

- LightGBM: Only 4KB, kept loadedimport db_utils

- FAISS: Required, but only 2MB

# Initialize

##### 4. **Optimized Start Script** ⚙️db_utils.init_db("papers.db")



Created `backend/start.sh` with all optimizations:# Add author

author_id = db_utils.upsert_author("papers.db", "Alice Smith", "MIT")

```bash

#!/usr/bin/env bash# Add paper

export TOKENIZERS_PARALLELISM=falsepaper_id, is_new = db_utils.upsert_paper(

export OMP_NUM_THREADS=1    "papers.db",

export MKL_NUM_THREADS=1    author_id=author_id,

export OPENBLAS_NUM_THREADS=1    title="Deep Learning Fundamentals",

    md5="abc123",

uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1    year=2023

```)



#### Results 📊# Query

papers = db_utils.get_all_papers("papers.db")

**Before Optimization:**```

- Startup memory: ~600MB

- Status: ❌ OOM errors### 3. PDF Parser (`backend/parser.py`)

- Free tier: Not working

**Features:**

**After Optimization:**- Multi-library fallback (pdfplumber → PyPDF2 → Tika)

- Startup memory: ~250-300MB ✅- Smart title extraction (metadata → heuristics → filename)

- Peak memory (after first request): ~450-480MB ✅- Year detection (metadata + regex patterns)

- Status: ✅ Working on free tier!- Abstract extraction (pattern-based)

- Author parsing with delimiter splitting

#### Trade-offs ⚖️- MD5 deduplication

- Resilient error handling (one bad PDF won't crash)

| Aspect | Impact |- Detailed statistics reporting

|--------|--------|

| **First Request** | ⚠️ Takes 10-15 seconds (loading model) |**Usage:**

| **Subsequent Requests** | ✅ Fast (~2-3 seconds) |```python

| **Cold Starts** | ⚠️ Backend sleeps after 15 min inactivity (free tier) |from pathlib import Path

| **Cost** | ✅ **$0/month** (completely free!) |from parser import walk_and_ingest



#### Alternative: Upgrade to Paid Tier# Directory structure: papers/Author Name/*.pdf

results = walk_and_ingest(

If you need better performance:    root_dir=Path("papers"),

    db_path=Path("papers.db")

- **Starter Plan:** $7/month)

- **RAM:** 2GB (plenty of headroom)

- **Always On:** No cold startsprint(f"Processed: {results['successful_pdfs']}/{results['total_pdfs']}")

- **Multiple Workers:** Better concurrencyprint(f"Abstracts: {results['abstract_percentage']:.1f}%")

- **Faster:** No lazy loading delaysprint(f"Years: {results['year_percentage']:.1f}%")

```

---

**Documentation:** See `backend/README-PARSER.md`

## 📦 Dataset

### 4. Text Utilities (`backend/utils.py`)

- **Source:** ArXiv papers (Computer Science, Machine Learning)

- **Papers:** 519 successfully ingested (98.7% success rate from 538 PDFs)**Features:**

- **Authors:** 589 unique researchers- Text cleaning and normalization

- **Features:** Title, Abstract, Authors, Affiliations, Year, Co-author network- Abstract/fulltext splitting (300-1500 word heuristic)

- Recency weighting (exponential decay)

---- Device detection (CUDA/CPU)

- Year validation and normalization

## 🏃 Local Setup- Text truncation utilities

- Pure functions (no side effects)

### Prerequisites- Safe imports (works without PyTorch)

- Python 3.11+

- Node.js 18+**Usage:**

- 4GB RAM recommended (for models)```python

- GPU optional (CPU works fine)from utils import clean_text, split_abstract_fulltext, recency_weight, device



### Backend Setup# Clean text

text = clean_text("  HELLO   WORLD  ")  # → "hello world"

```bash

# Clone repository# Split text

git clone https://github.com/merajuddinmohammed/Reviewer-Recommendation-System.gitpaper = {"fulltext": "..." * 1000}

cd Reviewer-Recommendation-Systemabstract, fulltext = split_abstract_fulltext(paper)



# Create virtual environment# Calculate recency weight

python -m venv .venvweight = recency_weight(2020, 2023, tau=3.0)  # → 0.368

source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Check device

# Install dependenciesdev = device()  # → "cuda" or "cpu"

cd backend```

pip install -r requirements.txt

**Documentation:** See `backend/README-UTILS.md`

# Run server

uvicorn app:app --reload --host 0.0.0.0 --port 8000### 5. TF-IDF Similarity Engine (`backend/tfidf_engine.py`)

```

**Features:**

Backend will be available at: http://localhost:8000- Sparse TF-IDF vectorization with sklearn

- Fast cosine similarity search (<1ms for 1000 docs)

### Frontend Setup- Configurable n-grams and document frequency filters

- Paper ID tracking for result retrieval

```bash- Model persistence with joblib

# Navigate to frontend- Top terms extraction per document

cd frontend- Memory-efficient sparse matrices (CSR format)



# Install dependencies**Usage:**

npm install```python

from tfidf_engine import TFIDFEngine

# Run development server

npm run dev# Initialize and fit

```engine = TFIDFEngine(

    max_features=50000,

Frontend will be available at: http://localhost:5173    ngram_range=(1, 2),

    min_df=2,

---    max_df=0.85

)

## 📖 Usage

engine.fit(corpus_texts, paper_ids)

### Via Web Interface

# Search for similar papers

1. Visit: https://reviewer-recommendation-system-1.onrender.com/results = engine.most_similar("deep learning", topn=10)

2. Upload a research paper (PDF)for paper_id, score in results:

3. Optionally add authors and affiliations for COI detection    print(f"Paper {paper_id}: {score:.4f}")

4. Select number of recommendations (1-50)

5. Click "Get Recommendations"# Save model

6. View ranked reviewers with evidence papersengine.save("models/tfidf_latest.joblib")



### Via API# Load later

engine = TFIDFEngine.load("models/tfidf_latest.joblib")

```bash```

# Health check

curl https://your-backend-url.onrender.com/health**Documentation:** See `backend/README-TFIDF.md`



# Get recommendations (PDF upload)### 6. Sentence Embeddings + FAISS (`backend/embedding.py`)

curl -X POST "https://your-backend-url.onrender.com/recommend" \

  -F "file=@paper.pdf" \**Features:**

  -F "k=10"- SciBERT/SPECTER models for scientific papers

```- GPU acceleration with automatic device detection

- Batched encoding with progress bars

**API Documentation:** Visit `/docs` endpoint for interactive Swagger UI- Mean pooling and float32 output

- L2 normalization for cosine similarity

---- FAISS IndexFlatIP for fast search

- Index persistence with ID mapping

## 🧪 Evaluation- NaN handling for robust encoding



Comprehensive evaluation on 81 test queries:**Usage:**

```python

```bashfrom embedding import Embeddings, build_faiss_index, save_index, search_index

cd backend

python eval_report.py --queries 100 --seed 42# Initialize with SciBERT

```emb = Embeddings(model_name="allenai/scibert_scivocab_uncased")



**Results:**# Encode texts

- Evaluated 81 queriestexts = ["deep learning", "neural networks", "transformers"]

- Average 5.5 relevant reviewers per papervectors = emb.encode_texts(texts, normalize=True)

- 98.02% precision@5 (embeddings method)

- 0.7978 nDCG@10 (excellent ranking quality)# Build FAISS index

index = build_faiss_index(vectors, dim=emb.dim, metric="ip")

See `EVALUATION_SCORES.md` for detailed metrics.

# Save with ID mapping

---save_index(index, "models/faiss_scibert", paper_ids=[101, 102, 103])



## 🗂️ Project Structure# Load and search

from embedding import load_index

```index, id_map = load_index("models/faiss_scibert")

Reviewer-Recommendation-System/query_vec = emb.encode_texts(["attention mechanisms"], normalize=True)

├── backend/scores, paper_ids = search_index(index, query_vec, k=10, id_map=id_map)

│   ├── app.py                    # FastAPI application

│   ├── config.py                 # Configuration (20+ parameters)print(f"Top-10 similar papers: {paper_ids[0]}")

│   ├── embedding.py              # SciBERT embeddingsprint(f"Similarity scores: {scores[0]}")

│   ├── tfidf_engine.py           # TF-IDF search```

│   ├── ranker.py                 # Feature aggregation

│   ├── parser.py                 # PDF parsing**Supported Models:**

│   ├── db_utils.py               # Database operations- `allenai/scibert_scivocab_uncased` - 768 dim, scientific papers (recommended)

│   ├── coauthor_graph.py         # Conflict detection- `allenai/specter` - 768 dim, paper-level embeddings (recommended)

│   ├── ingest.py                 # Dataset ingestion- `sentence-transformers/all-MiniLM-L6-v2` - 384 dim, fast general purpose

│   ├── build_pipeline.py         # Model building

│   ├── train_ranker.py           # LambdaRank training**Performance:**

│   ├── eval_report.py            # Evaluation script- Encoding: ~20 docs/sec (CPU), ~200 docs/sec (GPU)

│   ├── requirements.txt          # Python dependencies- Search: 0.5-20ms for 1K-100K documents

│   ├── start.sh                  # Optimized start script- Memory: 3MB per 1K papers (768 dim embeddings)

│   ├── render_config.py          # Memory optimization config

│   ├── data/**Documentation:** See `backend/README-EMBEDDING.md`

│   │   ├── papers.db             # SQLite database

│   │   ├── faiss_index.faiss     # Vector index### 7. Topic Modeling with BERTopic (`backend/topic_model.py`) - OPTIONAL

│   │   ├── id_map.npy            # Paper ID mapping

│   │   └── train.parquet         # Training data⚠️ **This module is OPTIONAL and the system works perfectly without it.**

│   └── models/

│       ├── tfidf_vectorizer.pkl  # TF-IDF model**Features:**

│       └── lgbm_ranker.pkl       # LightGBM model- BERTopic for topic discovery in paper abstracts

├── frontend/- UMAP dimensionality reduction + HDBSCAN clustering

│   ├── src/- Author expertise profiling by topics

│   │   ├── App.jsx               # Main app component- Topic overlap scoring (cosine/Jaccard)

│   │   ├── components/- Graceful degradation when dependencies unavailable

│   │   │   ├── FileUploader.jsx  # PDF upload- All functions return None if packages not installed

│   │   │   └── ReviewerList.jsx  # Results display

│   │   └── main.jsx**Usage:**

│   ├── package.json```python

│   └── vite.config.jsfrom topic_model import is_available, train_bertopic, author_topic_profile

├── render.yaml                   # Render deployment config

├── RENDER_DEPLOYMENT.md          # Deployment guide# Check if topic modeling available

├── EVALUATION_SCORES.md          # Evaluation resultsif is_available():

└── README.md                     # This file    # Train on abstracts

```    model = train_bertopic(abstracts, min_topic_size=10)

    

---    # Get author's expertise

    topics = author_topic_profile(author_id=42, db_path="papers.db")

## 🔧 Configuration    if topics:

        for topic_id, topic_name, weight in topics:

Key parameters in `backend/config.py`:            print(f"{topic_name}: {weight:.2f}")

else:

```python    print("Topic modeling not available - continuing without it")

# Retrieval```

N1_FAISS = 200        # Top-N for semantic search

N2_TFIDF = 150        # Top-N for lexical search**Installation (Optional):**

```powershell

# Ranking weights# Requires C++ Build Tools on Windows

W_S = 0.55            # Semantic similarity weightpip install bertopic umap-learn hdbscan

W_L = 0.25            # Lexical similarity weight```

W_R = 0.20            # Recency weight

**Note:** If installation fails (common on Windows due to C++ compiler requirement), the system will work without topic modeling. Use TF-IDF and embeddings for search instead.

# Recommendations

TOPK_RETURN = 10      # Default recommendations to return**When to Use:**

MAX_K = 50            # Maximum recommendations allowed- ✅ You have 50+ papers with abstracts

```- ✅ You want to discover research themes

- ✅ You need author expertise profiles

---- ❌ Skip if installation fails (not required)



## 🤝 Team**Documentation:** See `backend/README-TOPIC.md`



- **Tikkisetti Sri Dhruti** - SE22UARI175## 🔧 Requirements

- **Zuhair Hussain B** - SE22UARI096  

- **Meerajuddin Mohhammad D** - SE22UCSE307### System Requirements

- **OS**: Windows 10/11

---- **Python**: 3.10+

- **Node.js**: 20 LTS (optional, for frontend)

## 📄 License- **GPU**: NVIDIA GPU with CUDA 11.8 support (recommended)

- **RAM**: 8GB minimum, 16GB recommended

This project is for academic purposes.- **Disk**: 5GB for dependencies + space for PDFs + models



---### NVIDIA GPU Setup

1. Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx

## 🙏 Acknowledgments2. Verify with: `nvidia-smi`

3. CUDA toolkit 11.8 (optional, PyTorch includes CUDA libraries)

- **SciBERT:** Pretrained model by Allen Institute for AI

- **FAISS:** Vector search library by Facebook Research## 📚 Python Dependencies

- **Sentence-Transformers:** Hugging Face library

- **Render:** Free hosting platform### Core Framework

- `fastapi==0.104.1` - Modern async web framework

---- `uvicorn[standard]==0.24.0` - ASGI server

- `pydantic==2.5.0` - Data validation

## 📚 Additional Documentation

### Data Science

- **`RENDER_DEPLOYMENT.md`** - Detailed deployment guide- `numpy==1.24.3` - Numerical computing

- **`EVALUATION_SCORES.md`** - Complete evaluation results- `pandas==2.0.3` - Data manipulation

- **`backend/CONFIG_QUICK_REFERENCE.md`** - Configuration parameters- `scikit-learn==1.3.2` - Machine learning

- **`backend/EVALUATION_GUIDE.md`** - How to run evaluations- `scipy==1.11.4` - Scientific computing



---### Machine Learning

- `torch`, `torchvision`, `torchaudio` - PyTorch with CUDA 11.8

## 🐛 Known Issues- `lightgbm==4.1.0` - LambdaRank (CPU)

- `sentence-transformers==2.2.2` - Semantic embeddings

1. **LambdaRank evaluation bug:** Function signature mismatch in `eval_report.py`- `transformers==4.35.2` - Hugging Face transformers

2. **Hybrid method underperforms:** Needs weight tuning (currently 55/25/20)- `faiss-cpu==1.7.4` - Fast similarity search

3. **First request slow:** Embeddings model loads on demand (10-15 seconds)

4. **Free tier cold starts:** Backend sleeps after 15 minutes inactivity### PDF Processing

- `pdfplumber==0.10.3` - PDF text extraction (primary)

---- `pypdf2==3.0.1` - PDF manipulation (fallback)

- `tika==2.6.0` - Apache Tika wrapper (last resort)

## 🔮 Future Improvements

### Topic Modeling

- [ ] Fix LambdaRank evaluation- `bertopic==0.15.0` - Topic modeling

- [ ] Tune hybrid weights with grid search- `umap-learn==0.5.5` - Dimensionality reduction

- [ ] Add citation-based features- `hdbscan==0.8.33` - Clustering

- [ ] Implement user feedback loop

- [ ] Add batch processing for multiple papers## 🧪 Testing

- [ ] Optimize model quantization for smaller size

- [ ] Add author profile pages### Database Tests

- [ ] Implement semantic scholar integration```powershell

cd backend

---python db_utils.py

```

## 📞 Support

**Tests:**

For issues or questions:- ✅ Database initialization

- Open an issue on GitHub- ✅ Author upsert with duplicates

- Check API docs at `/docs` endpoint- ✅ Paper upsert with MD5 deduplication

- Review `RENDER_DEPLOYMENT.md` for deployment help- ✅ Co-author relationship building

- ✅ Query operations

---- ✅ Data integrity validation



## ⭐ Star This Project### Parser Tests

```powershell

If you find this useful, please star the repository!cd backend

python parser.py

[![GitHub stars](https://img.shields.io/github/stars/merajuddinmohammed/Reviewer-Recommendation-System?style=social)](https://github.com/merajuddinmohammed/Reviewer-Recommendation-System)```



---**Tests:**

- ✅ MD5 computation

**Built with ❤️ for the academic community**- ✅ Title extraction (metadata + heuristics)

- ✅ Year detection (multiple patterns)
- ✅ Abstract extraction
- ✅ Author name parsing
- ✅ Filename fallback
- ✅ Resilience to empty data

## 📖 Usage Examples

### Example 1: Basic Setup and Ingestion

```powershell
# 1. Run setup
.\setup.ps1

# 2. Organize PDFs
# papers/
#   Alice Smith/
#     paper1.pdf
#     paper2.pdf
#   Bob Johnson/
#     paper3.pdf

# 3. Activate environment
cd backend
.\.venv\Scripts\Activate.ps1

# 4. Run ingestion
python -c "from parser import walk_and_ingest; from pathlib import Path; results = walk_and_ingest(Path('../papers'), Path('papers.db')); print(results)"
```

### Example 2: Query Database

```python
from backend import db_utils

# List all authors
authors = db_utils.list_authors("papers.db")
for author in authors:
    print(f"{author['name']}: {author['paper_count']} papers")

# Get papers by author
alice = db_utils.find_author_by_name("papers.db", "Alice Smith")
if alice:
    papers = db_utils.get_author_papers("papers.db", alice['id'])
    for paper in papers:
        print(f"{paper['year']}: {paper['title']}")

# Get co-authors
coauthors = db_utils.get_coauthors("papers.db", alice['id'])
for coauthor in coauthors:
    print(f"{coauthor['coauthor_name']}: {coauthor['collaboration_count']} collaborations")
```

### Example 3: Custom PDF Processing

```python
from pathlib import Path
from backend.parser import (
    extract_text_with_fallback,
    extract_title,
    extract_year,
    extract_abstract,
    parse_author_names
)

# Process single PDF
pdf_path = Path("paper.pdf")

# Extract
text, metadata = extract_text_with_fallback(pdf_path)

# Parse
title = extract_title(text, pdf_path.name, metadata)
year = extract_year(text, metadata)
abstract = extract_abstract(text)
authors = parse_author_names(text, metadata)

print(f"Title: {title}")
print(f"Year: {year}")
print(f"Authors: {', '.join(authors)}")
print(f"Abstract: {abstract[:100]}...")
```

## 🐛 Troubleshooting

### Issue: `torch.cuda.is_available()` returns False

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify GPU compatibility with CUDA 11.8

### Issue: PDF extraction fails

**Solutions:**
1. Check if all PDF libraries are installed: `pip list | Select-String pdf`
2. Try different extraction method manually
3. Check if PDF is encrypted or scanned image
4. Use OCR for scanned PDFs (tesseract)

### Issue: Setup script fails

**Solutions:**
1. Run PowerShell as Administrator
2. Enable script execution:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Check Python version: `python --version` (must be 3.10+)
4. Use manual setup steps from `README-SETUP.md`

### Issue: Database locked error

**Solutions:**
1. Close all connections to database
2. WAL mode should prevent most locks
3. Check if another process is using the DB
4. Restart and try again

## 📊 Performance

### Benchmarks (typical hardware)

| Operation | Time | Notes |
|-----------|------|-------|
| Setup (first run) | ~5-10 min | Downloads PyTorch + deps |
| Setup (cached) | ~2 min | Packages cached |
| DB initialization | <1s | Creates empty schema |
| PDF ingestion | ~0.5-2s/PDF | Depends on PDF size |
| Query operations | <100ms | With proper indexes |

### Optimization Tips

1. **Pre-organize PDFs**: Use author-based folders from start
2. **Batch processing**: Process 100-1000 PDFs at once
3. **SSD storage**: Faster I/O for database and PDFs
4. **Skip full text**: Set `fulltext=None` if not needed for ML

## 🔜 Next Steps

### Completed Features ✅

1. ✅ **Vector Embeddings** - sentence-transformers with SciBERT/SPECTER
2. ✅ **FAISS Indexing** - Fast similarity search with GPU acceleration
3. ✅ **Topic Modeling** - BERTopic (optional module)
4. ✅ **TF-IDF Search** - Sparse keyword similarity

### Planned Features

1. **LambdaRank**
   - Personalized ranking with LightGBM
   - Training on user preferences

2. **REST API**
   - FastAPI endpoints
   - Search, recommendation, ranking

3. **Frontend UI**
   - React/Next.js interface
   - Paper browser and search

4. **Citation Analysis**
   - Citation network visualization
   - Influence metrics

### Implementation Priority

1. ✅ Environment setup
2. ✅ Database layer
3. ✅ PDF ingestion
4. ✅ Text utilities
5. ✅ TF-IDF similarity engine
6. ✅ Vector embeddings (SciBERT/SPECTER)
7. ✅ FAISS indexing
8. ✅ Topic modeling (BERTopic - optional)
9. 🔄 LambdaRank trainer (next)
10. 🔄 FastAPI backend
11. 🔄 Frontend UI

## 📝 Documentation

- **Main README**: `README.md` (this file)
- **Setup Guide**: `README-SETUP.md` - Detailed Windows setup
- **Parser Guide**: `backend/README-PARSER.md` - PDF ingestion details
- **Utils Guide**: `backend/README-UTILS.md` - Text utilities
- **TF-IDF Guide**: `backend/README-TFIDF.md` - Keyword similarity
- **Embeddings Guide**: `backend/README-EMBEDDING.md` - Semantic search
- **Topic Guide**: `backend/README-TOPIC.md` - Topic modeling (optional)
- **Technical Summary**: `SUMMARY.md` - Implementation details
- **Completion Reports**: `PROMPT6_COMPLETION.md`, `PROMPT7_COMPLETION.md`
- **Final Status**: `FINAL_PROJECT_STATUS.md` - Complete system overview
- **Inline Docs**: Comprehensive docstrings in all Python files

## 🤝 Contributing

### Code Style
- Use type hints
- Add docstrings for all functions
- Follow PEP 8 conventions
- Add unit tests for new features

### Testing
```powershell
# Run all tests
cd backend
python db_utils.py
python parser.py
```

### Git Workflow
```powershell
git add .
git commit -m "Description of changes"
git push
```

## 📄 License

[Add your license here]

## 👥 Authors

- **Meraj** - Initial implementation

## 🙏 Acknowledgments

- PyTorch team for CUDA support
- Hugging Face for transformers
- FastAPI team for the framework
- Open source PDF parsing libraries

---

**Version**: 1.0.0  
**Status**: Phase 1 Complete (Setup + Database + Parser + Utils + TF-IDF + Embeddings + Topics)  
**Prompts Completed**: 7 of 7 ✅  
**Last Updated**: December 2024  
**Python**: 3.10+  
**GPU**: CUDA 11.8

---

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting sections in:
   - `README-SETUP.md` - Setup issues
   - `backend/README-PARSER.md` - Parser issues
   - This file - General issues

2. Run tests to identify the problem:
   ```powershell
   python backend/db_utils.py
   python backend/parser.py
   ```

3. Check logs for error messages

4. Verify all prerequisites are installed correctly

---

**Happy Paper Mining! 📚🚀**

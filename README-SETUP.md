# Windows GPU Environment Setup Guide

This guide helps you set up a complete development environment for the Applied AI project on Windows with CUDA GPU support.

## Prerequisites

### Required Software

1. **Python 3.10+**
   - Download from: https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Node.js 20 LTS**
   - Download from: https://nodejs.org/
   - Recommended: Use the LTS version
   - Verify: `node --version` and `npm --version`

3. **NVIDIA GPU Drivers**
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Select your GPU model and Windows version
   - Verify: `nvidia-smi` (should display GPU info)

4. **CUDA Toolkit 11.8** (Optional but recommended)
   - Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - PyTorch will use CUDA if available
   - The setup script installs CUDA-enabled PyTorch wheels

### Checking Your GPU

Run this command to verify your NVIDIA GPU is detected:

```powershell
nvidia-smi
```

You should see output similar to:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
```

If you don't see this, install/update your NVIDIA drivers.

## Automated Setup

### Quick Start

Run the setup script from PowerShell:

```powershell
# Navigate to project directory
cd "C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment"

# Run setup script
.\setup.ps1
```

### Setup Script Options

```powershell
# Skip Node.js setup (Python only)
.\setup.ps1 -SkipNode

# Force recreate virtual environment
.\setup.ps1 -Force

# Combine flags
.\setup.ps1 -Force -SkipNode
```

### What the Script Does

1. ✅ Verifies Python 3.10+ is installed
2. ✅ Verifies Node.js 20 LTS is installed
3. ✅ Checks NVIDIA GPU availability with `nvidia-smi`
4. ✅ Creates `backend/.venv` Python virtual environment
5. ✅ Installs PyTorch with CUDA 11.8 support
6. ✅ Installs all Python dependencies with pinned versions
7. ✅ Verifies `torch.cuda.is_available()` returns `True`
8. ✅ Creates `backend/.env` with Hugging Face cache paths
9. ✅ Creates model cache directory structure
10. ✅ Installs Node.js dependencies in `frontend/`

## Python Dependencies

The setup script installs the following packages:

### Web Framework
- `fastapi==0.104.1` - Modern async web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `pydantic==2.5.0` - Data validation
- `python-multipart==0.0.6` - File upload support

### Data Science
- `numpy==1.24.3` - Numerical computing
- `pandas==2.0.3` - Data manipulation
- `scikit-learn==1.3.2` - Machine learning
- `scipy==1.11.4` - Scientific computing

### Machine Learning & Ranking
- `lightgbm==4.1.0` - LambdaRank implementation (CPU)
- `torch`, `torchvision`, `torchaudio` - PyTorch with CUDA 11.8
- `sentence-transformers==2.2.2` - Semantic embeddings
- `transformers==4.35.2` - Hugging Face transformers

### Vector Search
- `faiss-cpu==1.7.4` - Fast similarity search (CPU version for Windows stability)
  - Note: GPU FAISS is optional and can be installed manually if needed

### PDF Processing
- `pdfplumber==0.10.3` - PDF text extraction
- `pypdf2==3.0.1` - PDF manipulation
- `tika==2.6.0` - Apache Tika wrapper

### Topic Modeling
- `bertopic==0.15.0` - Topic modeling with transformers
- `umap-learn==0.5.5` - Dimensionality reduction
- `hdbscan==0.8.33` - Clustering algorithm

## Manual Setup (Fallback)

If the automated script fails, follow these manual steps:

### 1. Create Python Virtual Environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch with CUDA 11.8

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Core Dependencies

```powershell
pip install fastapi==0.104.1 "uvicorn[standard]==0.24.0" pydantic==2.5.0 python-multipart==0.0.6
```

### 4. Install Data Science Libraries

```powershell
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.2 scipy==1.11.4
```

### 5. Install ML Libraries

```powershell
pip install lightgbm==4.1.0 sentence-transformers==2.2.2 transformers==4.35.2 faiss-cpu==1.7.4
```

### 6. Install PDF Processing

```powershell
pip install pdfplumber==0.10.3 pypdf2==3.0.1 tika==2.6.0
```

### 7. Install Topic Modeling

```powershell
pip install bertopic==0.15.0 umap-learn==0.5.5 hdbscan==0.8.33
```

### 8. Verify CUDA

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 9. Create .env File

Create `backend/.env`:

```env
HF_HOME=./models/hf_cache
TRANSFORMERS_CACHE=./models/hf_cache
API_HOST=0.0.0.0
API_PORT=8000
CUDA_VISIBLE_DEVICES=0
```

### 10. Setup Frontend

```powershell
cd ..\frontend
npm install
```

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. **Check NVIDIA drivers:**
   ```powershell
   nvidia-smi
   ```
   If this fails, reinstall NVIDIA drivers.

2. **Verify PyTorch CUDA build:**
   ```powershell
   python -c "import torch; print(torch.version.cuda)"
   ```
   Should print `11.8` or similar.

3. **Check GPU detection:**
   ```powershell
   python -c "import torch; print(torch.cuda.device_count())"
   ```
   Should be greater than 0.

4. **Reinstall PyTorch:**
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Installation Errors

#### FAISS Installation Issues
If `faiss-cpu` fails to install:
```powershell
# Try conda instead (if using Anaconda)
conda install -c conda-forge faiss-cpu
```

#### LightGBM Compilation Errors
If LightGBM fails to build:
```powershell
# Install pre-built wheel
pip install lightgbm --install-option=--precompiled
```

#### Visual C++ Build Tools Missing
Some packages require Visual C++ build tools:
- Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install "Desktop development with C++" workload

### PowerShell Execution Policy

If you get "cannot be loaded because running scripts is disabled":

```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Virtual Environment Not Activating

If `.venv\Scripts\Activate.ps1` fails:

```powershell
# Alternative activation methods
.\.venv\Scripts\activate.bat  # Use batch file
# Or
.\.venv\Scripts\python.exe -m pip list  # Use full path
```

## Post-Setup Verification

### Check Installed Packages

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip list
```

### Run CUDA Test

Create `test_cuda.py`:

```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor creation
    x = torch.rand(5, 3).cuda()
    print(f"Test tensor on GPU: {x.device}")
else:
    print("WARNING: CUDA not available!")
```

Run it:
```powershell
python test_cuda.py
```

### Test FastAPI

Create `backend/main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
```

Start server:
```powershell
cd backend
uvicorn main:app --reload
```

Visit: http://localhost:8000

## Directory Structure

After setup, your project structure should look like:

```
Applied AI Assignment/
├── backend/
│   ├── .venv/              # Python virtual environment
│   ├── .env                # Environment variables
│   ├── models/
│   │   └── hf_cache/       # Hugging Face model cache
│   ├── main.py             # FastAPI application
│   └── requirements.txt    # (Optional) Generated dependency list
├── frontend/
│   ├── node_modules/       # Node.js dependencies
│   ├── package.json
│   └── package-lock.json
├── setup.ps1               # Setup script
└── README-SETUP.md         # This file
```

## Environment Variables

The `.env` file in `backend/` contains:

- `HF_HOME` - Hugging Face home directory for model downloads
- `TRANSFORMERS_CACHE` - Cache directory for transformer models
- `API_HOST` - API server host (default: 0.0.0.0)
- `API_PORT` - API server port (default: 8000)
- `CUDA_VISIBLE_DEVICES` - GPU device selection (0 = first GPU)

## Development Workflow

### Activate Environment

```powershell
# Every time you open a new terminal
cd backend
.\.venv\Scripts\Activate.ps1
```

### Run Backend

```powershell
cd backend
uvicorn main:app --reload
```

### Run Frontend

```powershell
cd frontend
npm run dev
```

### Update Dependencies

```powershell
# Python
cd backend
pip install <package>
pip freeze > requirements.txt

# Node.js
cd frontend
npm install <package>
```

## Performance Notes

### FAISS CPU vs GPU
- **faiss-cpu** is installed by default (Windows-safe)
- For better performance with large datasets, consider GPU FAISS:
  ```powershell
  pip uninstall faiss-cpu
  # Manual installation required - see FAISS documentation
  ```

### LightGBM
- CPU version is sufficient for LambdaRank
- Uses multi-threading for good performance

### Model Caching
- First run will download models (several GB)
- Models are cached in `backend/models/hf_cache/`
- Subsequent runs will be much faster

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)
- [Sentence Transformers](https://www.sbert.net/)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/)

## Support

If you encounter issues:

1. Check this README's Troubleshooting section
2. Verify all prerequisites are installed correctly
3. Run the setup script with verbose output
4. Check GPU compatibility with CUDA 11.8

---

**Setup script version:** 1.0.0  
**Last updated:** October 30, 2025

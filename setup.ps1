# Windows GPU Environment Setup Script
# Requires: Python 3.10+, Node 20 LTS
# Run with: .\setup.ps1

param(
    [switch]$SkipNode = $false,
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== Windows GPU Environment Bootstrap ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
    
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "ERROR: Python 3.10+ required, found Python $major.$minor" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "ERROR: Python not found. Install Python 3.10+ and add to PATH." -ForegroundColor Red
    exit 1
}

# Check Node version if not skipping
if (-not $SkipNode) {
    Write-Host "Checking Node.js version..." -ForegroundColor Yellow
    try {
        $nodeVersion = node --version 2>&1
        Write-Host "Found: Node $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Node.js not found. Install Node 20 LTS or use -SkipNode flag." -ForegroundColor Yellow
    }
}

# Check NVIDIA GPU
Write-Host "`nChecking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>&1
    Write-Host "GPU Info:" -ForegroundColor Green
    Write-Host $nvidiaSmi
} catch {
    Write-Host "WARNING: nvidia-smi not found. CUDA may not be available." -ForegroundColor Yellow
    Write-Host "Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
}

# Setup Python backend
Write-Host "`n=== Setting up Python Backend ===" -ForegroundColor Cyan

$backendDir = Join-Path $PSScriptRoot "backend"
if (-not (Test-Path $backendDir)) {
    Write-Host "Creating backend directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $backendDir | Out-Null
}

Set-Location $backendDir

# Create virtual environment
$venvPath = Join-Path $backendDir ".venv"
if (Test-Path $venvPath) {
    if ($Force) {
        Write-Host "Removing existing .venv (Force flag set)..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Green
        Write-Host "Use -Force to recreate." -ForegroundColor Gray
    }
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "Virtual environment created at: $venvPath" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
& $activateScript

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
Write-Host "`nInstalling PyTorch with CUDA 11.8 support..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
Write-Host "`nInstalling core Python dependencies..." -ForegroundColor Yellow
python -m pip install `
    fastapi==0.104.1 `
    "uvicorn[standard]==0.24.0" `
    pydantic==2.5.0 `
    python-multipart==0.0.6

# Install data science libraries
Write-Host "`nInstalling data science libraries..." -ForegroundColor Yellow
python -m pip install `
    numpy==1.24.3 `
    pandas==2.0.3 `
    scikit-learn==1.3.2 `
    scipy==1.11.4

# Install ML/ranking libraries
Write-Host "`nInstalling ML and ranking libraries..." -ForegroundColor Yellow
python -m pip install `
    lightgbm==4.1.0

# Install transformer and embedding libraries
Write-Host "`nInstalling transformer and embedding libraries..." -ForegroundColor Yellow
python -m pip install `
    sentence-transformers==2.2.2 `
    transformers==4.35.2

# Install FAISS (CPU version for Windows compatibility)
Write-Host "`nInstalling FAISS (CPU version)..." -ForegroundColor Yellow
python -m pip install faiss-cpu==1.7.4

# Install PDF processing libraries
Write-Host "`nInstalling PDF processing libraries..." -ForegroundColor Yellow
python -m pip install `
    pdfplumber==0.10.3 `
    pypdf2==3.0.1 `
    tika==2.6.0

# Install topic modeling libraries
Write-Host "`nInstalling topic modeling libraries..." -ForegroundColor Yellow
python -m pip install `
    bertopic==0.15.0 `
    umap-learn==0.5.5 `
    hdbscan==0.8.33

# Verify CUDA availability
Write-Host "`n=== Verifying CUDA Setup ===" -ForegroundColor Cyan
$cudaCheck = @"
import torch
import sys

print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA not available. Running on CPU only.')
    print('Check NVIDIA drivers and CUDA toolkit installation.')
"@

Write-Host "Running CUDA availability check..." -ForegroundColor Yellow
python -c $cudaCheck

# Create .env file
Write-Host "`nCreating .env configuration..." -ForegroundColor Yellow
$envContent = @"
# Hugging Face model cache directories
HF_HOME=./models/hf_cache
TRANSFORMERS_CACHE=./models/hf_cache

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
"@

$envPath = Join-Path $backendDir ".env"
Set-Content -Path $envPath -Value $envContent
Write-Host ".env file created at: $envPath" -ForegroundColor Green

# Create model cache directory
$modelCacheDir = Join-Path $backendDir "models\hf_cache"
if (-not (Test-Path $modelCacheDir)) {
    Write-Host "Creating model cache directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $modelCacheDir -Force | Out-Null
}

# Setup Node.js frontend
if (-not $SkipNode) {
    Write-Host "`n=== Setting up Node.js Frontend ===" -ForegroundColor Cyan
    
    $frontendDir = Join-Path $PSScriptRoot "frontend"
    if (-not (Test-Path $frontendDir)) {
        Write-Host "Creating frontend directory..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $frontendDir | Out-Null
    }
    
    Set-Location $frontendDir
    
    if (Test-Path "package.json") {
        Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
        npm install
        Write-Host "Node.js dependencies installed." -ForegroundColor Green
    } else {
        Write-Host "No package.json found. Skipping npm install." -ForegroundColor Yellow
        Write-Host "Run 'npm init' in the frontend directory to initialize." -ForegroundColor Gray
    }
}

# Return to root directory
Set-Location $PSScriptRoot

# Final summary
Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Python environment: $venvPath" -ForegroundColor Cyan
Write-Host "Environment file: $envPath" -ForegroundColor Cyan
Write-Host "Model cache: $modelCacheDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the Python environment, run:" -ForegroundColor Yellow
Write-Host "  .\backend\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To start the backend server:" -ForegroundColor Yellow
Write-Host "  cd backend" -ForegroundColor White
Write-Host "  uvicorn main:app --reload" -ForegroundColor White
Write-Host ""

if (-not $SkipNode -and (Test-Path (Join-Path $PSScriptRoot "frontend\package.json"))) {
    Write-Host "To start the frontend:" -ForegroundColor Yellow
    Write-Host "  cd frontend" -ForegroundColor White
    Write-Host "  npm run dev" -ForegroundColor White
    Write-Host ""
}

Write-Host "Setup complete. Happy coding! 🚀" -ForegroundColor Green

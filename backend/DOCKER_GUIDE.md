# Docker Build and Test Guide - Backend

This guide provides quick commands for building, testing, and deploying the backend Docker image.

---

## Prerequisites

- Docker Desktop installed and running
- Prebuilt models and data:
  - `data/papers.db` (SQLite database)
  - `data/faiss_index.faiss` (FAISS embeddings)
  - `data/id_map.npy` (ID mapping)
  - `models/tfidf_vectorizer.pkl` (TF-IDF model)
  - `models/lgbm_ranker.pkl` (LightGBM model)

---

## Build Commands

### Basic Build
```bash
cd backend
docker build -t reviewer-recommender-backend .
```

### Build with Tag
```bash
docker build -t reviewer-recommender-backend:v1.0 .
```

### Build with No Cache (Force Rebuild)
```bash
docker build --no-cache -t reviewer-recommender-backend .
```

### Build with Progress (Plain Output)
```bash
docker build --progress=plain -t reviewer-recommender-backend .
```

---

## Run Commands

### Basic Run (Port 8000)
```bash
docker run -p 8000:8000 reviewer-recommender-backend
```

### Run with Custom Port
```bash
docker run -p 5000:5000 -e PORT=5000 reviewer-recommender-backend
```

### Run in Detached Mode (Background)
```bash
docker run -d -p 8000:8000 --name backend reviewer-recommender-backend
```

### Run with Environment Variables
```bash
docker run -p 8000:8000 \
  -e BACKEND_DB=data/papers.db \
  -e FRONTEND_ORIGIN=http://localhost:5173 \
  reviewer-recommender-backend
```

### Run with Volume Mount (Custom Data)
```bash
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e BACKEND_DB=/app/data/papers.db \
  reviewer-recommender-backend
```

### Run Interactively (Debugging)
```bash
docker run -it --rm reviewer-recommender-backend /bin/bash
```

---

## Testing

### Health Check
```bash
# After starting container
curl http://localhost:8000/health

# Or with PowerShell
Invoke-WebRequest -Uri http://localhost:8000/health
```

### Test Recommendation Endpoint (JSON)
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning for Computer Vision",
    "abstract": "This paper presents a novel approach...",
    "k": 5
  }'
```

### Test Recommendation Endpoint (File Upload)
```bash
curl -X POST http://localhost:8000/recommend \
  -F "file=@sample.pdf" \
  -F "k=5"
```

### Check Logs
```bash
# View logs from running container
docker logs <container_id>

# Follow logs (live stream)
docker logs -f <container_id>

# View last 50 lines
docker logs --tail 50 <container_id>
```

---

## Container Management

### List Running Containers
```bash
docker ps
```

### List All Containers (Including Stopped)
```bash
docker ps -a
```

### Stop Container
```bash
docker stop <container_id>
# or
docker stop backend
```

### Remove Container
```bash
docker rm <container_id>
# or
docker rm backend
```

### Remove Image
```bash
docker rmi reviewer-recommender-backend
```

### Clean Up Unused Resources
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all unused resources
docker system prune -a
```

---

## Inspection and Debugging

### Check Image Size
```bash
docker images reviewer-recommender-backend
```

### View Image Layers
```bash
docker history reviewer-recommender-backend
```

### Inspect Container
```bash
docker inspect <container_id>
```

### Execute Commands in Running Container
```bash
# Open shell
docker exec -it <container_id> /bin/bash

# Run Python command
docker exec <container_id> python -c "import torch; print(torch.__version__)"

# Check models
docker exec <container_id> ls -lh /app/models/
```

### Check CPU Usage (Verify No GPU)
```bash
docker exec <container_id> python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: False
```

---

## Performance Testing

### Measure Response Time
```bash
# Using curl (Linux/Mac)
time curl http://localhost:8000/health

# Using PowerShell
Measure-Command { Invoke-WebRequest -Uri http://localhost:8000/health }
```

### Load Testing with Apache Bench
```bash
# 100 requests, 10 concurrent
ab -n 100 -c 10 http://localhost:8000/health
```

### Memory Usage
```bash
docker stats <container_id>
```

---

## Deployment

### Push to Docker Hub
```bash
# Tag image
docker tag reviewer-recommender-backend yourusername/reviewer-recommender-backend:v1.0

# Login
docker login

# Push
docker push yourusername/reviewer-recommender-backend:v1.0
```

### Deploy to Render
```bash
# Render automatically builds from Dockerfile in repo
# Just push to GitHub and connect to Render
git add backend/Dockerfile backend/requirements.txt
git commit -m "Add production Docker setup"
git push origin main
```

### Deploy to Railway
```bash
railway up
```

### Deploy to Fly.io
```bash
fly deploy
```

---

## Troubleshooting

### Container Exits Immediately
```bash
# Check logs for errors
docker logs <container_id>

# Run interactively to see errors
docker run -it --rm reviewer-recommender-backend
```

### Out of Memory
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 4 GB+

# Or run with memory limit
docker run -m 4g -p 8000:8000 reviewer-recommender-backend
```

### Port Already in Use
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /PID <pid> /F

# Or use different port
docker run -p 8001:8000 reviewer-recommender-backend
```

### Models Not Found
```bash
# Check if models directory exists in image
docker run --rm reviewer-recommender-backend ls -la /app/models/

# Check if data directory exists
docker run --rm reviewer-recommender-backend ls -la /app/data/

# If missing, rebuild image
docker build --no-cache -t reviewer-recommender-backend .
```

### Build Fails
```bash
# Check system dependencies
docker run --rm python:3.11-slim cat /etc/os-release

# Test pip install separately
docker run --rm -v "$(pwd):/app" python:3.11-slim pip install -r /app/requirements.txt

# Check disk space
docker system df
```

---

## Common Workflows

### Development Workflow
```bash
# 1. Make code changes
# 2. Rebuild image
docker build -t reviewer-recommender-backend .

# 3. Stop old container
docker stop backend
docker rm backend

# 4. Start new container
docker run -d -p 8000:8000 --name backend reviewer-recommender-backend

# 5. Test
curl http://localhost:8000/health
```

### Production Deployment
```bash
# 1. Build with version tag
docker build -t reviewer-recommender-backend:v1.0 .

# 2. Test locally
docker run -p 8000:8000 reviewer-recommender-backend:v1.0

# 3. Tag for registry
docker tag reviewer-recommender-backend:v1.0 yourusername/reviewer-recommender-backend:v1.0

# 4. Push to registry
docker push yourusername/reviewer-recommender-backend:v1.0

# 5. Deploy to production (Render, Railway, etc.)
```

---

## Windows-Specific Commands

### Build (PowerShell)
```powershell
cd "C:\Users\meraj\OneDrive\Desktop\Applied AI Assignment\backend"
docker build -t reviewer-recommender-backend .
```

### Run (PowerShell)
```powershell
docker run -p 8000:8000 reviewer-recommender-backend
```

### Volume Mount (PowerShell)
```powershell
docker run -p 8000:8000 `
  -v "${PWD}\data:/app/data" `
  -e BACKEND_DB=/app/data/papers.db `
  reviewer-recommender-backend
```

### Test (PowerShell)
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker build -t name .` | Build image |
| `docker run -p 8000:8000 name` | Run container |
| `docker ps` | List running containers |
| `docker logs <id>` | View logs |
| `docker stop <id>` | Stop container |
| `docker rm <id>` | Remove container |
| `docker rmi name` | Remove image |
| `docker exec -it <id> bash` | Shell into container |
| `docker system prune` | Clean up |

---

## Resources

- Docker Documentation: https://docs.docker.com/
- FastAPI Deployment: https://fastapi.tiangolo.com/deployment/docker/
- Render Docs: https://render.com/docs
- Railway Docs: https://docs.railway.app/
- Fly.io Docs: https://fly.io/docs/

---

**Happy Dockerizing! 🐳**

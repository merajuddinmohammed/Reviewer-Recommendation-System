# 🚂 Railway.app Deployment Guide

## Prerequisites

1. **GitHub Account** with your repository
2. **Railway.app Account** (free tier available)
3. **Git** installed locally
4. **All changes committed** to GitHub

---

## Step-by-Step Deployment

### Step 1: Create Railway Account

1. Go to [https://railway.app](https://railway.app)
2. Click **"Start a New Project"** or **"Login with GitHub"**
3. Authorize Railway to access your GitHub repositories

---

### Step 2: Create New Project

1. Click **"New Project"** on Railway dashboard
2. Select **"Deploy from GitHub repo"**
3. Choose your repository: `merajuddinmohammed/Reviewer-Recommendation-System`
4. Railway will automatically detect it's a Python project

---

### Step 3: Configure Deployment Settings

#### Option A: Automatic Configuration (Recommended)

Railway will auto-detect:
- ✅ Python project
- ✅ `requirements.txt` 
- ✅ `Procfile` for startup command
- ✅ `railway.json` for health checks

#### Option B: Manual Configuration

If auto-detection doesn't work:

1. Go to **Settings** → **Deploy**
2. Set **Root Directory**: `/backend`
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

### Step 4: Set Environment Variables

Railway needs environment variables for your app:

1. Go to **Variables** tab
2. Add these variables:

```
PORT=8000
FRONTEND_ORIGIN=https://your-frontend-url.vercel.app
PYTHON_VERSION=3.11
```

**Important Environment Variables:**

| Variable | Value | Description |
|----------|-------|-------------|
| `PORT` | `8000` | Port for Railway (Railway will override with `$PORT`) |
| `FRONTEND_ORIGIN` | Your frontend URL | CORS allowed origin |
| `PYTHON_VERSION` | `3.11` | Python version to use |

---

### Step 5: Configure Root Directory

**CRITICAL**: Your backend code is in the `/backend` folder, so:

1. Go to **Settings** → **Service Settings**
2. Set **Root Directory**: `backend`
3. Click **Save**

---

### Step 6: Deploy

1. Click **Deploy** button or push to GitHub
2. Railway will:
   - ✅ Clone your repository
   - ✅ Install Python 3.11
   - ✅ Install dependencies from `requirements.txt`
   - ✅ Run health check on `/health` endpoint
   - ✅ Start your FastAPI app

**Deployment Time**: ~5-10 minutes (first time)

---

### Step 7: Monitor Deployment

1. Go to **Deployments** tab
2. Watch the build logs:
   ```
   ⚙️ Building...
   📦 Installing dependencies...
   ✅ Build successful!
   🚀 Starting service...
   ✅ Healthy!
   ```

3. If deployment fails, check:
   - Build logs for errors
   - `/health` endpoint is accessible
   - All required files are committed

---

### Step 8: Get Your Backend URL

1. Once deployed, Railway provides a URL like:
   ```
   https://your-app-name.railway.app
   ```

2. Test your backend:
   ```bash
   curl https://your-app-name.railway.app/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "models_loaded": true,
     "tfidf_model": true,
     "faiss_index": true,
     "bertopic_model": false,
     "lgbm_model": true
   }
   ```

---

### Step 9: Update Frontend with New Backend URL

1. Go to your frontend deployment (Vercel/Netlify)
2. Update environment variable:
   ```
   VITE_API_URL=https://your-app-name.railway.app
   ```

3. Redeploy frontend to use new backend

---

### Step 10: Enable Custom Domain (Optional)

1. Go to **Settings** → **Domains**
2. Click **Generate Domain** (free Railway subdomain)
3. Or add your custom domain:
   - Click **Custom Domain**
   - Add CNAME record: `your-domain.com` → `your-app-name.railway.app`

---

## Railway vs Render Comparison

| Feature | Railway.app | Render.com |
|---------|-------------|------------|
| **Free Tier** | $5 credit/month | 750 hours/month |
| **Cold Start** | ~1-2 seconds | ~30-60 seconds |
| **Build Time** | ~3-5 minutes | ~5-10 minutes |
| **Memory Limit** | 8 GB (free) | 512 MB (free) |
| **Auto-deploy** | ✅ Yes | ✅ Yes |
| **Custom Domain** | ✅ Yes | ✅ Yes |
| **Persistent Storage** | ✅ Yes | ❌ No (free tier) |

**Recommendation**: Railway.app is **MUCH faster** than Render for cold starts!

---

## Troubleshooting

### Issue 1: Build Fails with "ModuleNotFoundError"

**Solution**: Check `requirements.txt` has all dependencies:
```bash
cd backend
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements.txt"
git push
```

### Issue 2: Health Check Fails

**Solution**: Ensure `/health` endpoint returns 200 status:
```python
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Issue 3: Models Not Loading

**Solution**: Ensure model files are committed to git:
```bash
git add backend/models/*.pkl
git add backend/data/*.faiss
git add backend/data/*.npy
git commit -m "Add model files"
git push
```

**Note**: If model files are too large (>100MB), use Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.faiss"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

### Issue 4: Port Binding Error

**Solution**: Use `$PORT` environment variable:
```python
import os
port = int(os.getenv("PORT", 8000))
```

In Procfile:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Issue 5: CORS Errors

**Solution**: Update `app.py` to allow Railway domain:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify Railway URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Post-Deployment Checklist

- [ ] Backend health check passes
- [ ] `/health` endpoint returns 200
- [ ] Models load successfully
- [ ] Frontend can connect to backend
- [ ] CORS is configured correctly
- [ ] Environment variables are set
- [ ] Custom domain configured (optional)
- [ ] Auto-deploy enabled for GitHub pushes

---

## Useful Railway CLI Commands

Install Railway CLI:
```bash
npm i -g @railway/cli
```

Login:
```bash
railway login
```

Link project:
```bash
railway link
```

View logs:
```bash
railway logs
```

Open in browser:
```bash
railway open
```

Deploy manually:
```bash
railway up
```

---

## Cost Estimation

**Railway Free Tier**:
- $5 credit/month (enough for ~100 hours of uptime)
- After credit exhausted, service pauses until next month

**Railway Pro Plan** ($20/month):
- Unlimited usage
- Priority support
- No cold starts
- Persistent storage

**Recommendation**: Start with free tier, upgrade if needed!

---

## Next Steps

1. ✅ Deploy backend to Railway
2. ✅ Update frontend environment variable
3. ✅ Test end-to-end flow
4. ✅ Monitor Railway dashboard for issues
5. ✅ Set up alerts for downtime (optional)

---

## Support

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Railway Status**: https://status.railway.app

---

**Happy Deploying! 🚂**

# 🚀 Render Deployment Guide

## Overview
This guide will help you deploy both the backend (FastAPI) and frontend (React) to Render.

## Prerequisites
- ✅ GitHub repository: https://github.com/merajuddinmohammed/Reviewer-Recommendation-System
- ✅ Render account (free tier works): https://render.com
- ✅ All code pushed to GitHub
- ✅ Models committed to repository (or available in cloud storage)

---

## 📦 Part 1: Deploy Backend (FastAPI)

### Step 1: Go to Render Dashboard
1. Go to https://render.com and sign in
2. Click **"New +"** button in top right
3. Select **"Web Service"**

### Step 2: Connect GitHub Repository
1. Click **"Connect account"** to connect GitHub (if not already)
2. Search for: `Reviewer-Recommendation-System`
3. Click **"Connect"** on your repository

### Step 3: Configure Backend Service
Fill in these settings:

**Basic Settings:**
- **Name:** `reviewer-recommendation-backend`
- **Region:** Choose closest to you (e.g., `Oregon (US West)`)
- **Branch:** `master`
- **Root Directory:** `backend`
- **Runtime:** `Python 3`

**Build & Deploy:**
- **Build Command:**
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```

- **Start Command:**
  ```bash
  uvicorn app:app --host 0.0.0.0 --port $PORT
  ```

**Instance Type:**
- Select: **Free** (or Starter $7/month for better performance)

**Environment Variables:**
Click "Advanced" and add:
```
PYTHON_VERSION=3.11.0
```

**Health Check Path:**
```
/health
```

### Step 4: Deploy Backend
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for deployment
3. You'll get a URL like: `https://reviewer-recommendation-backend.onrender.com`
4. Test it by visiting: `https://your-backend-url.onrender.com/health`

**Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "tfidf_vectorizer": true,
    "faiss_index": true,
    "id_map": true,
    "lgbm_model": true
  },
  "database_path": "data/papers.db"
}
```

---

## 🎨 Part 2: Deploy Frontend (React + Vite)

### Step 1: Create New Static Site
1. Click **"New +"** again
2. Select **"Static Site"**
3. Connect the same repository

### Step 2: Configure Frontend Service
Fill in these settings:

**Basic Settings:**
- **Name:** `reviewer-recommendation-frontend`
- **Branch:** `master`
- **Root Directory:** `frontend`

**Build Settings:**
- **Build Command:**
  ```bash
  npm install && npm run build
  ```

- **Publish Directory:**
  ```
  dist
  ```

**Environment Variables:**
Add this important variable:
```
VITE_API_BASE=https://reviewer-recommendation-backend.onrender.com
```
⚠️ **Important:** Replace with YOUR actual backend URL from Step 1!

### Step 3: Add Rewrite Rules (for SPA routing)
In the **Redirects/Rewrites** section, add:
- **Source:** `/*`
- **Destination:** `/index.html`
- **Action:** `Rewrite`

### Step 4: Deploy Frontend
1. Click **"Create Static Site"**
2. Wait 3-5 minutes for deployment
3. You'll get a URL like: `https://reviewer-recommendation-frontend.onrender.com`

---

## ✅ Part 3: Verification

### Test Backend:
1. Visit: `https://your-backend-url.onrender.com/docs`
2. You should see the FastAPI Swagger UI
3. Test the `/health` endpoint
4. Try the `/recommend` endpoint with a test PDF

### Test Frontend:
1. Visit: `https://your-frontend-url.onrender.com`
2. You should see the upload interface
3. Try uploading a PDF
4. Verify recommendations appear

---

## 🐛 Troubleshooting

### Backend Issues:

**"Models not found" error:**
- Check if model files are in GitHub (they're large ~2MB each)
- If models aren't in Git, you need to:
  - Option A: Upload to Git LFS (GitHub Large File Storage)
  - Option B: Store in S3/Google Cloud and download during build
  - Option C: Build models during deployment (slow)

**"Out of memory" error:**
- Upgrade to Starter plan ($7/month) with 512MB RAM
- Free tier has 512MB which might be tight with models loaded

**"Application failed to respond" error:**
- Check logs in Render dashboard
- Ensure `$PORT` is used in start command
- Check if all dependencies installed correctly

### Frontend Issues:

**"Failed to fetch" or CORS errors:**
- Verify `VITE_API_BASE` environment variable is correct
- Check backend is running and accessible
- Backend must allow CORS from frontend domain

**Blank page or 404 errors:**
- Verify rewrite rule is set: `/*` → `/index.html`
- Check build command completed successfully
- Ensure `dist` folder was published

---

## 💰 Cost Estimate

**Free Tier (Both services):**
- ✅ Backend: Free (with 750 hours/month, spins down after 15 min inactivity)
- ✅ Frontend: Free (100GB bandwidth/month)
- ⚠️ Cold starts: 30-60 seconds when inactive

**Paid Tier (Recommended for production):**
- Backend Starter: $7/month (always on, 512MB RAM)
- Frontend: Free (static sites are always free)
- Total: **$7/month**

---

## 🚀 Quick Deploy Checklist

- [ ] Created Render account
- [ ] Connected GitHub repository
- [ ] Deployed backend service
- [ ] Noted backend URL
- [ ] Deployed frontend service
- [ ] Set `VITE_API_BASE` environment variable
- [ ] Added rewrite rule for frontend
- [ ] Tested `/health` endpoint
- [ ] Tested uploading PDF
- [ ] Verified recommendations work

---

## 📝 Important Notes

1. **First deployment takes 10-15 minutes** (installing dependencies + models)
2. **Free tier sleeps after 15 minutes** of inactivity (first request takes 30-60s to wake)
3. **Models are ~2MB total** - should fit in free tier
4. **Database is included** in the repo (papers.db)
5. **No GPU on Render free tier** - uses CPU inference (still fast enough)

---

## 🔗 URLs After Deployment

After completing deployment, update this section with your URLs:

**Backend API:**
- URL: `https://reviewer-recommendation-backend.onrender.com`
- Health Check: `https://reviewer-recommendation-backend.onrender.com/health`
- API Docs: `https://reviewer-recommendation-backend.onrender.com/docs`

**Frontend App:**
- URL: `https://reviewer-recommendation-frontend.onrender.com`

---

## 🎯 Next Steps After Deployment

1. **Custom Domain (Optional):**
   - Add your own domain in Render dashboard
   - Update DNS records
   - Free SSL certificate included

2. **Monitoring:**
   - Check Render dashboard for logs
   - Monitor response times
   - Set up email alerts for failures

3. **Optimization:**
   - Consider upgrading to Starter plan to remove cold starts
   - Enable caching for faster responses
   - Compress models if deployment is slow

---

Need help? Check the Render documentation: https://render.com/docs

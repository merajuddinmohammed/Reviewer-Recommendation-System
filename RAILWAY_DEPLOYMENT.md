# 🚂 Railway.app Deployment Guide - Step by Step

## Your Current Setup
- ✅ **Frontend**: https://reviewer-recommendation-system-1.onrender.com/ (Render)
- 🔧 **Backend**: Deploying to Railway.app (this guide)

---

## Prerequisites Checklist

Before you start, make sure you have:
- ✅ GitHub Account with repository pushed
- ✅ All latest changes committed to GitHub (LambdaRank fixes included)
- ✅ Railway configuration files committed:
  - `backend/Procfile`
  - `backend/railway.json`
  - `backend/nixpacks.toml`
  - `backend/requirements.txt`

---

## Step-by-Step Deployment (10 Minutes)

### Step 1: Create Railway Account (2 minutes)

1. **Open Railway.app**:
   - Go to: https://railway.app
   
2. **Sign Up/Login**:
   - Click **"Login with GitHub"**
   - Authorize Railway to access your repositories
   - Accept permissions

✅ **You're now logged into Railway!**

---

### Step 2: Create New Project (1 minute)

1. **On Railway Dashboard**:
   - Click **"New Project"** (big purple button)
   
2. **Select Deployment Method**:
   - Choose **"Deploy from GitHub repo"**
   
3. **Choose Your Repository**:
   - Find and select: `merajuddinmohammed/Reviewer-Recommendation-System`
   - Click on it

4. **Wait for Initial Setup**:
   - Railway will analyze your repository
   - It will auto-detect it's a Python project

✅ **Project created!**

---

### Step 3: Configure Root Directory (CRITICAL!) (2 minutes)

Your backend code is in the `backend/` folder, not the root. You MUST configure this:

1. **In Railway Dashboard**:
   - Click on your service (you should see it)
   
2. **Go to Settings**:
   - Click **"Settings"** tab on the left
   
3. **Set Root Directory**:
   - Find **"Root Directory"** section
   - Click **"Configure"** or the edit icon
   - Enter: `backend`
   - Click **"Save"** or checkmark

✅ **Root directory configured!**

---

### Step 4: Add Environment Variables (Optional) (2 minutes)

1. **Go to Variables Tab**:
   - Click **"Variables"** tab on the left
   
2. **Add Frontend URL** (recommended):
   - Click **"+ New Variable"**
   - **Variable Name**: `FRONTEND_ORIGIN`
   - **Value**: `https://reviewer-recommendation-system-1.onrender.com`
   - Click **"Add"**

**Note**: Railway auto-provides `PORT` variable, so you don't need to add it!

✅ **Environment variables set!**

---

### Step 5: Deploy! (5-10 minutes)

Railway will now automatically deploy your backend:

1. **Watch the Deployment**:
   - Click **"Deployments"** tab on the left
   - You'll see the current deployment building
   
2. **Monitor Build Logs**:
   - Click on the deployment to see logs in real-time
   - You should see:
     ```
     ⚙️ Preparing build...
     📦 Installing Python 3.11...
     📥 Installing dependencies from requirements.txt...
     ✅ Build complete
     🚀 Starting application...
     ✅ Application started on port $PORT
     ```

3. **Wait for "Running" Status**:
   - Deployment status will change from "Building" → "Running"
   - First deployment takes ~5-10 minutes

✅ **Backend deployed!**

---

### Step 6: Get Your Backend URL (1 minute)

1. **Go to Networking Tab**:
   - Click **"Networking"** tab on the left
   
2. **Find Public Domain**:
   - Under "Public Networking", you'll see a generated domain like:
     ```
     reviewer-recommendation-backend-production-XXXX.up.railway.app
     ```
   - **Copy this URL** (you'll need it!)

✅ **Backend URL obtained!**

---

### Step 7: Test Your Backend (1 minute)

Let's verify your backend is working:

1. **Test Health Endpoint**:
   ```bash
   curl https://your-railway-url.railway.app/health
   ```
   
   **Expected Response**:
   ```json
   {
     "status": "healthy",
     "models_loaded": true,
     "tfidf_model": true,
     "faiss_index": true,
     "lgbm_model": true
   }
   ```

2. **If you get an error**:
   - Check Deployments tab for errors
   - See Troubleshooting section below

✅ **Backend is healthy!**

---

### Step 8: Update Frontend to Use Railway Backend (5 minutes)

Now connect your frontend to the new Railway backend:

1. **Go to Render Dashboard**:
   - Visit: https://dashboard.render.com
   - Find your frontend service: `reviewer-recommendation-system-1`
   
2. **Update Environment Variable**:
   - Click on your frontend service
   - Go to **"Environment"** tab
   - Find `VITE_API_URL` or add new variable:
     - **Key**: `VITE_API_URL`
     - **Value**: `https://your-railway-url.railway.app` *(your Railway backend URL)*
     - **Important**: No trailing slash!
   - Click **"Save Changes"**
   
3. **Redeploy Frontend**:
   - Go to **"Manual Deploy"** section
   - Click **"Deploy latest commit"**
   - Wait 2-3 minutes for redeploy

✅ **Frontend connected to Railway backend!**

---

### Step 9: Test End-to-End (2 minutes)

1. **Open Your Frontend**:
   - Go to: https://reviewer-recommendation-system-1.onrender.com/
   
2. **Submit a Test Paper**:
   - Enter a paper title and abstract
   - Add some authors (optional)
   - Click submit
   
3. **Verify Recommendations**:
   - You should see reviewer recommendations
   - Check browser console (F12) for any errors
   - LambdaRank should be working (53% P@5 accuracy)

✅ **Everything working end-to-end!**

---

### Step 10: Enable Auto-Deploy (1 minute)

Set up automatic deployments when you push to GitHub:

1. **In Railway Settings**:
   - Go to **"Settings"** tab
   - Find **"Service"** section
   
2. **Verify Auto-Deploy**:
   - Make sure **"Automatic Deployments"** is enabled
   - It should be ON by default
   
3. **Test It**:
   - Make a small change to your code
   - Push to GitHub
   - Railway will automatically deploy!

✅ **Auto-deploy enabled!**

---

## 🔧 Troubleshooting Common Issues

### Issue 1: "Application not found" or 404 Error

**Cause**: Root directory not set correctly

**Solution**:
1. Go to Railway **Settings** tab
2. Set **Root Directory** to: `backend`
3. Click **Save**
4. Wait for redeploy

---

### Issue 2: Build Fails with "ModuleNotFoundError"

**Cause**: Missing dependencies in requirements.txt

**Solution**:
1. Check Railway build logs for the missing module
2. Add it to `backend/requirements.txt`
3. Commit and push:
   ```bash
   git add backend/requirements.txt
   git commit -m "Add missing dependency"
   git push
   ```
4. Railway will auto-deploy

---

### Issue 3: Models Not Loading

**Cause**: Model files too large or not committed

**Check file sizes**:
```bash
cd backend
ls -lh models/*.pkl data/*.faiss
```

**If files are >100MB**, use Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.faiss"
git lfs track "*.db"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

---

### Issue 4: Health Check Returns Error

**Cause**: Application not starting properly

**Solution**:
1. Check Railway **Deployments** logs
2. Look for Python errors or missing files
3. Common fixes:
   - Ensure all model files exist in `backend/models/`
   - Ensure database exists in `backend/data/papers.db`
   - Check `backend/app.py` loads correctly

---

### Issue 5: Frontend Can't Connect to Backend

**Cause**: CORS or wrong URL

**Solution**:
1. Verify backend URL is correct (no trailing slash)
2. Check CORS in `app.py` allows your frontend:
   ```python
   allow_origins=["*"]  # Should already allow all origins
   ```
3. Check browser console for specific error messages

---

### Issue 6: "Port already in use" Error

**Cause**: Port configuration issue

**Your Procfile should have**:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Railway automatically provides `$PORT`, so this should work by default.

---

## ✅ Post-Deployment Checklist

Use this to verify everything is working:

- [ ] Railway deployment shows "Running" status
- [ ] Health check passes: `curl https://your-url.railway.app/health`
- [ ] Frontend environment variable updated with Railway URL
- [ ] Frontend redeployed on Render
- [ ] Can submit paper and get recommendations
- [ ] No CORS errors in browser console
- [ ] LambdaRank model working (check recommendation quality)
- [ ] Auto-deploy enabled for future updates

---

## 📊 What Your Backend Includes

Your Railway deployment includes:

✅ **Improved LambdaRank Model**:
- 53.33% P@5 accuracy (was 0%)
- 49.59% nDCG@10
- Trained on 5000 samples with 40% positive labels

✅ **All Models**:
- `lgbm_ranker.pkl` - LambdaRank model (5.6 KB)
- `tfidf_vectorizer.pkl` - TF-IDF model
- `faiss_index.faiss` - Vector search index
- `id_map.npy` - FAISS ID mapping

✅ **Database**:
- `papers.db` - SQLite database with papers and authors

✅ **Fast Performance**:
- Cold start: ~1-2 seconds
- Warm requests: <100ms

---

## 🎯 Quick Command Reference

```bash
# Test backend health
curl https://your-railway-url.railway.app/health

# View Railway logs (requires Railway CLI)
railway logs

# Redeploy manually (requires Railway CLI)
railway up

# Link local project to Railway (requires Railway CLI)
railway link
```

---

## 💰 Cost & Usage

**Railway Free Tier**:
- $5 credit per month
- ~100 hours of uptime
- Resets monthly
- Perfect for testing and light use

**When to Upgrade to Pro ($20/month)**:
- Need 24/7 uptime
- High traffic (>100 hours/month)
- Need priority support
- Want no cold starts

---

## 📚 Helpful Resources

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway (great community!)
- **Railway Status**: https://status.railway.app
- **Your Repository**: https://github.com/merajuddinmohammed/Reviewer-Recommendation-System

---

## 🚀 Next Steps After Deployment

1. **Monitor Your App**:
   - Check Railway dashboard daily
   - Set up alerts for failures
   - Monitor credit usage

2. **Optimize Performance**:
   - Review API response times
   - Check model loading times
   - Monitor memory usage

3. **Future Improvements**:
   - Add caching for common queries
   - Implement rate limiting
   - Add API analytics

---

**Your backend is now deployed on Railway! Much faster than Render! ⚡**

**Need Help?** Check the troubleshooting section or contact Railway support.

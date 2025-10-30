# 🚀 Deployment Information

## Frontend
- **URL**: https://reviewer-recommendation-system-1.onrender.com/
- **Platform**: Render.com
- **Status**: ✅ Deployed

## Backend Options

### Option 1: Current Setup (Render.com)
If you're also deploying backend to Render:
- **Expected URL**: `https://reviewer-recommendation-system-production.up.railway.app` or similar
- **CORS**: Already configured to accept all origins (`allow_origins=["*"]`)
- **Health Check**: `/health` endpoint available

### Option 2: Deploy to Railway.app (Recommended - Faster!)
See `RAILWAY_DEPLOYMENT.md` for full instructions.

**Quick Steps:**
1. Go to https://railway.app
2. Login with GitHub
3. Create New Project → Deploy from GitHub
4. Select: `merajuddinmohammed/Reviewer-Recommendation-System`
5. Set Root Directory: `backend`
6. Wait for deployment (~5-10 minutes)
7. Get your Railway URL (e.g., `https://your-app.railway.app`)

---

## Environment Variables

### Backend Environment Variables
Set these in your deployment platform (Render/Railway):

```bash
# Optional - defaults to localhost:5173
FRONTEND_ORIGIN=https://reviewer-recommendation-system-1.onrender.com

# Railway auto-provides this
PORT=8000

# Python version
PYTHON_VERSION=3.11
```

### Frontend Environment Variables
Update your frontend deployment with backend URL:

```bash
# If deploying backend to Railway
VITE_API_URL=https://your-backend-app.railway.app

# If deploying backend to Render
VITE_API_URL=https://your-backend-app.onrender.com
```

---

## Testing Your Deployment

### 1. Test Backend Health
Once backend is deployed, test the health endpoint:

```bash
curl https://your-backend-url.com/health
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

### 2. Test Frontend Connection
Open your frontend: https://reviewer-recommendation-system-1.onrender.com/

Try submitting a paper and check browser console for:
- ✅ No CORS errors
- ✅ API calls successful
- ✅ Recommendations displayed

---

## CORS Configuration (Already Set Up!)

Your backend (`backend/app.py`) already has CORS configured:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (including your frontend)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This means:
- ✅ Your frontend can make requests from any domain
- ✅ No CORS errors expected
- ✅ Works with Render, Railway, Vercel, Netlify, etc.

**Note**: For production, you can restrict to specific origins:
```python
allow_origins=[
    "https://reviewer-recommendation-system-1.onrender.com",
    "http://localhost:5173"  # For local development
]
```

---

## Deployment Checklist

### Backend Deployment
- [ ] Choose platform (Railway recommended, or Render)
- [ ] Deploy backend from GitHub
- [ ] Set root directory to `backend`
- [ ] Wait for build to complete
- [ ] Test `/health` endpoint
- [ ] Copy backend URL

### Frontend Configuration
- [ ] Update `VITE_API_URL` in frontend deployment
- [ ] Redeploy frontend
- [ ] Test end-to-end flow

### Verification
- [ ] Frontend loads successfully
- [ ] Can submit paper for review
- [ ] Recommendations appear
- [ ] No console errors
- [ ] LambdaRank model working (check recommendations quality)

---

## Important Files

### Backend Configuration
- `backend/Procfile` - Startup command for deployment
- `backend/railway.json` - Railway configuration
- `backend/requirements.txt` - Python dependencies
- `backend/app.py` - FastAPI application

### Models (Included in Deployment)
- `backend/models/lgbm_ranker.pkl` - LambdaRank model (53% P@5)
- `backend/models/tfidf_vectorizer.pkl` - TF-IDF model
- `backend/data/faiss_index.faiss` - FAISS vector index
- `backend/data/id_map.npy` - ID mapping for FAISS
- `backend/data/papers.db` - SQLite database

---

## Troubleshooting

### Issue: Frontend can't connect to backend
**Solution**: 
1. Check `VITE_API_URL` is set correctly in frontend
2. Verify backend is deployed and health check passes
3. Check browser console for errors

### Issue: CORS errors
**Solution**: 
- Backend already allows all origins
- If still seeing errors, check backend logs
- Ensure backend URL doesn't have trailing slash

### Issue: Models not loading
**Solution**:
1. Check backend logs during startup
2. Verify model files are committed to git
3. Check file sizes (use Git LFS if >100MB)

### Issue: Slow cold starts (Render)
**Solution**: 
- Deploy to Railway instead (1-2 second cold starts vs 30-60 seconds)
- Or upgrade to Render paid plan

---

## Performance Comparison

| Platform | Cold Start | Build Time | Free Tier | Best For |
|----------|-----------|------------|-----------|----------|
| **Railway** | ~1-2 sec | ~5 min | $5/month credit | Production |
| **Render** | ~30-60 sec | ~8 min | 750 hrs/month | Testing |
| **Local** | Instant | N/A | Free | Development |

**Recommendation**: Deploy backend to Railway.app for best performance!

---

## Next Steps

1. **Deploy Backend**
   - Follow `RAILWAY_DEPLOYMENT.md` for Railway
   - Or deploy to Render if preferred

2. **Update Frontend**
   - Set backend URL in environment variables
   - Redeploy frontend

3. **Test**
   - Submit test paper
   - Verify recommendations
   - Check LambdaRank is working

4. **Monitor**
   - Watch deployment logs
   - Set up alerts for downtime
   - Monitor API response times

---

## Support & Resources

- **Railway Docs**: https://docs.railway.app
- **Render Docs**: https://render.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Repository**: https://github.com/merajuddinmohammed/Reviewer-Recommendation-System

---

**Your frontend is live! Deploy the backend to complete your setup.** 🚀

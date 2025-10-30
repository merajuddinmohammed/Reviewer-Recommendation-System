# Frontend Deployment Guide

Quick reference for deploying the React frontend to various platforms.

---

## Option 1: Render Static Site (Recommended)

**Why**: Free tier, automatic builds, CDN included, no Docker needed

### Steps

1. **Push to GitHub**:
   ```bash
   git add frontend/
   git commit -m "Add frontend for deployment"
   git push origin main
   ```

2. **Create Static Site**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Static Site"
   - Connect to GitHub repository
   - Select branch: `main`

3. **Configure Build**:
   ```
   Name: reviewer-recommender-frontend
   Root Directory: frontend
   Build Command: npm ci && npm run build
   Publish Directory: dist
   ```

4. **Add Environment Variables**:
   ```
   VITE_API_BASE=https://your-backend.onrender.com
   ```

5. **Deploy**:
   - Click "Create Static Site"
   - Wait 2-5 minutes
   - Your site will be live at: `https://your-frontend.onrender.com`

### After Deployment

**Test**:
- Visit your URL
- Try uploading a PDF
- Try pasting an abstract
- Check that recommendations load

**Update Backend CORS**:
```python
# In backend/app.py
FRONTEND_ORIGIN = "https://your-frontend.onrender.com"
```

**Redeploy Backend**:
- Render will auto-deploy if connected to GitHub
- Or manually trigger deployment

---

## Option 2: Render Docker Web Service

**Why**: Full control over nginx configuration, multiple environments

### Steps

1. **Push to GitHub** (with Dockerfile):
   ```bash
   git add frontend/Dockerfile frontend/nginx.conf frontend/.dockerignore
   git commit -m "Add Docker setup"
   git push origin main
   ```

2. **Create Web Service**:
   - Render Dashboard → "New +" → "Web Service"
   - Connect to GitHub repository
   - Select branch: `main`
   - Environment: **Docker**

3. **Configure**:
   ```
   Name: reviewer-recommender-frontend
   Root Directory: frontend
   Instance Type: Free (or Starter $7/mo)
   Health Check Path: /health
   ```

4. **Environment Variables**:
   ```
   PORT=10000  # Render assigns automatically
   ```

5. **Deploy**:
   - Click "Create Web Service"
   - Wait 5-10 minutes
   - Live at: `https://your-frontend.onrender.com`

---

## Option 3: Netlify

**Why**: Excellent DX, instant rollbacks, preview deploys

### Via Web UI

1. **Push to GitHub**
2. Go to [Netlify](https://app.netlify.com/)
3. Click "Add new site" → "Import an existing project"
4. Connect to GitHub
5. Configure:
   ```
   Base directory: frontend
   Build command: npm ci && npm run build
   Publish directory: dist
   ```
6. Add environment variable:
   ```
   VITE_API_BASE=https://your-backend.com
   ```
7. Deploy

### Via CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
cd frontend
netlify deploy --prod

# Follow prompts:
# - Build command: npm ci && npm run build
# - Publish directory: dist
```

---

## Option 4: Vercel

**Why**: Optimized for React/Vite, automatic preview deployments

### Via Web UI

1. **Push to GitHub**
2. Go to [Vercel](https://vercel.com/)
3. Click "Add New" → "Project"
4. Import from GitHub
5. Configure:
   ```
   Framework Preset: Vite
   Root Directory: frontend
   Build Command: npm ci && npm run build
   Output Directory: dist
   ```
6. Add environment variable:
   ```
   VITE_API_BASE=https://your-backend.com
   ```
7. Deploy

### Via CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
cd frontend
vercel --prod
```

---

## Option 5: Railway

**Why**: Simple CLI, good for fullstack apps, automatic HTTPS

### Steps

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
cd frontend
railway init

# Deploy
railway up

# Set environment variables
railway variables set VITE_API_BASE=https://your-backend.railway.app

# View logs
railway logs

# Open in browser
railway open
```

---

## Option 6: Fly.io

**Why**: Edge deployment, good performance, global distribution

### Steps

```bash
# Install flyctl
# Windows: iwr https://fly.io/install.ps1 -useb | iex
# Mac/Linux: curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Initialize
cd frontend
fly launch --dockerfile Dockerfile

# Deploy
fly deploy

# Set secrets
fly secrets set VITE_API_BASE=https://your-backend.fly.dev

# Check status
fly status

# View logs
fly logs

# Open in browser
fly open
```

---

## Local Docker Testing

### Build

```bash
cd frontend
docker build -t reviewer-recommender-frontend .
```

### Run

```bash
docker run -p 3000:80 reviewer-recommender-frontend
```

### Test

```bash
# Health check
curl http://localhost:3000/health

# Main page
curl http://localhost:3000/

# PowerShell
Invoke-WebRequest -Uri http://localhost:3000/health
```

### Stop

```bash
docker ps  # Get container ID
docker stop <container_id>
```

---

## Environment Variables

### Build-Time Variables (Vite)

**Important**: Vite environment variables are injected at **build time**, not runtime.

**Set in Dockerfile**:
```dockerfile
# Add build arg
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE
```

**Build with custom API**:
```bash
docker build --build-arg VITE_API_BASE=https://api.example.com \
  -t reviewer-recommender-frontend .
```

**Set in Static Site** (Render, Netlify, Vercel):
- Go to environment variables section
- Add: `VITE_API_BASE=https://your-backend.com`
- Redeploy

### Accessing in Code

```javascript
// In React components
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

// Make API calls
axios.post(`${API_BASE}/recommend`, data);
```

---

## Troubleshooting

### "npm ci" Fails

```bash
# Delete package-lock.json and node_modules
rm -rf node_modules package-lock.json

# Regenerate
npm install

# Try again
npm ci
```

### Build Succeeds But dist/ Is Empty

```bash
# Check build command in package.json
cat package.json | grep "build"

# Should be: "build": "vite build"

# Test locally
npm run build
ls -la dist/
```

### API Calls Don't Work

**Check CORS**:
```python
# backend/app.py
FRONTEND_ORIGIN = "https://your-frontend.onrender.com"
```

**Check Environment Variable**:
```bash
# In frontend/.env
VITE_API_BASE=https://your-backend.onrender.com
```

**Check Browser Console**:
- F12 → Console
- Look for CORS errors
- Look for 404 errors

### SPA Routing Returns 404

**For Docker**:
- Verify nginx.conf has `try_files $uri $uri/ /index.html;`
- Rebuild image

**For Static Sites**:
- Render/Netlify/Vercel handle this automatically
- Check their documentation for SPA configuration

---

## Performance Tips

### 1. Enable Gzip Compression

Already enabled in nginx.conf:
```nginx
gzip on;
gzip_comp_level 6;
```

### 2. Optimize Images

```bash
# Use WebP format
# Compress images before adding to repo
# Use lazy loading for images
```

### 3. Code Splitting

Vite automatically splits code. To verify:
```bash
npm run build
ls -la dist/assets/

# Should see multiple JS chunks
```

### 4. CDN

- Render Static Sites: CDN included
- Netlify/Vercel: CDN included
- Docker: Use Cloudflare or similar

### 5. Cache Headers

Already configured in nginx.conf:
- Static assets: 1 year cache
- index.html: No cache

---

## Cost Comparison

| Platform | Free Tier | Paid Plans | Notes |
|----------|-----------|------------|-------|
| **Render Static Site** | ✅ Yes | $0/mo | CDN included |
| **Render Docker** | ✅ Yes (limited) | $7/mo+ | More control |
| **Netlify** | ✅ Yes (100GB/mo) | $19/mo+ | Generous free tier |
| **Vercel** | ✅ Yes | $20/mo+ | Best for React |
| **Railway** | ✅ $5 credit/mo | Pay as you go | Good for fullstack |
| **Fly.io** | ✅ $5 credit/mo | Pay as you go | Edge deployment |

**Recommendation**: Start with **Render Static Site** (free, CDN, automatic)

---

## Deployment Checklist

### Pre-Deployment

- [ ] Test build locally: `npm run build`
- [ ] Check dist/ directory exists
- [ ] Verify .env has VITE_API_BASE
- [ ] Test Docker build (if using Docker)
- [ ] Update backend CORS with frontend URL

### During Deployment

- [ ] Push to GitHub
- [ ] Connect to deployment platform
- [ ] Set environment variables
- [ ] Configure build settings
- [ ] Deploy

### Post-Deployment

- [ ] Test health endpoint: `/health`
- [ ] Test main page loads
- [ ] Test file upload works
- [ ] Test paste abstract works
- [ ] Test recommendations display
- [ ] Test SPA routing (direct URLs)
- [ ] Check browser console for errors
- [ ] Update backend CORS if needed

---

## Quick Commands Reference

```bash
# Local development
npm run dev                     # Start dev server

# Build
npm run build                   # Build for production
npm run preview                 # Preview production build

# Docker
docker build -t app .           # Build image
docker run -p 3000:80 app       # Run container
docker ps                       # List containers
docker logs <id>                # View logs
docker stop <id>                # Stop container

# Deployment
git push origin main            # Deploy (if auto-deploy enabled)
netlify deploy --prod           # Netlify
vercel --prod                   # Vercel
railway up                      # Railway
fly deploy                      # Fly.io
```

---

## Support Resources

- **Render Docs**: https://render.com/docs/static-sites
- **Netlify Docs**: https://docs.netlify.com/
- **Vercel Docs**: https://vercel.com/docs
- **Railway Docs**: https://docs.railway.app/
- **Fly.io Docs**: https://fly.io/docs/
- **Vite Docs**: https://vitejs.dev/guide/build.html
- **Nginx Docs**: https://nginx.org/en/docs/

---

**Need help?** Check PROMPT18_COMPLETION.md for detailed troubleshooting.

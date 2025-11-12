# Quick Render Deployment Guide

## Step-by-Step: Deploy Backend to Render

### 1. Push Code to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Prepare for Render deployment"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy on Render

1. **Go to Render**: https://render.com
2. **Sign up** with GitHub (recommended)
3. **New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your repo

4. **Configure**:
   - **Name**: `phishing-detection-api`
   - **Environment**: `Docker`
   - **Region**: Choose closest to users
   - **Branch**: `main`
   - **Dockerfile Path**: `./Dockerfile`

5. **Environment Variables**:
   ```
   PORT=8000
   PYTHONUNBUFFERED=1
   ALLOWED_ORIGINS=*
   ```

6. **Plan**: 
   - Free tier: 512MB (may not be enough for ML models)
   - Starter: $7/month (512MB, always on)
   - Standard: $25/month (2GB, always on) ⭐ Recommended

7. **Deploy**: Click "Create Web Service"
   - First build takes 10-15 minutes
   - Watch the logs for progress

### 3. Get Your API URL

After deployment, your API will be at:
```
https://YOUR_SERVICE_NAME.onrender.com
```

Test it:
```bash
curl https://YOUR_SERVICE_NAME.onrender.com/health
```

### 4. Update Extension

In `background.js`, line 96, update:
```javascript
this.apiBaseUrl = 'https://YOUR_SERVICE_NAME.onrender.com';
```

### 5. Important Notes

- **Free Tier**: Spins down after 15 min inactivity (cold start ~30 seconds)
- **Memory**: ML models need at least 512MB, preferably 2GB
- **Model Files**: Ensure all `.pickle.dat`, `.json`, `.bin` files are in repo
- **First Request**: May be slow (downloading BERT model from HuggingFace)

### Troubleshooting

**Build fails?**
- Check Render logs
- Ensure `requirements.txt` has all dependencies
- Verify Dockerfile is correct

**Out of memory?**
- Upgrade to Starter or Standard plan
- Reduce model size if possible

**Slow responses?**
- Free tier has cold starts
- Upgrade to paid plan for always-on

**CORS errors?**
- Check `ALLOWED_ORIGINS` environment variable
- Verify CORS settings in `backend.py`

---

## Quick Checklist

- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] Web service created and configured
- [ ] Environment variables set
- [ ] Service deployed successfully
- [ ] API URL obtained
- [ ] Extension updated with API URL
- [ ] Tested extension with Render API

Done! Your backend is now live on Render! 🎉


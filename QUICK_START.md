# Quick Start: Deploy & Publish

## 🚀 Fast Track Deployment

### Part 1: Deploy Backend (15 minutes)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push
   ```

2. **Deploy to Render**
   - Go to https://render.com
   - Sign up with GitHub
   - New → Web Service
   - Connect repo → Select Docker
   - Name: `phishing-detection-api`
   - Add env vars:
     - `PORT=8000`
     - `PYTHONUNBUFFERED=1`
     - `ALLOWED_ORIGINS=*`
   - Choose plan (Starter $7/month recommended)
   - Deploy!

3. **Get Your URL**
   - Wait for deployment (10-15 min)
   - Your API: `https://YOUR_NAME.onrender.com`

4. **Update Extension**
   - Edit `background.js` line 96
   - Replace URL with your Render URL

### Part 2: Publish Extension (30 minutes)

1. **Prepare Package**
   - Create folder with only extension files
   - Remove: `.py`, `.md`, test files, model files
   - Keep: `manifest.json`, `.js`, `.html`, `icons/`
   - ZIP it: `phishing-detector-v1.0.0.zip`

2. **Chrome Web Store**
   - Go to https://chrome.google.com/webstore/devconsole
   - Pay $5 registration
   - New Item → Upload ZIP
   - Fill store listing (see CHROME_STORE_GUIDE.md)
   - Add privacy policy URL
   - Submit for review

3. **Privacy Policy**
   - Use `privacy-policy.html` template
   - Host on GitHub Pages (free)
   - Add URL to Chrome Web Store listing

## ✅ Checklist

**Backend:**
- [ ] Code on GitHub
- [ ] Render service created
- [ ] Environment variables set
- [ ] Deployed successfully
- [ ] API URL obtained
- [ ] Extension updated with API URL

**Extension:**
- [ ] Icons included ✅
- [ ] Manifest updated ✅
- [ ] Package cleaned (no test files)
- [ ] ZIP created
- [ ] Privacy policy created and hosted

**Chrome Web Store:**
- [ ] Developer account ($5 paid)
- [ ] Extension uploaded
- [ ] Store listing filled
- [ ] Screenshots added
- [ ] Privacy policy URL added
- [ ] Submitted for review

## 📚 Detailed Guides

- **Full Deployment**: See `DEPLOYMENT_GUIDE.md`
- **Render Specific**: See `RENDER_DEPLOYMENT.md`
- **Chrome Store**: See `CHROME_STORE_GUIDE.md`

## 🆘 Need Help?

- **Render Issues**: Check Render logs, upgrade plan if out of memory
- **Store Rejection**: Check permissions justification, privacy policy
- **Extension Not Working**: Verify API URL, check browser console

Good luck! 🎉


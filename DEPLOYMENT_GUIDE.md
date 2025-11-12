# Deployment Guide: Phishing Website Detector

This guide will walk you through deploying the backend API to Render and publishing the Chrome extension to the Chrome Web Store.

## Prerequisites

- GitHub account
- Render account (free tier available)
- Chrome Web Store Developer account ($5 one-time fee)
- Git installed on your computer

---

## Part 1: Deploy Backend API to Render

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** (if you haven't already)
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Phishing Detection Extension"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Ensure these files are in your repository:**
   - `Dockerfile`
   - `.dockerignore`
   - `backend.py`
   - `requirements.txt`
   - `render.yaml` (optional but recommended)
   - All model files (`.pickle.dat`, `.json`, `.bin`)

### Step 2: Deploy to Render

1. **Sign up/Login to Render**
   - Go to https://render.com
   - Sign up with your GitHub account (recommended)

2. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your backend code

3. **Configure the Service**
   - **Name**: `phishing-detection-api` (or your preferred name)
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or specify if backend is in a subfolder)
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Context**: `.` (current directory)

4. **Set Environment Variables**
   - Click "Advanced" → "Environment Variables"
   - Add:
     ```
     PORT=8000
     PYTHONUNBUFFERED=1
     ALLOWED_ORIGINS=*
     ```

5. **Choose a Plan**
   - **Free Tier**: 512MB RAM, spins down after 15 min inactivity
   - **Starter Plan ($7/month)**: 512MB RAM, always on
   - **Standard Plan ($25/month)**: 2GB RAM, always on (recommended for ML models)

   **Note**: For ML models with PyTorch/Transformers, the free tier may not have enough memory. Consider the Starter or Standard plan.

6. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your Docker container
   - This may take 10-15 minutes for the first build (downloading dependencies)

7. **Get Your API URL**
   - Once deployed, your API will be available at:
     `https://YOUR_SERVICE_NAME.onrender.com`
   - Example: `https://phishing-detection-api.onrender.com`

8. **Test Your API**
   ```bash
   # Health check
   curl https://YOUR_SERVICE_NAME.onrender.com/health
   
   # Test prediction
   curl -X POST https://YOUR_SERVICE_NAME.onrender.com/predict \
        -H "Content-Type: application/json" \
        -d '{"url": "https://www.google.com"}'
   ```

### Step 3: Update Extension with Render URL

1. **Update `background.js`**
   - Open `background.js`
   - Find line 96: `this.apiBaseUrl = 'https://phishing-detection-api.onrender.com';`
   - Replace with your actual Render URL:
     ```javascript
     this.apiBaseUrl = 'https://YOUR_SERVICE_NAME.onrender.com';
     ```

2. **Test the Extension Locally**
   - Load the extension in Chrome (chrome://extensions → Developer mode → Load unpacked)
   - Test that it connects to your Render API
   - Check the browser console for any errors

---

## Part 2: Publish to Chrome Web Store

### Step 1: Prepare Extension Package

1. **Create a clean package folder**
   ```bash
   mkdir extension-package
   ```

2. **Copy only necessary files:**
   ```
   extension-package/
   ├── manifest.json
   ├── background.js
   ├── content.js
   ├── popup.html
   ├── popup.js
   ├── warning.html
   ├── warning.js
   ├── featureExtractor.js
   ├── icons/
   │   ├── icon16.png
   │   ├── icon48.png
   │   └── icon128.png
   └── (any other required files)
   ```

3. **Remove unnecessary files:**
   - All Python files (`.py`)
   - Test files (`test_*.py`, `test.html`)
   - Documentation (`.md` files - except README if needed)
   - Batch/shell scripts (`.bat`, `.sh`)
   - Model files (they're on Render now)
   - CSV data files
   - Jupyter notebooks

4. **Create ZIP file**
   - Select all files in `extension-package` folder
   - Right-click → "Send to" → "Compressed (zipped) folder"
   - Name it: `phishing-detector-v1.0.0.zip`

### Step 2: Create Chrome Web Store Developer Account

1. **Go to Chrome Web Store Developer Dashboard**
   - Visit: https://chrome.google.com/webstore/devconsole
   - Sign in with your Google account

2. **Pay Registration Fee**
   - One-time payment of $5 USD
   - Required to publish extensions

3. **Complete Registration**
   - Accept the Developer Agreement
   - Complete your developer profile

### Step 3: Upload Extension

1. **Create New Item**
   - Click "New Item" button
   - Click "Choose File" and select your ZIP file
   - Click "Upload"

2. **Fill Store Listing**

   **Required Fields:**
   - **Name**: `Phishing Website Detector`
   - **Summary** (132 chars max): 
     ```
     ML-powered phishing detection. Real-time URL analysis with XGBoost and BERT models to protect you from malicious websites.
     ```
   - **Description**: 
     ```
     Phishing Website Detector is a powerful Chrome extension that uses advanced machine learning to protect you from phishing attacks.

     Features:
     • Real-time URL analysis using XGBoost machine learning model
     • Deep dive analysis with BERT transformer model
     • Feature importance explanations (SHAP values)
     • Automatic blocking of suspicious websites
     • Whitelist support for trusted sites
     • Statistics tracking
     • Beautiful, intuitive interface

     How it works:
     The extension analyzes URLs in real-time as you browse, extracting 16 different features including URL structure, domain information, and HTML content. It uses a trained XGBoost model (86.4% accuracy) to predict phishing probability and can perform deep content analysis using BERT for more sophisticated detection.

     Privacy:
     • All analysis happens server-side
     • No personal data is collected
     • URLs are only sent to our secure API for analysis
     • Whitelist and settings stored locally in your browser

     Perfect for:
     • Personal browsing protection
     • Business users
     • Anyone concerned about online security
     ```
   - **Category**: `Productivity` or `Security`
   - **Language**: Select your primary language

   **Optional but Recommended:**
   - **Detailed Description**: Expand on features, use cases, technical details
   - **Screenshots**: 
     - Take 1-5 screenshots (1280x800 or 640x400 recommended)
     - Show: popup interface, warning page, settings, analysis results
   - **Promotional Images** (if you have them):
     - Small tile: 440x280
     - Large tile: 920x680
     - Marquee: 1400x560

3. **Privacy Practices**
   - **Single Purpose**: Yes (phishing detection)
   - **Hosted on Chrome Web Store**: Yes
   - **Privacy Policy URL**: 
     - Create a privacy policy (see below)
     - Host it on GitHub Pages, Google Sites, or your website
     - Enter the URL here
   - **Data Handling**:
     - Specify: "URLs are sent to our API for analysis"
     - Specify: "No personal information is collected"
     - Specify: "Settings stored locally in browser"

4. **Distribution**
   - **Visibility**: 
     - `Public` - Anyone can find and install
     - `Unlisted` - Only people with the link can install (for testing)
   - **Countries/Regions**: All or specific
   - **Pricing**: Free

### Step 4: Create Privacy Policy

Create a simple privacy policy page. Here's a template:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Privacy Policy - Phishing Website Detector</title>
</head>
<body>
    <h1>Privacy Policy for Phishing Website Detector</h1>
    <p><strong>Last Updated:</strong> [Date]</p>
    
    <h2>Data Collection</h2>
    <p>Phishing Website Detector collects minimal data:</p>
    <ul>
        <li><strong>URLs:</strong> URLs are sent to our secure API for phishing analysis</li>
        <li><strong>No Personal Data:</strong> We do not collect personal information, browsing history, or any identifiable data</li>
    </ul>
    
    <h2>Data Usage</h2>
    <p>URLs are analyzed using machine learning models to detect phishing attempts. Analysis results are not stored.</p>
    
    <h2>Data Storage</h2>
    <p>All extension settings and whitelist are stored locally in your browser. No data is stored on our servers.</p>
    
    <h2>Third-Party Services</h2>
    <p>We use Render.com to host our API. URLs are sent to this service for analysis only.</p>
    
    <h2>Contact</h2>
    <p>For questions about this privacy policy, contact: [Your Email]</p>
</body>
</html>
```

Host this on:
- GitHub Pages (free)
- Google Sites (free)
- Your own website

### Step 5: Submit for Review

1. **Review Checklist**
   - [ ] Extension works with Render API
   - [ ] All permissions justified
   - [ ] Privacy policy URL provided
   - [ ] Icons included
   - [ ] Description complete
   - [ ] Screenshots provided
   - [ ] No test/debug code
   - [ ] Version number correct

2. **Submit**
   - Click "Submit for Review"
   - Review typically takes 1-7 business days
   - You'll receive email updates

3. **After Approval**
   - Extension goes live
   - Share your Chrome Web Store link
   - Monitor reviews and feedback

---

## Troubleshooting

### Render Deployment Issues

**Problem**: Build fails
- **Solution**: Check Render logs, ensure all dependencies are in `requirements.txt`

**Problem**: Out of memory errors
- **Solution**: Upgrade to Starter ($7/month) or Standard ($25/month) plan

**Problem**: Slow cold starts (free tier)
- **Solution**: Upgrade to paid plan for always-on service

### Chrome Web Store Issues

**Problem**: Extension rejected for permissions
- **Solution**: Justify each permission in the description

**Problem**: Extension rejected for privacy
- **Solution**: Ensure privacy policy is accessible and complete

**Problem**: Extension doesn't work after publishing
- **Solution**: Verify Render API URL is correct and accessible

---

## Post-Deployment

### Monitor Your Service

1. **Render Dashboard**
   - Monitor API health
   - Check logs for errors
   - Monitor resource usage

2. **Chrome Web Store**
   - Respond to user reviews
   - Monitor crash reports
   - Update extension as needed

### Update Extension

When updating:
1. Update version in `manifest.json`
2. Create new ZIP package
3. Upload to Chrome Web Store
4. Submit for review

---

## Support

For issues or questions:
- Check Render documentation: https://render.com/docs
- Check Chrome Web Store policies: https://developer.chrome.com/docs/webstore/program-policies/
- Review extension logs in Chrome DevTools

Good luck with your deployment! 🚀


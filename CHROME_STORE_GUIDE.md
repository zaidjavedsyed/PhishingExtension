# Chrome Web Store Publishing Guide

## Quick Checklist

### Before Publishing

- [ ] Backend deployed to Render and working
- [ ] Extension tested with Render API
- [ ] Icons created (16x16, 48x48, 128x128)
- [ ] Manifest.json updated with icons
- [ ] Version number set correctly
- [ ] All test files removed from package
- [ ] Privacy policy created and hosted

### Required Assets

1. **Icons** ✅ (You have these)
   - icon16.png
   - icon48.png
   - icon128.png

2. **Screenshots** (1-5 images)
   - Recommended size: 1280x800 or 640x400
   - Show: popup, warning page, settings

3. **Privacy Policy** (Required)
   - Host on GitHub Pages, Google Sites, or your website
   - See template in DEPLOYMENT_GUIDE.md

## Step-by-Step Publishing

### Step 1: Prepare Package

1. Create clean folder with only extension files:
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
   └── icons/
       ├── icon16.png
       ├── icon48.png
       └── icon128.png
   ```

2. Remove:
   - All `.py` files
   - All `.md` files (except README if needed)
   - Test files (`test_*.py`, `test.html`)
   - Batch files (`.bat`, `.sh`)
   - Model files (on Render now)
   - Documentation files

3. Create ZIP:
   - Select all files in extension-package
   - Right-click → "Send to" → "Compressed (zipped) folder"
   - Name: `phishing-detector-v1.0.0.zip`

### Step 2: Chrome Web Store Account

1. Go to: https://chrome.google.com/webstore/devconsole
2. Sign in with Google account
3. Pay $5 registration fee (one-time)
4. Accept Developer Agreement

### Step 3: Upload Extension

1. Click "New Item"
2. Upload your ZIP file
3. Fill in store listing (see below)

### Step 4: Store Listing

**Name**: `Phishing Website Detector`

**Summary** (132 chars max):
```
ML-powered phishing detection. Real-time URL analysis with XGBoost and BERT models to protect you from malicious websites.
```

**Description**:
```
Phishing Website Detector uses advanced machine learning to protect you from phishing attacks.

Features:
• Real-time URL analysis using XGBoost (86.4% accuracy)
• Deep dive analysis with BERT transformer
• Feature importance explanations (SHAP)
• Automatic blocking of suspicious sites
• Whitelist support
• Statistics tracking
• Beautiful interface

Privacy: URLs analyzed server-side. No personal data collected. Settings stored locally.
```

**Category**: `Productivity` or `Security`

**Language**: Your primary language

**Screenshots**: Upload 1-5 screenshots showing:
- Extension popup
- Warning page
- Settings panel
- Analysis results

### Step 5: Privacy & Permissions

**Privacy Policy URL**: 
- Create privacy policy (see DEPLOYMENT_GUIDE.md)
- Host on GitHub Pages or Google Sites
- Enter URL here

**Permissions Justification**:
- `<all_urls>`: Required to analyze URLs on all websites
- `activeTab`: Required to access current page URL
- `storage`: Required to store whitelist and settings locally

### Step 6: Distribution

- **Visibility**: Public (or Unlisted for testing)
- **Pricing**: Free
- **Countries**: All regions

### Step 7: Submit

1. Review all information
2. Click "Submit for Review"
3. Wait 1-7 business days
4. Receive email when approved/rejected

## Common Rejection Reasons

1. **Missing Privacy Policy** - Must provide URL
2. **Permissions Not Justified** - Explain each permission
3. **Extension Doesn't Work** - Test thoroughly before submitting
4. **Poor Description** - Be clear and detailed
5. **Missing Icons** - Ensure all sizes included

## After Approval

- Extension goes live automatically
- Share your Chrome Web Store link
- Monitor reviews and respond to feedback
- Update as needed (increment version number)

## Updating Your Extension

1. Update version in `manifest.json` (e.g., 1.0.0 → 1.0.1)
2. Create new ZIP package
3. Go to Chrome Web Store dashboard
4. Click on your extension
5. Upload new package
6. Submit for review

Good luck! 🚀


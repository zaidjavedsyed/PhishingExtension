# Demo URLs for Your Presentation

## 🎯 Test URLs (Safe for Demo - Won't Redirect or Harm)

### For Testing Phishing Detection (Should Trigger Warnings):

1. **Long URL with Suspicious Patterns**
   ```
   http://secure-bank-verify-update-account-info-urgent-action-required-now.com
   ```

2. **URL with IP Address**
   ```
   http://192.168.1.100/login
   ```

3. **URL with @ Symbol**
   ```
   http://example@login.com
   ```

4. **URL with Hyphens (Suspicious)**
   ```
   http://secure-paypal-verification-center.com
   ```

5. **URL with TinyURL pattern**
   ```
   http://bit.ly/abcdef123
   ```

6. **Deep Nested Path**
   ```
   http://example.com/verify/update/account/login/secure/action
   ```

### For Testing Safe Sites (Should NOT Trigger):

1. **Google**
   ```
   https://www.google.com
   ```

2. **GitHub**
   ```
   https://www.github.com
   ```

3. **Wikipedia**
   ```
   https://www.wikipedia.org
   ```

4. **Microsoft**
   ```
   https://www.microsoft.com
   ```

## 🧪 How to Test During Demo:

### 1. Safe Site Test (15 seconds):
```
1. Open extension popup
2. Navigate to https://www.google.com
3. Shows: "✅ Safe Website"
4. Click "Why This Prediction?"
5. Shows: Top contributing features (likely protective features)
```

### 2. Phishing Site Test (20 seconds):
```
1. Try to navigate to suspicious URL
2. Shows: Blocked or warning page
3. Click "Why This Prediction?"
4. Shows: "🚨 UNSAFE - Top contributing factors: URL_Length, URL_Depth"
```

### 3. Deep Dive Test (25 seconds):
```
1. Navigate to suspicious URL
2. Click "🔬 Deep Dive Analysis"  
3. Shows: BERT analysis results
4. Shows: Content analysis and confidence scores
```

## 🎤 Demo Script for URL: `http://secure-paypal-verification-center.com`

### Step 1: Navigate to URL
```
"This URL looks suspicious - let me show you why our system detected it."
```

### Step 2: Click "Why This Prediction?"
```
Expected output:
"🚨 UNSAFE — Our detection system shows high phishing risk (92%). 
Top contributing factors: URL_Length, Prefix/Suffix, Domain_Age"

📊 Feature Analysis:
🔴 URL_Length increases risk by +6.91
🔴 Prefix/Suffix increases risk by +3.24  
🔴 Domain_Age increases risk by +2.15
🟢 DNS_Record reduces risk by -1.50
🔴 TinyURL increases risk by +1.23
```

### Step 3: Explain What This Means
```
"Notice how SHAP tells us WHY it's phishing:
- The URL is too long (security red flag)
- Has hyphens in the domain name (suspicious pattern)
- The domain is brand new (phishing sites are often short-lived)

This is explainable AI in action - users understand the reasoning."
```

## 📋 Quick Tips for Demo:

1. **Start with a safe site** (Google) to show it works correctly
2. **Then test suspicious URL** to show detection
3. **Click "Why This Prediction?"** to demonstrate SHAP
4. **Optional: Deep Dive** to show BERT content analysis
5. **Emphasize**: "Users see WHY it's flagged, not just that it's flagged"

## ⚠️ Important Notes:

- All test URLs are simulation patterns only
- Don't visit actual malicious sites
- If testing with real malicious sites, use a VM/sandbox
- Your extension works with the backend API on localhost:8000

## 🚀 Quick Setup for Demo:

```bash
# 1. Start backend
python backend.py

# 2. Load extension in Chrome
chrome://extensions/ → Load unpacked → Select your folder

# 3. Test URLs listed above
# Open in browser and observe extension behavior
```

Good luck with your presentation! 🎓




# Phishing Detection System - Presentation Summary

## Project Overview
**Goal**: Build a Chrome extension that uses machine learning to detect phishing websites with explainability and advanced analysis capabilities.

---

## 1. THE CORE SYSTEM: XGBoost Model

### What is XGBoost?
- **Gradient boosting algorithm** - iterative ensemble method
- **86.4% accuracy** on 10,000 URLs (5,000 legitimate + 5,000 phishing)
- Fast predictions perfect for real-time detection

### 16 Features Extracted from URLs:

#### **Address Bar Features (8 features):**
1. **Have_IP** - Is the domain an IP address instead of a domain name?
2. **Have_At** - Does the URL contain an '@' symbol? (suspicious!)
3. **URL_Length** - Is the URL longer than 75 characters?
4. **URL_Depth** - How many nested directories in the path?
5. **Redirection** - Does the URL contain multiple '//' patterns?
6. **https_Domain** - Is 'https' embedded in the domain name?
7. **TinyURL** - Is this a URL shortening service (bit.ly, tinyurl, etc.)?
8. **Prefix/Suffix** - Does the domain contain hyphens (e.g., "secure-paypal.com")?

#### **Domain-Based Features (4 features):**
9. **DNS_Record** - Does the domain have valid DNS records?
10. **Web_Traffic** - Is this a high-traffic, well-known domain?
11. **Domain_Age** - How old is the domain? (new domains suspicious)
12. **Domain_End** - When does the domain expire? (short expiration suspicious)

#### **HTML & JavaScript Features (4 features):**
13. **iFrame** - Does the page use suspicious iframe redirects?
14. **Mouse_Over** - Is right-click or mouseover disabled?
15. **Right_Click** - Are text selection blocked?
16. **Web_Forwards** - Does the page have excessive redirects?

### Training Process:
```python
# Simplified version of training
X = extract_features_from_urls(10,000_urls)  # 16 features x 10,000 samples
y = labels  # 1 = phishing, 0 = legitimate

model = XGBoostClassifier()
model.fit(X, y)
model.save('BalancedXGBoostModel.pickle.dat')
```

### Feature Importance (What the Model Learned):
- **DNS_Record**: 15% - Most important!
- **Have_IP**: 15% - IP addresses are very suspicious
- **URL_Length**: 12% - Long URLs hide phishing attempts
- **TinyURL**: 12% - URL shorteners hide destination
- **Domain_Age**: 12% - New domains are suspicious
- **Other features**: 34%

---

## 2. SHAP IMPLEMENTATION (Explainability)

### What is SHAP?
**SHapley Additive exPlanations** - A game theory-based method to explain ML predictions
- Shows **WHY** the model made a prediction
- Quantifies **how much each feature contributed** to the final decision
- Makes black-box models transparent and trustworthy

### How We Implemented SHAP:

#### **Backend Implementation** (`backend.py`):

```python
@app.post("/explain")
async def explain_prediction(request: URLRequest):
    # 1. Extract features from URL
    features = feature_extractor.featureExtraction(url)
    
    # 2. Make prediction
    prediction = model.predict_proba(features)
    
    # 3. Create SHAP explainer (TreeExplainer works with XGBoost)
    explainer = shap.TreeExplainer(model)
    
    # 4. Calculate SHAP values (feature contributions)
    shap_values = explainer.shap_values(features)
    
    # 5. For each feature, calculate:
    for feature in features:
        feature_importance.append({
            "feature": name,
            "value": feature_value,
            "importance": shap_contribution,  # How much it contributed
            "contribution": importance * value  # Total impact
        })
    
    # 6. Return sorted by importance
    return ExplainabilityResponse(
        feature_importance=sorted_by_importance,
        summary=human_readable_summary
    )
```

#### **Frontend Integration** (popup.js):

1. **Button Click** → User clicks "Why This Prediction?" button
2. **API Call** → Calls `/explain` endpoint via background script
3. **Display Results** → Shows top 5 features with:
   - 🔴 Red: Features that INCREASE phishing risk
   - 🟢 Green: Features that DECREASE phishing risk (protect the user)

```javascript
async performExplainabilityAnalysis() {
    // Send URL to backend
    const response = await chrome.runtime.sendMessage({
        action: 'explainPrediction',
        url: current_url
    });
    
    // Display results
    updateExplainabilityUI(response.data);
    
    // Show: "URL_Length increases risk by 6.91"
    //       "Domain_End decreases risk by 0.51"
}
```

### Example SHAP Output:
```
URL: https://secure-paypal-verification-center.com

Top Contributing Features (SHAP Analysis):
🔴 URL_Length (value: 1) increases phishing risk by 6.91
🔴 URL_Depth (value: 1) increases phishing risk by 0.75
🔴 TinyURL (value: 1) increases phishing risk by 0.62
🟢 Domain_End (value: 0) decreases phishing risk by 0.51
🟢 DNS_Record (value: 0) decreases phishing risk by 0.42

Summary: Model predicts PHISHING (92% confidence)
         because URL is very long, deeply nested, and uses suspicious patterns.
```

### Why SHAP is Important:
1. **Transparency**: Users can see WHY the system flagged something
2. **Trust**: Makes AI decisions understandable
3. **Debugging**: Helps identify feature extraction issues
4. **Research**: Validates which features are most predictive

---

## 3. BERT TRANSFORMER MODEL (Deep Dive Analysis)

### What is BERT?
**Bidirectional Encoder Representations from Transformers**
- Pre-trained on millions of text examples
- Understands context, meaning, and linguistic patterns
- Can detect subtle phishing language in page content

### How We Integrated BERT:

#### **Backend Implementation** (`backend.py`):

```python
# Load pre-trained BERT model from Hugging Face
model_name = "ealvaradob/bert-finetuned-phishing"
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/deep-dive")
async def deep_dive_analysis(url):
    # 1. Fetch HTML content from URL
    response = requests.get(url)
    html_content = response.text
    
    # 2. Extract text content (remove scripts, styles)
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()  # Clean text only
    
    # 3. Prepare input for BERT
    analysis_text = f"URL: {url}\nContent: {text_content}"
    
    # 4. Tokenize (convert text to numbers)
    inputs = bert_tokenizer(
        analysis_text, 
        return_tensors="pt", 
        max_length=512,
        truncation=True
    )
    
    # 5. Run BERT inference
    with torch.no_grad():
        outputs = bert_model(**inputs)
        predictions = softmax(outputs.logits)
    
    # 6. Extract results
    phishing_score = predictions[0][1]  # Probability of phishing
    legitimate_score = predictions[0][0]  # Probability of legitimate
    
    return {
        "is_phishing": phishing_score > 0.5,
        "confidence": max(phishing_score, legitimate_score),
        "bert_scores": {
            "phishing": phishing_score,
            "legitimate": legitimate_score
        },
        "text_content": text_content[:500]
    }
```

#### **Frontend Integration** (popup.js):

```javascript
async performDeepDiveAnalysis() {
    // Show loading state
    button.textContent = '🔬 Analyzing...';
    
    // Call deep dive API
    const response = await chrome.runtime.sendMessage({
        action: 'deepDiveAnalysis',
        url: current_url
    });
    
    // Display BERT results
    updateDeepDiveUI(response.result);
    
    // Show:
    // - BERT phishing score: 95%
    // - BERT legitimate score: 5%
    // - Analyzed content preview
}
```

### Why BERT is Important:
1. **Content Analysis**: Analyzes actual page content, not just URL features
2. **Linguistic Patterns**: Detects phishing language (e.g., "Verify your account", "Urgent action required")
3. **Higher Accuracy**: More sophisticated than URL-only analysis
4. **Complementary**: Works alongside XGBoost for comprehensive protection

### Example BERT Output:
```
URL: https://secure-paypal-verification-center.com

BERT Analysis Results:
🔬 Deep Dive Analysis (BERT Transformer)
🚨 Phishing Detected (95% confidence)

BERT Scores:
• Phishing Score: 94.2%
• Legitimate Score: 5.8%
• Model: BERT (ealvaradob/bert-finetuned-phishing)

Analyzed Content:
"URGENT: Your PayPal account has been temporarily suspended.
 Please click here to verify your identity immediately. 
 Failure to respond within 24 hours will result in account closure."

Analysis: Page contains urgent language, identity verification request,
and consequences - classic phishing patterns detected.
```

---

## 4. SYSTEM INTEGRATION (How Everything Works Together)

### Architecture Overview:

```
User Browses Website
        ↓
Chrome Extension (background.js)
        ↓
    ┌───────────────────────┐
    │  XGBoost (Primary)    │  ← Fast, real-time detection
    │  16 URL features      │
    │  86.4% accuracy       │
    └───────────────────────┘
            ↓
    ┌───────────────────────┐
    │  Block or Allow?      │
    │  (Check threshold)    │
    └───────────────────────┘
            ↓
    ┌───────────────────────┐
    │  User Wants Details?   │
    └───────────────────────┘
        ↓              ↓
   SHAP Analysis   BERT Analysis
        ↓              ↓
  📊 Explainability  🔬 Deep Dive
        ↓              ↓
    Display Results in Popup
```

### Detailed Flow:

#### **1. Real-Time Detection (Automatic):**
```
User navigates to URL
    ↓
background.js intercepts tab update
    ↓
Send URL to backend.py /predict endpoint
    ↓
Extract 16 features from URL
    ↓
XGBoost model makes prediction
    ↓
If phishing_score > 0.9:
    → Block URL
    → Redirect to warning.html
Else:
    → Allow navigation
```

#### **2. Explainability (On-Demand):**
```
User clicks "Why This Prediction?" button
    ↓
popup.js sends message to background.js
    ↓
background.js calls /explain endpoint
    ↓
Backend uses SHAP TreeExplainer:
    → Calculate feature contributions
    → Sort by absolute importance
    → Generate human-readable summary
    ↓
Display in popup:
    → Top contributing features
    → How much each feature added/subtracted
    → Overall explanation
```

#### **3. Deep Dive Analysis (On-Demand):**
```
User clicks "🔬 Deep Dive Analysis" button
    ↓
popup.js sends message to background.js
    ↓
background.js calls /deep-dive endpoint
    ↓
Backend:
    → Fetches HTML content from URL
    → Extracts text using BeautifulSoup
    → Concatenates URL + text content
    → Tokenizes with BERT tokenizer
    → Runs BERT model inference
    → Returns phishing/legitimate scores
    ↓
Display in popup:
    → BERT confidence scores
    → Analyzed content preview
    → Analysis summary
```

### API Endpoints:

```
FastAPI Backend (Python)
├── GET  /health              - Check if model loaded
├── GET  /model-info          - Get model details
├── POST /predict             - Main detection (XGBoost)
├── POST /explain             - SHAP explainability
├── POST /deep-dive           - BERT analysis
├── GET  /features/{url}      - Extract features only
└── POST /predict-features    - Predict from features
```

### System Components:

#### **Backend (`backend.py`):**
- Loads XGBoost model (pickle format)
- Loads BERT model (Hugging Face)
- Feature extraction (16 features)
- SHAP explainer initialization
- API endpoints for all operations
- CORS enabled for Chrome extension

#### **Chrome Extension (`background.js`):**
- Intercepts tab updates
- Calls backend API
- Manages whitelist/trusted domains
- Routes messages from popup

#### **Popup (`popup.js`):**
- Displays current URL status
- Shows feature analysis
- Triggers explainability analysis
- Triggers deep dive analysis
- Updates UI with results

---

## 5. KEY INNOVATIONS & RESEARCH VALUE

### Why This System is Special:

1. **Hybrid Approach**: Combines traditional ML (XGBoost) with modern AI (BERT)
2. **Explainability**: SHAP makes predictions transparent
3. **User-Centric**: Gives users control with detailed explanations
4. **Real-Time**: Fast enough for browser integration
5. **Accurate**: 86.4% on URL features + BERT for content analysis

### Contribution to Research:
- Demonstrates **interpretable ML** in cybersecurity
- Shows **ensemble approach** (XGBoost + BERT)
- Validates **SHAP explainability** in phishing detection
- Proves **real-world deployment** feasibility

---

## 6. DEMONSTRATION FOR PRESENTATION

### What to Show:

1. **Live Demo**:
   - Navigate to a phishing URL
   - Show automatic blocking
   - Click "Why This Prediction?" - show SHAP analysis
   - Click "Deep Dive" - show BERT analysis

2. **Explain the Flow**:
   ```
   URL → 16 Features → XGBoost → Prediction
                         ↓
                    SHAP calculates contribution of each feature
                         ↓
                    Display to user: "URL_Length increased risk by 6.91"
   ```

3. **Show the Innovation**:
   - Traditional: "This is phishing" (black box)
   - Our system: "This is phishing because URL is too long, contains @, and uses IP address"
   - Adds transparency and user education

4. **Highlight Research Aspects**:
   - Feature engineering (16 URL features)
   - Model selection (XGBoost vs others)
   - SHAP for explainability
   - BERT for content understanding
   - System integration (Chrome extension + FastAPI)

---

## QUICK SUMMARY FOR YOUR PRESENTATION:

### What We Built:
✅ **Chrome Extension** that detects phishing URLs in real-time
✅ **XGBoost Model** trained on 16 URL features (86.4% accuracy)
✅ **SHAP Integration** for explaining predictions
✅ **BERT Analysis** for deep content analysis
✅ **FastAPI Backend** serving ML models
✅ **User-Friendly UI** with popup and warning pages

### Key Technologies:
- **XGBoost**: Gradient boosting for URL-based detection
- **SHAP**: Explainable AI for transparency
- **BERT**: Transformer for content analysis
- **FastAPI**: Modern Python backend
- **Chrome Extension API**: Browser integration

### Research Contributions:
1. Demonstrates interpretable ML in cybersecurity
2. Combines URL features with content analysis
3. Provides real-world deployment example
4. Validates SHAP for phishing detection

### How to Present:
1. **Demo**: Show live detection → SHAP explanation → BERT analysis
2. **Technical Deep Dive**: Explain XGBoost + SHAP + BERT integration
3. **Research Value**: Interpretability, hybrid approach, real-world deployment
4. **Results**: 86.4% accuracy + explainability + advanced analysis

---

## FILES TO REFERENCE:

- **Model Training**: `Phishing Website Detection_Models & Training.ipynb`
- **Feature Extraction**: `URLFeatureExtraction.py`
- **Backend API**: `backend.py`
- **SHAP Implementation**: Lines 687-788 in `backend.py`
- **BERT Integration**: Lines 547-669 in `backend.py`
- **Frontend Integration**: `popup.js` (lines 158-240)
- **Documentation**: `SHAP_IMPLEMENTATION.md`, `DEEP_DIVE_IMPLEMENTATION.md`

---

**Good luck with your presentation!** 🎓




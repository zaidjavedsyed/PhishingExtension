# Feature Extraction: XGBoost, BERT, and SHAP - Complete Code Explanation

## Overview

This document explains how feature extraction works for each model in our phishing detection system.

---

## 1. XGBOOST FEATURE EXTRACTION (16 URL Features)

### Code Location: `backend.py` lines 206-247

### How It Works:

```python
def featureExtraction(self, url):
    features = []
    
    # STEP 1: Extract 8 URL-based features
    features.append(self.havingIP(url))        # Is domain an IP address?
    features.append(self.haveAtSign(url))      # Does URL contain @ ?
    features.append(self.getLength(url))        # Is URL longer than 75 chars?
    features.append(self.getDepth(url))        # How many nested directories?
    features.append(self.redirection(url))     # Multiple // patterns?
    features.append(self.httpDomain(url))       # "https" in domain name?
    features.append(self.tinyURL(url))          # URL shortening service?
    features.append(self.prefixSuffix(url))     # Hyphens in domain?
    
    # STEP 2: Extract 4 domain-based features
    try:
        domain_name = whois.whois(urlparse(url).netloc)
        dns = 0  # Valid DNS record
    except:
        dns = 1  # Invalid DNS record
    
    features.append(dns)                                      # DNS_Record
    features.append(self.web_traffic(url))                   # Web_Traffic
    features.append(self.domainAge(domain_name) if dns==0 else 1)  # Domain_Age
    features.append(self.domainEnd(domain_name) if dns==0 else 1) # Domain_End
    
    # STEP 3: Extract 4 HTML/JavaScript features
    try:
        response = requests.get(url, timeout=5)
    except:
        response = ""
    
    features.append(self.iframe(response))      # Suspicious iframe?
    features.append(self.mouseOver(response))  # Disabled right-click?
    features.append(self.rightClick(response)) # Mouse over tricks?
    features.append(self.forwarding(response)) # Multiple redirects?
    
    return features  # Returns array of 16 binary/continuous values
```

### Example for URL: `https://secure-paypal-verification-center.com`

```python
features = [
    0,  # Have_IP: 0 (no IP address)
    0,  # Have_At: 0 (no @ symbol)
    1,  # URL_Length: 1 (URL > 75 chars - SUSPICIOUS!)
    0,  # URL_Depth: 0 (not deeply nested)
    0,  # Redirection: 0 (no suspicious redirects)
    0,  # https_Domain: 0 (no https in domain)
    0,  # TinyURL: 0 (not a shortening service)
    1,  # Prefix/Suffix: 1 (has hyphens - SUSPICIOUS!)
    0,  # DNS_Record: 0 (valid DNS)
    1,  # Web_Traffic: 1 (unknown traffic)
    1,  # Domain_Age: 1 (new domain - SUSPICIOUS!)
    1,  # Domain_End: 1 (short expiration - SUSPICIOUS!)
    0,  # iFrame: 0 (normal iframe usage)
    0,  # Mouse_Over: 0 (normal behavior)
    0,  # Right_Click: 0 (normal behavior)
    0   # Web_Forwards: 0 (no excessive redirects)
]
```

### How XGBoost Uses These Features:

```python
# In /predict endpoint (line 427):
@app.post("/predict")
async def predict_phishing(request: URLRequest):
    # 1. Extract 16 features
    features = feature_extractor.featureExtraction(request.url)
    #    Example: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
    
    # 2. Convert to numpy array for model
    features_array = np.array(features).reshape(1, -1)
    #    Shape: (1, 16) - one sample, 16 features
    
    # 3. Make prediction with XGBoost model
    prediction_proba = model.predict_proba(features_array)[0]
    #    Returns: [0.08, 0.92] - [probability_legitimate, probability_phishing]
    
    phishing_probability = prediction_proba[1]  # 0.92 (92% phishing)
    
    # 4. Classify based on threshold
    is_phishing = phishing_probability > 0.9  # True
    
    return {
        "is_phishing": True,
        "confidence": 0.92,
        "probability": 0.92,
        "features": features,  # [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
        "feature_names": feature_names
    }
```

### What XGBoost Does Internally:
1. **Loads trained model** (86.4% accuracy from 10,000 URLs)
2. **Input**: Array of 16 feature values
3. **Process**: Each decision tree in the ensemble evaluates features
4. **Output**: Probability of phishing (0.0 to 1.0)
5. **Classification**: If probability > 0.9 → phishing

---

## 2. BERT FEATURE EXTRACTION (Text Content Analysis)

### Code Location: `backend.py` lines 586-666

### How It Works:

```python
@app.post("/deep-dive")
async def deep_dive_analysis(request: URLRequest):
    # STEP 1: Fetch HTML content from URL
    headers = {
        'User-Agent': 'Mozilla/5.0...'  # Fake browser to avoid blocking
    }
    response = requests.get(request.url, headers=headers, timeout=10)
    html_content = response.text
    
    # STEP 2: Extract text from HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove scripts and styles (not relevant text)
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get clean text content
    text_content = soup.get_text()
    # Example: "URGENT: Your PayPal account is suspended. Click here to verify..."
    
    # Clean up whitespace
    lines = (line.strip() for line in text_content.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text_content = ' '.join(chunk for chunk in chunks if chunk)
    
    # Limit to 2000 chars (BERT max is 512 tokens)
    if len(text_content) > 2000:
        text_content = text_content[:2000] + "..."
    
    # STEP 3: Prepare input for BERT
    analysis_text = f"URL: {request.url}\nContent: {text_content}"
    # Combines URL and content into one text string
    
    # STEP 4: Tokenize text (convert words to numbers)
    inputs = bert_tokenizer(
        analysis_text,
        return_tensors="pt",          # PyTorch tensors
        truncation=True,              # Cut if > 512 tokens
        max_length=512,               # BERT maximum input
        padding=True                  # Pad shorter sequences
    )
    # Example: "URGENT" → 1234, "verify" → 5678, etc.
    
    # STEP 5: Run BERT model
    with torch.no_grad():
        outputs = bert_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # STEP 6: Extract results
    probabilities = predictions[0].numpy()
    phishing_score = float(probabilities[1])      # LABEL_1 is phishing
    legitimate_score = float(probabilities[0])    # LABEL_0 is legitimate
    
    return {
        "is_phishing": phishing_score > 0.5,
        "confidence": max(phishing_score, legitimate_score),
        "bert_scores": {
            "phishing": phishing_score,     # 0.94 (94%)
            "legitimate": legitimate_score    # 0.06 (6%)
        },
        "text_content": text_content
    }
```

### Example for URL: `https://secure-paypal-verification-center.com`

#### **Step 1: Fetch and Parse HTML**
```python
html_content = """
<html>
  <body>
    <h1>URGENT: PayPal Account Suspended</h1>
    <p>Your account has been temporarily suspended.</p>
    <p>Click here to verify your identity.</p>
    <script>...</script>
  </body>
</html>
"""

# After BeautifulSoup extraction:
text_content = "URGENT: PayPal Account Suspended Your account has been temporarily suspended. Click here to verify your identity."
```

#### **Step 2: Prepare for BERT**
```python
analysis_text = """
URL: https://secure-paypal-verification-center.com
Content: URGENT: PayPal Account Suspended Your account has been temporarily suspended. Click here to verify your identity.
"""

# Tokenizer converts to:
# [101, 2535, 2813, 5243, 3054, ...]  ← Token IDs
```

#### **Step 3: BERT Prediction**
```python
# BERT analyzes linguistic patterns:
# - "URGENT" → suspicious language
# - "verify your identity" → phishing pattern
# - "Click here" → suspicious link text
# - Multiple urgency words → phishing attempt

# Model outputs:
probabilities = [0.06, 0.94]  # [legitimate, phishing]
# 94% probability of phishing
```

### What BERT Uses as "Features":
**BERT doesn't use manual features** - it learns features automatically!

1. **Word embeddings**: Each word → vector representation
2. **Attention mechanism**: Finds which words are important
3. **Contextual understanding**: "verify" + "urgent" = suspicious
4. **Pre-training**: Learned from millions of text examples
5. **Fine-tuning**: Specifically trained for phishing detection

### How BERT Works Internally:
```
Input Text: "URGENT: Verify account"
      ↓
[Tokenization]
      ↓
[Word Embeddings] → [Encoder Layers] → [Attention Weights]
      ↓                                            ↓
   "urgent" connects to "verify" (both suspicious)
      ↓
[Classifier Layer]
      ↓
Output: [0.06 legitimate, 0.94 phishing]
```

---

## 3. SHAP FEATURE EXTRACTION (Same as XGBoost + Explainer)

### Code Location: `backend.py` lines 687-788

### How It Works:

```python
@app.post("/explain")
async def explain_prediction(request: URLRequest):
    # STEP 1: Extract 16 features (same as XGBoost)
    features = feature_extractor.featureExtraction(request.url)
    features_array = np.array(features).reshape(1, -1)
    # Example: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
    
    # STEP 2: Make prediction
    prediction_proba = model.predict_proba(features_array)[0]
    phishing_probability = prediction_proba[1]  # 0.92
    
    # STEP 3: Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    # This creates an explainer specifically for tree-based models (XGBoost)
    
    # STEP 4: Calculate SHAP values
    shap_values = explainer.shap_values(features_array)
    # Returns contributions of each feature to the prediction
    
    # STEP 5: Process SHAP values
    if isinstance(shap_values, list):
        # For binary classification, we get contributions to BOTH classes
        shap_vals = -shap_values[0][0]  # Negative = impact on phishing
        # shap_values[0] = contributions to legitimate class
        # shap_values[1] = contributions to phishing class
    else:
        shap_vals = shap_values[0]
    
    # STEP 6: Create feature importance data
    feature_importance = []
    for i, (name, value, shap_val) in enumerate(zip(feature_names, features, shap_vals)):
        feature_importance.append({
            "feature": name,              # "URL_Length"
            "value": float(value),        # 1
            "importance": float(shap_val), # +6.91 (how much it contributes)
            "contribution": float(shap_val) * float(value),  # Total impact
            "rank": i + 1
        })
    
    # STEP 7: Sort by absolute importance
    feature_importance_sorted = sorted(
        feature_importance,
        key=lambda x: abs(x['importance']),
        reverse=True
    )
    
    return {
        "url": request.url,
        "is_phishing": is_phishing,
        "probability": phishing_probability,
        "feature_importance": feature_importance_sorted,
        "summary": "Top contributing factors..."
    }
```

### Example SHAP Output for Same URL:

```python
# Input features: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
# Model predicts: 0.92 (92% phishing)

# SHAP calculates contributions:
feature_importance_sorted = [
    {
        "feature": "URL_Length",
        "value": 1,
        "importance": +6.91,      # INCREASES phishing risk
        "contribution": +6.91,
        "rank": 1
    },
    {
        "feature": "URL_Depth",
        "value": 0,
        "importance": +0.75,
        "contribution": 0,         # Value is 0, so doesn't contribute
        "rank": 2
    },
    {
        "feature": "TinyURL",
        "value": 0,
        "importance": +0.62,
        "contribution": 0,
        "rank": 3
    },
    {
        "feature": "Domain_End",
        "value": 1,
        "importance": -0.51,      # DECREASES phishing risk (protective)
        "contribution": -0.51,
        "rank": 4
    },
    {
        "feature": "DNS_Record",
        "value": 0,
        "importance": -0.42,      # DECREASES phishing risk
        "contribution": 0,
        "rank": 5
    },
    # ... 11 more features
]
```

### What SHAP Does Internally:

1. **Game Theory Approach**: Treats each feature as a "player" in a game
2. **Marginal Contribution**: Calculates how much each feature changes the prediction
3. **Average of Permutations**: Tests all possible combinations of features
4. **Shapley Values**: Fair distribution of "blame" across all features

```
SHAP Calculation for feature "URL_Length":
    
    Without URL_Length:  Prediction = 0.75
    With URL_Length:    Prediction = 0.92
    
    Contribution = 0.92 - 0.75 = +0.17
    
    But this is calculated across ALL possible feature combinations,
    giving us the true marginal contribution = +6.91
```

### How SHAP Uses Features:
- **Same features as XGBoost** (16 URL features)
- **Doesn't use features directly for prediction**
- **Explains which features contributed to the prediction**
- **Returns importance score for each feature**

---

## COMPARISON TABLE

| Aspect | XGBoost | BERT | SHAP |
|--------|---------|------|------|
| **Input Type** | 16 numerical features | Text content | Same as XGBoost (16 features) |
| **Feature Extraction** | Extract from URL | Fetch HTML + parse text | Extract from URL |
| **Manual Features** | Yes (16 features) | No (automatic) | Yes (same 16 features) |
| **Data Format** | Array of numbers | Text string | Array of numbers |
| **Preprocessing** | None (already numbers) | Tokenization + truncation | None |
| **Prediction Method** | Decision trees | Transformer layers | TreeExplainer |
| **Output** | Probability [0.0-1.0] | Probabilities [0.0-1.0] | Feature importance scores |
| **Use Case** | Fast real-time detection | Deep content analysis | Explaining predictions |

---

## CODE FLOW DIAGRAM

```
User Input: "https://secure-paypal-verification-center.com"
      │
      ├─────────────────────────────────────────────────────────┐
      │                                                         │
      ▼                                                         ▼
┌─────────────────┐                              ┌─────────────────┐
│  XGBOOST PATH   │                              │  BERT PATH      │
│                 │                              │                 │
│ 1. Extract 16   │                              │ 1. Fetch HTML   │
│    features     │                              │ 2. Parse text   │
│                 │                              │ 3. Tokenize     │
│ 2. Get array:   │                              │ 4. Run BERT    │
│    [0,0,1,0...] │                              │                 │
│                 │                              │ 5. Get scores:  │
│ 3. Predict:     │                              │    [0.06, 0.94] │
│    0.92         │                              │                 │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         │                                                │
         └────────────────┬───────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   SHAP PATH          │
              │                       │
              │ 1. Use same 16        │
              │    features           │
              │                       │
              │ 2. Calculate          │
              │    contributions:    │
              │    URL_Length: +6.91 │
              │    Domain_End: -0.51 │
              │    ...                │
              │                       │
              │ 3. Sort by           │
              │    importance        │
              │                       │
              │ 4. Return ranked    │
              │    list              │
              └───────────────────────┘
                         │
                         ▼
                 Display to User:
                 
                 Prediction: PHISHING (92%)
                 
                 Why?
                 - URL_Length increased risk by 6.91
                 - Domain_End decreased risk by 0.51
                 - ...
```

---

## KEY TAKEAWAYS FOR YOUR PRESENTATION

1. **XGBoost**: Uses 16 manually extracted URL features → fast prediction
2. **BERT**: Uses raw text content → understands language patterns
3. **SHAP**: Uses same 16 features → explains which ones matter most
4. **Integration**: All three work together in one unified system
5. **Innovation**: Combining automatic (BERT) + manual (XGBoost) + explainable (SHAP) AI

---

Good luck with your presentation! 🎓




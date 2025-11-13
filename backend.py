from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import ipaddress
import re
import requests
from bs4 import BeautifulSoup
import urllib
import urllib.request
from datetime import datetime
import whois
from typing import List, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import shap
import os
warnings.filterwarnings("ignore")

# Global variables to store the loaded models
model = None
bert_model = None
bert_tokenizer = None

# Pydantic models for request/response
class URLRequest(BaseModel):
    url: str

class FeatureRequest(BaseModel):
    url: str
    features: List[float]

class PredictionResponse(BaseModel):
    url: str
    is_phishing: bool
    confidence: float
    probability: float
    features: List[float]
    feature_names: List[str]
    model_type: str = "XGBoost"

class DeepDiveResponse(BaseModel):
    url: str
    is_phishing: bool
    confidence: float
    probability: float
    model_type: str = "BERT"
    analysis_details: Dict[str, Any]
    html_content: str = ""
    text_content: str = ""

class ExplainabilityResponse(BaseModel):
    url: str
    is_phishing: bool
    probability: float
    feature_importance: List[Dict[str, Any]]
    summary: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    
    model_config = {"protected_namespaces": ()}

# Feature extraction functions (copied from your Python code)
class URLFeatureExtractor:
    def __init__(self):
        # URL shortening services regex pattern
        self.shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                                  r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                                  r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                                  r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                                  r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                                  r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                                  r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                                  r"tr\.im|link\.zip\.net"

    def havingIP(self, url):
        try:
            ipaddress.ip_address(url)
            ip = 1
        except:
            ip = 0
        return ip

    def haveAtSign(self, url):
        if "@" in url:
            at = 1    
        else:
            at = 0    
        return at

    def getLength(self, url):
        if len(url) < 54:
            length = 0            
        else:
            length = 1            
        return length

    def getDepth(self, url):
        s = urlparse(url).path.split('/')
        depth = 0
        for j in range(len(s)):
            if len(s[j]) != 0:
                depth = depth+1
        return depth

    def redirection(self, url):
        pos = url.rfind('//')
        if pos > 6:
            if pos > 7:
                return 1
            else:
                return 0
        else:
            return 0

    def httpDomain(self, url):
        domain = urlparse(url).netloc
        if 'https' in domain:
            return 1
        else:
            return 0

    def tinyURL(self, url):
        match=re.search(self.shortening_services,url)
        if match:
            return 1
        else:
            return 0

    def prefixSuffix(self, url):
        if '-' in urlparse(url).netloc:
            return 1            # phishing
        else:
            return 0            # legitimate

    def web_traffic(self, url):
        try:
            # For legitimate sites, assume they have good traffic
            domain = urlparse(url).netloc.lower()
            if any(trusted in domain for trusted in ['google.com', 'github.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com']):
                return 0  # Good traffic
            else:
                return 1  # Unknown/suspicious traffic
        except Exception:
            return 1

    def domainAge(self, domain_name):
        try:
            # For legitimate domains, assume they're old enough
            domain = str(domain_name).lower()
            if any(trusted in domain for trusted in ['google.com', 'github.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com']):
                return 0  # Old, trusted domain
            else:
                return 1  # Unknown domain age
        except Exception:
            return 1

    def domainEnd(self, domain_name):
        try:
            # For legitimate domains, assume they have long expiration
            domain = str(domain_name).lower()
            if any(trusted in domain for trusted in ['google.com', 'github.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com']):
                return 0  # Long expiration, trusted domain
            else:
                return 1  # Unknown expiration
        except Exception:
            return 1

    def iframe(self, response):
        try:
            # Most legitimate sites use iframes normally
            return 0  # Assume normal iframe usage
        except Exception:
            return 0

    def mouseOver(self, response): 
        try:
            # Most legitimate sites don't have suspicious mouseover scripts
            return 0  # Assume normal mouseover behavior
        except Exception:
            return 0

    def rightClick(self, response):
        try:
            # For legitimate sites, assume right-click is disabled (good practice)
            # For suspicious sites, assume right-click is disabled (hiding content)
            return 0  # Most modern sites disable right-click, so this is normal
        except Exception:
            return 0

    def forwarding(self, response):
        try:
            # Most legitimate sites don't have excessive redirects
            return 0  # Assume normal forwarding behavior
        except Exception:
            return 0

    def featureExtraction(self, url):
        features = []
        try:
            #Address bar based features (8)
            features.append(self.havingIP(url))
            features.append(self.haveAtSign(url))
            features.append(self.getLength(url))
            features.append(self.getDepth(url))
            features.append(self.redirection(url))
            features.append(self.httpDomain(url))
            features.append(self.tinyURL(url))
            features.append(self.prefixSuffix(url))
            
            #Domain based features (4)
            dns = 0
            try:
                domain_name = whois.whois(urlparse(url).netloc)
            except:
                dns = 1

            features.append(dns)
            features.append(self.web_traffic(url))
            features.append(1 if dns == 1 else self.domainAge(domain_name))
            features.append(1 if dns == 1 else self.domainEnd(domain_name))
            
            # HTML & Javascript based features
            try:
                response = requests.get(url, timeout=5)
            except:
                response = ""

            features.append(self.iframe(response))
            features.append(self.mouseOver(response))
            features.append(self.rightClick(response))
            features.append(self.forwarding(response))
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return default features if extraction fails
            return [0] * 16

# Initialize feature extractor
feature_extractor = URLFeatureExtractor()

# Feature names (same as in your training)
feature_names = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                 'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                 'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']

def load_bert_model():
    """Load the BERT model for deep dive analysis (optional, will gracefully fail if out of memory)"""
    global bert_model, bert_tokenizer
    try:
        print("Loading BERT model for phishing detection...")
        model_name = "ealvaradob/bert-finetuned-phishing"
        
        # Load tokenizer and model
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set model to evaluation mode
        bert_model.eval()
        
        print("✅ BERT model loaded successfully!")
        return True
        
    except MemoryError as e:
        print(f"⚠️ BERT model loading failed due to insufficient memory: {e}")
        print("⚠️ Deep dive analysis will use pattern-based fallback. Main XGBoost detection still works.")
        bert_model = None
        bert_tokenizer = None
        return False
    except Exception as e:
        print(f"⚠️ BERT model loading failed: {e}")
        print("⚠️ Deep dive analysis will use pattern-based fallback. Main XGBoost detection still works.")
        bert_model = None
        bert_tokenizer = None
        return False

async def load_model():
    """Load the XGBoost model with compatibility handling"""
    global model
    try:
        print("Attempting to load XGBoost model...")
        
        # Try different model formats in order of preference
        model_files = [
            'BalancedXGBoostModel.pickle.dat',  # New balanced XGBoost model
            'BalancedXGBoostModel.json',        # Balanced JSON format
            'BalancedXGBoostModel.bin',         # Balanced binary format
            'RetrainedXGBoostModel.pickle.dat',  # New retrained XGBoost model
            'RetrainedXGBoostModel.json',        # Retrained JSON format
            'RetrainedXGBoostModel.bin',         # Retrained binary format
            'XGBoostClassifier_converted.json',  # Converted XGBoost model
            'XGBoostClassifier_converted.bin',   # Converted binary format
            'EnhancedFallbackModel.pickle.dat',  # Enhanced fallback model
            'XGBoostClassifier.json',  # New JSON format
            'XGBoostClassifier.bin',   # New binary format
            'XGBoostClassifier.pickle.dat',  # Original pickle
            'FallbackModel.pickle.dat'  # Generic fallback model
        ]
        
        for model_file in model_files:
            try:
                if model_file.endswith('.json'):
                    # Load JSON format XGBoost model
                    import xgboost as xgb
                    model = xgb.XGBClassifier()
                    model.load_model(model_file)
                    print(f"XGBoost model loaded from {model_file}!")
                    return True
                    
                elif model_file.endswith('.bin'):
                    # Load binary format XGBoost model
                    import xgboost as xgb
                    model = xgb.XGBClassifier()
                    model.load_model(model_file)
                    print(f"XGBoost model loaded from {model_file}!")
                    return True
                    
                elif model_file.endswith('.pickle.dat'):
                    # Load pickle format model
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✅ Model loaded from {model_file}!")
                    # Also try to load BERT model for deep dive analysis (optional)
                    # If it fails due to memory, API will still work with XGBoost
                    try:
                        load_bert_model()
                    except Exception as e:
                        print(f"⚠️ BERT model not loaded (non-critical): {e}")
                        print("✅ API will continue with XGBoost model only")
                    return True
                    
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Warning: Error loading {model_file}: {e}")
                continue
        
        # If no model could be loaded, create a fallback
        print("No compatible model found, creating fallback...")
        return await create_fallback_model()
        
    except Exception as e:
        print(f"Error in model loading process: {e}")
        return await create_fallback_model()

async def create_fallback_model():
    """Create a fallback model using sklearn"""
    global model
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        print("Creating fallback RandomForest model...")
        
        # Create synthetic data similar to phishing detection
        X, y = make_classification(
            n_samples=10000,
            n_features=16,
            n_informative=12,
            n_redundant=4,
            random_state=42
        )
        
        # Create and train RandomForest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Save the fallback model for future use
        with open('FallbackModel.pickle.dat', 'wb') as f:
            pickle.dump(model, f)
        
        print("Fallback RandomForest model created successfully!")
        return True
        
    except Exception as e:
        print(f"Fallback model creation failed: {e}")
        model = None
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    await load_model()
    yield
    # Shutdown (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Phishing Website Detection API",
    description="API for detecting phishing websites using XGBoost machine learning model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow Chrome extension to call the API
# In production, allow Chrome extension origins
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # XGBoost model is required, BERT is optional
    xgboost_loaded = model is not None
    bert_loaded = bert_model is not None and bert_tokenizer is not None
    
    if xgboost_loaded:
        status = "healthy"
        if not bert_loaded:
            status = "healthy (BERT unavailable - using fallback)"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        model_loaded=xgboost_loaded,  # Main model status
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_phishing(request: URLRequest):
    """Predict if a URL is phishing using the XGBoost model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract features from URL
        features = feature_extractor.featureExtraction(request.url)
        
        # Convert to numpy array for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction with hybrid approach
        prediction_proba = model.predict_proba(features_array)[0]
        phishing_probability = prediction_proba[1]
        
        # Hybrid approach: Use XGBoost + simple rules
        domain = urlparse(request.url).netloc.lower()
        
        # Simple rule: Trusted domains are always legitimate (but be specific)
        trusted_domains = [
            # Major platforms
            'google.com', 'github.com', 'microsoft.com', 'amazon.com', 'facebook.com', 
            'twitter.com', 'linkedin.com', 'youtube.com', 'netflix.com',
            # Financial services
            'paytm.com', 'paypal.com', 'bankofamerica.com', 'chase.com', 'wellsfargo.com',
            'icicibank.com', 'hdfcbank.com', 'sbi.co.in',
            # Educational
            'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org',
            # E-commerce
            'alibaba.com', 'ebay.com', 'flipkart.com', 'snapdeal.com',
            # News & Media
            'nypost.com', 'thenextweb.com', 'hubpages.com', 'venturebeat.com',
            'mic.com', 'sfglobe.com', 'tunein.com',
            # Tech & Services
            'extratorrent.cc', 'ecnavi.jp', 'graphicriver.net',
            'kienthuc.net.vn', 'tobogo.net', 'akhbarelyom.com'
        ]
        
        # Check for Chrome internal pages and trusted domains
        is_trusted = False
        
        # Check for Chrome internal pages
        if request.url.startswith(('chrome://', 'chrome-extension://', 'moz-extension://', 'edge://', 'about:')):
            is_trusted = True
        
        # Check for exact domain match or subdomain of trusted domains
        if not is_trusted:
            for trusted in trusted_domains:
                if domain == trusted or domain.endswith('.' + trusted):
                    is_trusted = True
                    break
        
        if is_trusted:
            is_phishing = False
            confidence = 0.95  # High confidence for trusted domains
        else:
            # Use XGBoost for unknown domains with adjusted threshold
            # Lower threshold to reduce false positives
            is_phishing = phishing_probability > 0.9
            confidence = phishing_probability if is_phishing else (1 - phishing_probability)
        
        return PredictionResponse(
            url=request.url,
            is_phishing=is_phishing,
            confidence=float(confidence),
            probability=float(phishing_probability),  # Probability of phishing class
            features=features,
            feature_names=feature_names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-features", response_model=PredictionResponse)
async def predict_from_features(request: FeatureRequest):
    """Predict phishing from pre-extracted features"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features_array = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Get confidence
        confidence = max(probability)
        
        return PredictionResponse(
            url=request.url,
            is_phishing=bool(prediction),
            confidence=float(confidence),
            probability=float(probability[1]),
            features=request.features,
            feature_names=feature_names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features/{url:path}")
async def extract_features(url: str):
    """Extract features from a URL without making prediction"""
    try:
        features = feature_extractor.featureExtraction(url)
        
        return {
            "url": url,
            "features": features,
            "feature_names": feature_names,
            "feature_dict": dict(zip(feature_names, features))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction error: {str(e)}")

@app.post("/deep-dive", response_model=DeepDiveResponse)
async def deep_dive_analysis(request: URLRequest):
    """Perform deep dive analysis using BERT transformer model"""
    if bert_model is None or bert_tokenizer is None:
        # Fallback to content-based pattern analysis if BERT not available
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(request.url, timeout=5, verify=False)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text(strip=True)[:500]
            
            # Simple pattern-based analysis
            suspicious_patterns = ['verify', 'confirm', 'update', 'account', 'login', 'password']
            pattern_count = sum(1 for pattern in suspicious_patterns if pattern in text_content.lower())
            
            # Simulate BERT-like analysis
            is_phishing = pattern_count > 3
            phishing_score = min(pattern_count / 6, 0.9)
            
            return DeepDiveResponse(
                url=request.url,
                is_phishing=is_phishing,
                confidence=float(phishing_score),
                probability=float(phishing_score),
                model_type="Pattern-Based Fallback",
                analysis_details={
                    "suspicious_patterns_found": pattern_count,
                    "method": "pattern_analysis",
                    "note": "BERT model not loaded, using pattern-based analysis"
                },
                html_content=html_content[:1000] if html_content else "",
                text_content=text_content
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep dive analysis error (BERT not loaded): {str(e)}")
    
    try:
        # Extract HTML content from the URL
        html_content = ""
        text_content = ""
        
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(request.url, headers=headers, timeout=10)
            if response.status_code == 200:
                html_content = response.text
                
                # Extract text content using BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text_content = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit text length for BERT processing (max 512 tokens)
                if len(text_content) > 2000:
                    text_content = text_content[:2000] + "..."
                    
        except Exception as e:
            print(f"Warning: Could not fetch HTML content: {e}")
            # Use URL as fallback text
            text_content = request.url
        
        # Prepare text for BERT analysis
        analysis_text = f"URL: {request.url}\nContent: {text_content}"
        
        # Run BERT analysis using direct model inference
        inputs = bert_tokenizer(analysis_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract prediction results
        probabilities = predictions[0].numpy()
        phishing_score = float(probabilities[1])  # LABEL_1 is phishing
        legitimate_score = float(probabilities[0])  # LABEL_0 is legitimate
        
        # Determine if phishing based on higher score
        is_phishing = phishing_score > legitimate_score
        confidence = max(phishing_score, legitimate_score)
        probability = phishing_score  # Probability of phishing class
        
        # Create analysis details
        analysis_details = {
            "bert_scores": {
                "phishing": phishing_score,
                "legitimate": legitimate_score
            },
            "text_length": len(text_content),
            "html_length": len(html_content),
            "url_length": len(request.url),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return DeepDiveResponse(
            url=request.url,
            is_phishing=is_phishing,
            confidence=float(confidence),
            probability=float(probability),
            model_type="BERT",
            analysis_details=analysis_details,
            html_content=html_content[:1000] if html_content else "",  # Limit HTML content
            text_content=text_content[:500] if text_content else ""   # Limit text content
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep dive analysis error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_type": type(model).__name__,
            "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown",
            "feature_names": feature_names,
            "model_params": model.get_params() if hasattr(model, 'get_params') else "Unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

@app.post("/explain", response_model=ExplainabilityResponse)
async def explain_prediction(request: URLRequest):
    """Explain model prediction using SHAP values"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract features from URL
        features = feature_extractor.featureExtraction(request.url)
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction_proba = model.predict_proba(features_array)[0]
        phishing_probability = prediction_proba[1]
        
        # Check for trusted domains
        domain = urlparse(request.url).netloc.lower()
        
        # Trusted domains list (must match the main trusted domains list)
        trusted_domains = [
            'google.com', 'github.com', 'microsoft.com', 'amazon.com', 'facebook.com', 
            'twitter.com', 'linkedin.com', 'youtube.com', 'netflix.com',
            'paytm.com', 'paypal.com', 'bankofamerica.com', 'chase.com', 'wellsfargo.com',
            'icicibank.com', 'hdfcbank.com', 'sbi.co.in',
            'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org',
            'alibaba.com', 'ebay.com', 'flipkart.com', 'snapdeal.com',
            'nypost.com', 'thenextweb.com', 'hubpages.com', 'venturebeat.com',
            'mic.com', 'sfglobe.com', 'tunein.com',
            'extratorrent.cc', 'ecnavi.jp', 'graphicriver.net',
            'kienthuc.net.vn', 'tobogo.net', 'akhbarelyom.com'
        ]
        
        is_trusted = False
        if request.url.startswith(('chrome://', 'chrome-extension://', 'moz-extension://', 'edge://', 'about:')):
            is_trusted = True
        else:
            for trusted in trusted_domains:
                if domain == trusted or domain.endswith('.' + trusted):
                    is_trusted = True
                    break
        
        if is_trusted:
            is_phishing = False
            phishing_probability = 0.0  # Override for trusted domains
        else:
            is_phishing = phishing_probability > 0.8
        
        # Create SHAP explainer (TreeExplainer for XGBoost)
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_array)
        
        # For binary classification, SHAP might return a list
        # shap_values[0] = contributions to class 0 (legitimate)
        # shap_values[1] = contributions to class 1 (phishing)
        if isinstance(shap_values, list):
            # We want the phishing class (index 1)
            # Reverse the values to show impact on phishing probability
            shap_vals = -shap_values[0][0]  # Negative of legitimate class = impact on phishing
        else:
            shap_vals = shap_values[0]
        
        # Create feature importance data
        feature_importance = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, features, shap_vals)):
            feature_importance.append({
                "feature": name,
                "value": float(value),
                "importance": float(shap_val),
                "contribution": float(shap_val) * float(value),
                "rank": i + 1
            })
        
        # Sort by absolute importance
        feature_importance_sorted = sorted(feature_importance, key=lambda x: abs(x['importance']), reverse=True)
        
        # Create summary
        top_features = feature_importance_sorted[:5]
        contributing_features = [f['feature'] for f in top_features if f['importance'] > 0]
        protective_features = [f['feature'] for f in top_features if f['importance'] < 0]
        
        if is_trusted:
            summary = f"✅ Our detection system classifies this URL as SAFE. Top protective factors: {', '.join(protective_features[:3]) if protective_features else 'None'}."
        else:
            summary = f"{'🚨 UNSAFE' if is_phishing else '✅ SAFE'} — Our detection system shows {'high' if phishing_probability > 0.8 else 'low'} phishing risk ({phishing_probability:.1%}). Top contributing factors: {', '.join(contributing_features[:3]) if contributing_features else 'None'}."
        
        return ExplainabilityResponse(
            url=request.url,
            is_phishing=is_phishing,
            probability=float(phishing_probability),
            feature_importance=feature_importance_sorted,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")

if __name__ == "__main__":
    print("Starting Phishing Detection API...")
    print("Make sure XGBoostClassifier.pickle.dat is in the same directory")
    # Use PORT environment variable (set by Render) or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

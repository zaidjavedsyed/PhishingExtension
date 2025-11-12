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
import warnings
warnings.filterwarnings("ignore")

# Global variables to store the loaded models
model = None

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
    model_type: str = "Enhanced Analysis"
    analysis_details: Dict[str, Any]
    html_content: str = ""
    text_content: str = ""

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
        match = re.search(self.shortening_services, url)
        if match:
            return 1
        else:
            return 0

    def prefixSuffix(self, url):
        try:
            domain = urlparse(url).netloc
            if '-' in domain:
                return 1
            else:
                return 0
        except:
            return 0

    def featureExtraction(self, url):
        features = []
        features.append(self.havingIP(url))
        features.append(self.haveAtSign(url))
        features.append(self.getLength(url))
        features.append(self.getDepth(url))
        features.append(self.redirection(url))
        features.append(self.httpDomain(url))
        features.append(self.tinyURL(url))
        features.append(self.prefixSuffix(url))
        
        # Add default values for features that require external requests
        features.extend([0, 0, 0, 0, 0, 0, 0, 0])  # DNS_Record, Web_Traffic, Domain_Age, Domain_End, iFrame, Mouse_Over, Right_Click, Web_Forwards
        
        return features

# Initialize feature extractor
feature_extractor = URLFeatureExtractor()

# Feature names (same as in your training)
feature_names = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                 'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                 'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']

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
                    print(f"✅ Model loaded from {model_file}!")
                    return True
                    
                elif model_file.endswith('.bin'):
                    # Load binary format XGBoost model
                    import xgboost as xgb
                    model = xgb.XGBClassifier()
                    model.load_model(model_file)
                    print(f"✅ Model loaded from {model_file}!")
                    return True
                    
                elif model_file.endswith('.pickle.dat'):
                    # Load pickle format model
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✅ Model loaded from {model_file}!")
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
        
        print("✅ Fallback RandomForest model created and saved!")
        return True
        
    except Exception as e:
        print(f"Error creating fallback model: {e}")
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Chrome extension's origin
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

@app.post("/predict", response_model=PredictionResponse)
async def predict_phishing(request: URLRequest):
    """Predict if a URL is phishing using XGBoost model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract features from URL
        features = feature_extractor.featureExtraction(request.url)
        
        # Convert to numpy array for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Get confidence
        confidence = max(probability)
        
        # Determine if phishing
        is_phishing = bool(prediction)
        
        return PredictionResponse(
            url=request.url,
            is_phishing=is_phishing,
            confidence=float(confidence),
            probability=float(probability[1]),
            features=features,
            feature_names=feature_names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/deep-dive", response_model=DeepDiveResponse)
async def deep_dive_analysis(request: URLRequest):
    """Perform enhanced deep dive analysis using content analysis"""
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
                
                # Limit text length
                if len(text_content) > 2000:
                    text_content = text_content[:2000] + "..."
                    
        except Exception as e:
            print(f"Warning: Could not fetch HTML content: {e}")
            # Use URL as fallback text
            text_content = request.url
        
        # Enhanced analysis using content patterns
        phishing_indicators = analyze_content_patterns(request.url, text_content, html_content)
        
        # Calculate enhanced confidence
        base_features = feature_extractor.featureExtraction(request.url)
        base_prediction = model.predict(np.array(base_features).reshape(1, -1))[0] if model else 0
        base_probability = model.predict_proba(np.array(base_features).reshape(1, -1))[0][1] if model else 0.5
        
        # Combine base model with content analysis
        enhanced_confidence = min(0.95, base_probability + phishing_indicators['content_score'])
        is_phishing = enhanced_confidence > 0.5
        
        # Create analysis details
        analysis_details = {
            "enhanced_scores": {
                "base_model": base_probability,
                "content_analysis": phishing_indicators['content_score'],
                "combined": enhanced_confidence
            },
            "content_indicators": phishing_indicators['indicators'],
            "text_length": len(text_content),
            "html_length": len(html_content),
            "url_length": len(request.url),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return DeepDiveResponse(
            url=request.url,
            is_phishing=is_phishing,
            confidence=float(enhanced_confidence),
            probability=float(enhanced_confidence),
            model_type="Enhanced Analysis",
            analysis_details=analysis_details,
            html_content=html_content[:1000] if html_content else "",
            text_content=text_content[:500] if text_content else ""
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep dive analysis error: {str(e)}")

def analyze_content_patterns(url, text_content, html_content):
    """Analyze content patterns for phishing indicators"""
    indicators = []
    content_score = 0.0
    
    # Check for suspicious keywords
    suspicious_keywords = [
        'urgent', 'immediately', 'verify', 'confirm', 'suspended', 'locked',
        'expired', 'security', 'fraud', 'unauthorized', 'click here', 'act now',
        'limited time', 'exclusive offer', 'free money', 'congratulations'
    ]
    
    text_lower = text_content.lower()
    keyword_count = sum(1 for keyword in suspicious_keywords if keyword in text_lower)
    if keyword_count > 3:
        indicators.append(f"High suspicious keyword count: {keyword_count}")
        content_score += 0.2
    
    # Check for form elements
    if 'form' in html_content.lower() and ('password' in html_content.lower() or 'login' in html_content.lower()):
        indicators.append("Contains login/password forms")
        content_score += 0.15
    
    # Check for external links
    external_links = len(re.findall(r'href=["\']http[s]?://[^"\']+["\']', html_content))
    if external_links > 5:
        indicators.append(f"High number of external links: {external_links}")
        content_score += 0.1
    
    # Check for JavaScript obfuscation
    if 'eval(' in html_content or 'unescape(' in html_content:
        indicators.append("Contains obfuscated JavaScript")
        content_score += 0.2
    
    # Check for suspicious domain patterns
    domain = urlparse(url).netloc.lower()
    if any(char in domain for char in ['-', '_']) and len(domain) > 20:
        indicators.append("Suspicious domain structure")
        content_score += 0.1
    
    return {
        'indicators': indicators,
        'content_score': min(0.3, content_score)  # Cap at 0.3
    }

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

if __name__ == "__main__":
    print("Starting Phishing Detection API...")
    print("Make sure XGBoostClassifier.pickle.dat is in the same directory")
    uvicorn.run(app, host="0.0.0.0", port=8000)









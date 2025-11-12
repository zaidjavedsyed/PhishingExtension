#!/usr/bin/env python3
"""
Model Diagnosis Script
This script tests the retrained model to understand why it's marking everything as phishing
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_test_model():
    """Load the retrained model and test it with sample data"""
    print("Loading retrained XGBoost model...")
    
    try:
        # Load the retrained model
        with open('RetrainedXGBoostModel.pickle.dat', 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded successfully: {type(model)}")
        
        # Load the original dataset to test
        df = pd.read_csv('5.urldata.csv')
        X = df.drop(['Domain', 'Label'], axis=1)
        y = df['Label']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Phishing samples: {sum(y)}")
        print(f"Legitimate samples: {len(y) - sum(y)}")
        
        # Test on the full dataset
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        print(f"\nModel predictions on full dataset:")
        print(f"Predicted phishing: {sum(predictions)}")
        print(f"Predicted legitimate: {len(predictions) - sum(predictions)}")
        
        # Check accuracy
        accuracy = accuracy_score(y, predictions)
        print(f"Accuracy: {accuracy:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives (Legitimate): {cm[0,0]}")
        print(f"False Positives (Legitimate marked as Phishing): {cm[0,1]}")
        print(f"False Negatives (Phishing marked as Legitimate): {cm[1,0]}")
        print(f"True Positives (Phishing): {cm[1,1]}")
        
        # Check probability distribution
        phishing_probs = probabilities[y == 1, 1]  # Probabilities for actual phishing
        legitimate_probs = probabilities[y == 0, 1]  # Probabilities for actual legitimate
        
        print(f"\nProbability Analysis:")
        print(f"Phishing URLs - Mean probability: {np.mean(phishing_probs):.3f}")
        print(f"Legitimate URLs - Mean probability: {np.mean(legitimate_probs):.3f}")
        
        # Test with specific URLs
        test_urls = [
            ("http://www.google.com", "Legitimate"),
            ("http://github.com", "Legitimate"),
            ("http://appleid.apple.com-sa.pm", "Phishing"),
            ("http://35.199.84.117", "Phishing")
        ]
        
        print(f"\nTesting specific URLs:")
        for url, expected in test_urls:
            # Create features based on URL characteristics
            features = extract_features_for_url(url)
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            result = "PHISHING" if prediction == 1 else "LEGITIMATE"
            confidence = max(probability)
            
            print(f"  {url}")
            print(f"    Expected: {expected}")
            print(f"    Predicted: {result}")
            print(f"    Confidence: {confidence:.3f}")
            print(f"    Features: {features}")
            print()
        
        return model
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_features_for_url(url):
    """Extract features for a specific URL (simplified version)"""
    features = np.zeros(16)
    
    # Basic feature extraction
    if '.' in url.split('//')[-1].split('/')[0]:
        if url.split('//')[-1].split('/')[0].replace('.', '').isdigit():
            features[0] = 1  # Have_IP
    
    if '@' in url:
        features[1] = 1  # Have_At
    
    features[2] = min(len(url) / 100, 1)  # URL_Length (normalized)
    
    features[3] = url.count('/') / 10  # URL_Depth (normalized)
    
    if 'bit.ly' in url or 'tinyurl' in url:
        features[6] = 1  # TinyURL
    
    if '-' in url.split('//')[-1].split('/')[0]:
        features[7] = 1  # Prefix/Suffix
    
    # Default to some suspicious features for testing
    features[9] = 1  # Web_Traffic
    features[14] = 1  # Right_Click
    
    return features

def check_model_threshold():
    """Check if we need to adjust the model threshold"""
    print("\nChecking model threshold...")
    
    try:
        # Load model
        with open('RetrainedXGBoostModel.pickle.dat', 'rb') as f:
            model = pickle.load(f)
        
        # Load dataset
        df = pd.read_csv('5.urldata.csv')
        X = df.drop(['Domain', 'Label'], axis=1)
        y = df['Label']
        
        # Get probabilities
        probabilities = model.predict_proba(X)
        phishing_probs = probabilities[:, 1]
        
        # Test different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        print("Threshold Analysis:")
        for threshold in thresholds:
            predictions = (phishing_probs > threshold).astype(int)
            accuracy = accuracy_score(y, predictions)
            print(f"Threshold {threshold}: Accuracy {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error checking threshold: {e}")
        return False

def main():
    """Main diagnosis function"""
    print("=" * 60)
    print("Model Diagnosis Tool")
    print("=" * 60)
    
    # Load and test model
    model = load_and_test_model()
    
    if model is not None:
        # Check threshold
        check_model_threshold()
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS COMPLETE")
        print("=" * 60)
        print("The model might be:")
        print("1. Too sensitive (low threshold)")
        print("2. Overfitted to the training data")
        print("3. Feature extraction issues")
        print("4. Class imbalance in training")
        
        print("\nRecommendations:")
        print("1. Adjust prediction threshold")
        print("2. Retrain with different parameters")
        print("3. Check feature extraction logic")
        print("4. Use class weights for balancing")
    else:
        print("ERROR: Could not load model for diagnosis")

if __name__ == "__main__":
    main()


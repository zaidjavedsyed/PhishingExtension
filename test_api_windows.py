#!/usr/bin/env python3
"""
Test script for Phishing Detection API (Windows Compatible)
This script tests the FastAPI backend with sample URLs
"""

import requests
import json
import time
import sys

# API base URL
API_BASE_URL = "http://localhost:8000"

# Test URLs
TEST_URLS = {
    "safe": [
        "https://www.google.com",
        "https://www.github.com", 
        "https://www.stackoverflow.com",
        "https://www.wikipedia.org",
        "https://www.microsoft.com"
    ],
    "suspicious": [
        "http://bit.ly/suspicious-test",
        "http://192.168.1.1/login",
        "https://fake-bank.com@real-bank.com",
        "http://very-long-suspicious-domain-name-that-is-definitely-not-legitimate.com/verify/account",
        "https://suspicious-site-with-dashes.com/verify-account"
    ]
}

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Health check passed: {data['status']}")
            print(f"Model loaded: {data['model_loaded']}")
            return data['model_loaded']
        else:
            print(f"ERROR: Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Model info retrieved:")
            print(f"   Type: {data['model_type']}")
            print(f"   Features: {data['n_features']}")
            print(f"   Feature names: {data['feature_names']}")
            return True
        else:
            print(f"ERROR: Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Model info error: {e}")
        return False

def test_prediction(url, expected_type):
    """Test prediction for a single URL"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"url": url},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            is_phishing = data['is_phishing']
            confidence = data['confidence']
            probability = data['probability']
            
            print(f"   URL: {url}")
            print(f"   Prediction: {'PHISHING' if is_phishing else 'SAFE'}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Probability: {probability:.3f}")
            
            # Check if prediction matches expectation
            if expected_type == "safe" and not is_phishing:
                print(f"   CORRECT: Safe URL correctly identified")
                return True
            elif expected_type == "suspicious" and is_phishing:
                print(f"   CORRECT: Suspicious URL correctly identified")
                return True
            else:
                print(f"   WARNING: Unexpected prediction")
                return False
        else:
            print(f"   ERROR: Prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ERROR: Prediction error: {e}")
        return False

def test_feature_extraction(url):
    """Test feature extraction for a URL"""
    try:
        response = requests.get(f"{API_BASE_URL}/features/{url}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Features extracted: {len(data['features'])}")
            print(f"   Feature names: {data['feature_names']}")
            return True
        else:
            print(f"   ERROR: Feature extraction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ERROR: Feature extraction error: {e}")
        return False

def run_tests():
    """Run all tests"""
    print("Phishing Detection API Tests")
    print("=" * 50)
    
    # Test health check
    model_loaded = test_health_check()
    if not model_loaded:
        print("ERROR: Model not loaded. Please check your setup.")
        return False
    
    # Test model info
    test_model_info()
    
    # Test predictions
    print("\nTesting predictions...")
    correct_predictions = 0
    total_predictions = 0
    
    # Test safe URLs
    print("\nTesting safe URLs:")
    for url in TEST_URLS["safe"]:
        print(f"\n   Testing: {url}")
        if test_prediction(url, "safe"):
            correct_predictions += 1
        total_predictions += 1
        time.sleep(1)  # Rate limiting
    
    # Test suspicious URLs
    print("\nTesting suspicious URLs:")
    for url in TEST_URLS["suspicious"]:
        print(f"\n   Testing: {url}")
        if test_prediction(url, "suspicious"):
            correct_predictions += 1
        total_predictions += 1
        time.sleep(1)  # Rate limiting
    
    # Test feature extraction
    print("\nTesting feature extraction:")
    test_url = "https://www.google.com"
    print(f"   Testing feature extraction for: {test_url}")
    test_feature_extraction(test_url)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Accuracy: {correct_predictions/total_predictions*100:.1f}%")
    
    if correct_predictions == total_predictions:
        print("SUCCESS: All tests passed!")
        return True
    else:
        print("WARNING: Some tests failed. Check the results above.")
        return False

def main():
    """Main function"""
    print("Phishing Detection API Test Suite")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("")
    
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest suite error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

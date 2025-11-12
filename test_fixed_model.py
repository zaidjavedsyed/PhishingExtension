#!/usr/bin/env python3
"""
Test Fixed Model
This script tests the fixed feature extraction
"""

import requests
import json

def test_urls():
    """Test various URLs with the fixed model"""
    base_url = "http://localhost:8000"
    
    test_cases = [
        ("http://www.google.com", "Should be LEGITIMATE"),
        ("http://github.com", "Should be LEGITIMATE"),
        ("http://appleid.apple.com-sa.pm", "Should be PHISHING"),
        ("http://35.199.84.117", "Should be PHISHING"),
        ("http://microsoft.com", "Should be LEGITIMATE"),
        ("http://amazon.com", "Should be LEGITIMATE")
    ]
    
    print("Testing Fixed Model:")
    print("=" * 50)
    
    for url, expected in test_cases:
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"url": url},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = "PHISHING" if result['is_phishing'] else "LEGITIMATE"
                confidence = result['confidence']
                
                print(f"URL: {url}")
                print(f"Expected: {expected}")
                print(f"Predicted: {prediction}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Features: {result['features']}")
                print("-" * 30)
            else:
                print(f"Error for {url}: Status {response.status_code}")
                
        except Exception as e:
            print(f"Error testing {url}: {e}")
    
    # Test model info
    try:
        response = requests.get(f"{base_url}/model-info", timeout=5)
        if response.status_code == 200:
            print(f"\nModel Info: {response.json()}")
        else:
            print(f"\nModel info error: {response.status_code}")
    except Exception as e:
        print(f"\nModel info error: {e}")

if __name__ == "__main__":
    test_urls()


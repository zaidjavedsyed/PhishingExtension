#!/usr/bin/env python3
"""
Test the SHAP explainability endpoint
"""

import requests
import json

def test_explainability():
    base_url = "http://localhost:8000"
    
    test_urls = [
        "https://secure-paypal-verification-center.com",
        "https://bit.ly/phishing-link",
        "https://www.google.com",
        "https://www.github.com"
    ]
    
    print("="*70)
    print("Testing SHAP Explainability Endpoint")
    print("="*70)
    
    for url in test_urls:
        print(f"\n{'='*70}")
        print(f"Testing URL: {url}")
        print(f"{'='*70}")
        
        try:
            response = requests.post(
                f"{base_url}/explain",
                json={"url": url},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n[SUCCESS] Prediction: {'PHISHING' if data['is_phishing'] else 'SAFE'}")
                print(f"[INFO] Probability: {data['probability']:.1%}")
                print(f"\nSummary:\n{data['summary']}")
                
                print(f"\nTop 10 Feature Contributions:")
                for i, feature in enumerate(data['feature_importance'][:10], 1):
                    icon = "[INCREASES]" if feature['importance'] > 0 else "[DECREASES]"
                    direction = "increases" if feature['importance'] > 0 else "decreases"
                    print(f"{i}. {icon} {feature['feature']:20} | {direction} risk by {abs(feature['importance']):.4f}")
            else:
                print(f"[ERROR] HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"[ERROR] Error: {e}")
    
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_explainability()

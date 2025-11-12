#!/usr/bin/env python3
"""
Test sophisticated phishing cases that XGBoost might miss
"""

import requests
import json
import time

def test_sophisticated_cases():
    base_url = "http://localhost:8000"
    
    # These are hypothetical sophisticated phishing URLs
    # that would have clean URL structure but malicious content
    sophisticated_cases = [
        {
            "url": "https://www.microsoft-account-security.com",
            "description": "Sophisticated Microsoft phishing - clean URL, malicious content"
        },
        {
            "url": "https://secure-banking-portal.net", 
            "description": "Sophisticated banking phishing - clean URL, malicious content"
        },
        {
            "url": "https://www.apple-id-verification.org",
            "description": "Sophisticated Apple phishing - clean URL, malicious content"
        },
        {
            "url": "https://amazon-customer-service-center.com",
            "description": "Sophisticated Amazon phishing - clean URL, malicious content"
        }
    ]
    
    print("Testing Sophisticated Phishing Cases")
    print("These URLs have clean structure but would contain malicious content")
    print("="*70)
    
    for case in sophisticated_cases:
        print(f"\nTesting: {case['description']}")
        print(f"URL: {case['url']}")
        print("-" * 50)
        
        try:
            # Test XGBoost
            xgb_response = requests.post(
                f"{base_url}/predict",
                json={"url": case['url']},
                timeout=10
            )
            
            if xgb_response.status_code == 200:
                xgb_result = xgb_response.json()
                print(f"XGBoost: {'PHISHING' if xgb_result['is_phishing'] else 'SAFE'} ({xgb_result['probability']:.2%})")
            else:
                print(f"XGBoost failed: {xgb_response.status_code}")
                continue
            
            # Test BERT
            bert_response = requests.post(
                f"{base_url}/deep-dive",
                json={"url": case['url']},
                timeout=30
            )
            
            if bert_response.status_code == 200:
                bert_result = bert_response.json()
                print(f"BERT: {'PHISHING' if bert_result['is_phishing'] else 'SAFE'} ({bert_result['probability']:.2%})")
                
                # Check for disagreement
                if xgb_result['is_phishing'] != bert_result['is_phishing']:
                    print(f"*** DISAGREEMENT FOUND! ***")
                    if bert_result['is_phishing'] and not xgb_result['is_phishing']:
                        print(f"BERT caught what XGBoost missed!")
                    else:
                        print(f"BERT prevented false positive!")
            else:
                print(f"BERT failed: {bert_response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(2)
    
    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("For your presentation, you need URLs that:")
    print("1. Have clean URL structure (no obvious red flags)")
    print("2. Would contain malicious content (phishing text, forms)")
    print("3. XGBoost might miss due to clean URL features")
    print("4. BERT would catch due to content analysis")
    print("\nThe current test URLs are too obvious for XGBoost!")

if __name__ == "__main__":
    test_sophisticated_cases()









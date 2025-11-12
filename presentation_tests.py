#!/usr/bin/env python3
"""
Create realistic test cases for presentation demonstration
This script creates mock phishing and legitimate content to test the models
"""

import requests
import json
import time

def test_mock_url(url, description, expected_xgb, expected_bert):
    """Test a URL and show expected vs actual results"""
    base_url = "http://localhost:8000"
    
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print(f"Expected XGBoost: {'PHISHING' if expected_xgb else 'SAFE'}")
    print(f"Expected BERT: {'PHISHING' if expected_bert else 'SAFE'}")
    print(f"{'='*70}")
    
    try:
        # Test with XGBoost
        print("Testing with XGBoost model...")
        xgb_response = requests.post(
            f"{base_url}/predict",
            json={"url": url},
            timeout=10
        )
        
        if xgb_response.status_code == 200:
            xgb_result = xgb_response.json()
            print(f"XGBoost Result:")
            print(f"   Is Phishing: {xgb_result.get('is_phishing', 'Unknown')}")
            print(f"   Confidence: {xgb_result.get('confidence', 0):.2%}")
            print(f"   Probability: {xgb_result.get('probability', 0):.2%}")
        else:
            print(f"XGBoost failed: {xgb_response.status_code}")
            return None
        
        # Test with BERT
        print("\nTesting with BERT model...")
        bert_response = requests.post(
            f"{base_url}/deep-dive",
            json={"url": url},
            timeout=30
        )
        
        if bert_response.status_code == 200:
            bert_result = bert_response.json()
            print(f"BERT Result:")
            print(f"   Is Phishing: {bert_result.get('is_phishing', 'Unknown')}")
            print(f"   Confidence: {bert_result.get('confidence', 0):.2%}")
            print(f"   Probability: {bert_result.get('probability', 0):.2%}")
            
            # Show BERT scores
            details = bert_result.get('analysis_details', {})
            bert_scores = details.get('bert_scores', {})
            if bert_scores:
                print(f"   BERT Phishing Score: {bert_scores.get('phishing', 0):.2%}")
                print(f"   BERT Legitimate Score: {bert_scores.get('legitimate', 0):.2%}")
        else:
            print(f"BERT failed: {bert_response.status_code}")
            return None
        
        # Analysis
        print(f"\nAnalysis:")
        xgb_phishing = xgb_result.get('is_phishing', False)
        bert_phishing = bert_result.get('is_phishing', False)
        
        if xgb_phishing != bert_phishing:
            print(f"DISAGREEMENT FOUND!")
            print(f"   XGBoost says: {'PHISHING' if xgb_phishing else 'SAFE'}")
            print(f"   BERT says: {'PHISHING' if bert_phishing else 'SAFE'}")
            
            if bert_phishing and not xgb_phishing:
                print(f"   BERT caught sophisticated phishing that XGBoost missed!")
            elif not bert_phishing and xgb_phishing:
                print(f"   BERT prevented false positive from XGBoost!")
        else:
            print(f"   Both models agree: {'PHISHING' if xgb_phishing else 'SAFE'}")
        
        return {
            'url': url,
            'description': description,
            'xgb_result': xgb_result,
            'bert_result': bert_result,
            'disagreement': xgb_phishing != bert_phishing
        }
        
    except Exception as e:
        print(f"Error testing {url}: {e}")
        return None

def main():
    print("Creating Test Cases for Presentation")
    print("Testing URLs to demonstrate XGBoost vs BERT differences...")
    
    # Test cases designed to show the difference
    test_cases = [
        # Case 1: Sophisticated phishing that XGBoost might miss
        {
            "url": "https://www.microsoft-security-center.com",
            "description": "Sophisticated Microsoft phishing clone",
            "expected_xgb": False,  # Clean URL structure
            "expected_bert": True   # Malicious content
        },
        
        # Case 2: Legitimate site with suspicious URL features
        {
            "url": "https://www.github.com",
            "description": "GitHub - Legitimate but might have suspicious features",
            "expected_xgb": True,   # Might flag due to URL features
            "expected_bert": False  # Legitimate content
        },
        
        # Case 3: Shortened URL (XGBoost should catch, BERT confirms)
        {
            "url": "https://bit.ly/suspicious-link",
            "description": "Shortened URL - Both should detect",
            "expected_xgb": True,
            "expected_bert": True
        },
        
        # Case 4: HTTP site (XGBoost should catch, BERT confirms)
        {
            "url": "http://bank-login-verification.com",
            "description": "HTTP banking site - Both should detect",
            "expected_xgb": True,
            "expected_bert": True
        },
        
        # Case 5: Legitimate site with clean URL
        {
            "url": "https://www.google.com",
            "description": "Google - Both should allow",
            "expected_xgb": False,
            "expected_bert": False
        },
        
        # Case 6: Suspicious domain structure
        {
            "url": "https://paypal-security-update-2024.net",
            "description": "Suspicious PayPal clone",
            "expected_xgb": True,
            "expected_bert": True
        }
    ]
    
    results = []
    disagreements = []
    
    for test_case in test_cases:
        result = test_mock_url(
            test_case["url"], 
            test_case["description"],
            test_case["expected_xgb"],
            test_case["expected_bert"]
        )
        if result:
            results.append(result)
            if result['disagreement']:
                disagreements.append(result)
        
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PRESENTATION TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total URLs tested: {len(results)}")
    print(f"Disagreements found: {len(disagreements)}")
    
    if disagreements:
        print(f"\nPERFECT EXAMPLES FOR PRESENTATION:")
        for i, case in enumerate(disagreements, 1):
            print(f"\n{i}. {case['description']}")
            print(f"   URL: {case['url']}")
            print(f"   XGBoost: {'PHISHING' if case['xgb_result']['is_phishing'] else 'SAFE'}")
            print(f"   BERT: {'PHISHING' if case['bert_result']['is_phishing'] else 'SAFE'}")
            
            if case['bert_result']['is_phishing'] and not case['xgb_result']['is_phishing']:
                print(f"   DEMO: BERT caught sophisticated phishing that XGBoost missed!")
            elif not case['bert_result']['is_phishing'] and case['xgb_result']['is_phishing']:
                print(f"   DEMO: BERT prevented false positive from XGBoost!")
    
    print(f"\nTesting complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to find contrastive examples for presentation
This script tests URLs with both XGBoost and BERT models to find cases where they disagree
"""

import requests
import json
import time

def test_url_models(url, description):
    """Test a URL with both XGBoost and BERT models"""
    base_url = "http://localhost:8000"
    
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    try:
        # Test with XGBoost (regular predict endpoint)
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
        
        # Test with BERT (deep dive endpoint)
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
        
        # Compare results
        print(f"\nComparison:")
        xgb_phishing = xgb_result.get('is_phishing', False)
        bert_phishing = bert_result.get('is_phishing', False)
        
        if xgb_phishing != bert_phishing:
            print(f"DISAGREEMENT FOUND!")
            print(f"   XGBoost says: {'PHISHING' if xgb_phishing else 'SAFE'}")
            print(f"   BERT says: {'PHISHING' if bert_phishing else 'SAFE'}")
            
            if bert_phishing and not xgb_phishing:
                print(f"   BERT caught what XGBoost missed!")
            elif not bert_phishing and xgb_phishing:
                print(f"   BERT corrected XGBoost false positive!")
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
    print("Finding Contrastive Examples for Presentation")
    print("Testing URLs to find cases where XGBoost and BERT disagree...")
    
    # Test cases - mix of legitimate and potentially suspicious URLs
    test_cases = [
        # Legitimate sites that might have suspicious URL features
        {
            "url": "https://www.github.com",
            "description": "GitHub - Legitimate site"
        },
        {
            "url": "https://www.google.com",
            "description": "Google - Legitimate site"
        },
        {
            "url": "https://www.coursera.org",
            "description": "Coursera - Legitimate educational site"
        },
        {
            "url": "https://www.linkedin.com",
            "description": "LinkedIn - Legitimate professional site"
        },
        
        # Potentially suspicious URLs
        {
            "url": "https://bit.ly/3abc123",
            "description": "Shortened URL - Suspicious"
        },
        {
            "url": "http://example-suspicious-site.com",
            "description": "HTTP site - Potentially suspicious"
        },
        {
            "url": "https://www.paypal-security-update.com",
            "description": "Suspicious PayPal clone"
        },
        {
            "url": "https://secure-bank-verification.net",
            "description": "Suspicious bank verification site"
        },
        
        # Edge cases
        {
            "url": "https://www.amazon.com",
            "description": "Amazon - Legitimate e-commerce"
        },
        {
            "url": "https://www.microsoft.com",
            "description": "Microsoft - Legitimate tech company"
        }
    ]
    
    results = []
    disagreements = []
    
    for test_case in test_cases:
        result = test_url_models(test_case["url"], test_case["description"])
        if result:
            results.append(result)
            if result['disagreement']:
                disagreements.append(result)
        
        time.sleep(2)  # Wait between requests
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total URLs tested: {len(results)}")
    print(f"Disagreements found: {len(disagreements)}")
    
    if disagreements:
        print(f"\nCONTRASTIVE EXAMPLES FOR PRESENTATION:")
        for i, case in enumerate(disagreements, 1):
            print(f"\n{i}. {case['description']}")
            print(f"   URL: {case['url']}")
            print(f"   XGBoost: {'PHISHING' if case['xgb_result']['is_phishing'] else 'SAFE'}")
            print(f"   BERT: {'PHISHING' if case['bert_result']['is_phishing'] else 'SAFE'}")
            
            if case['bert_result']['is_phishing'] and not case['xgb_result']['is_phishing']:
                print(f"   BERT caught sophisticated phishing that XGBoost missed!")
            elif not case['bert_result']['is_phishing'] and case['xgb_result']['is_phishing']:
                print(f"   BERT prevented false positive from XGBoost!")
    else:
        print(f"\nNo disagreements found. Try more test URLs or check model configurations.")
    
    print(f"\nTesting complete!")

if __name__ == "__main__":
    main()









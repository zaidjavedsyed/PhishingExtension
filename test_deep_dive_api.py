#!/usr/bin/env python3
"""
Test script for the Deep Dive BERT API endpoint
This script tests the /deep-dive endpoint to ensure it's working correctly
"""

import requests
import json
import time

def test_deep_dive_api():
    """Test the deep dive API endpoint"""
    base_url = "http://localhost:8000"
    
    # Test URLs
    test_urls = [
        "https://www.google.com",  # Safe URL
        "https://www.github.com",  # Safe URL
        "https://example-phishing-site.com",  # Suspicious URL
    ]
    
    print("🔬 Testing Deep Dive BERT API...")
    print("=" * 50)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\nTest {i}: Testing URL - {url}")
        print("-" * 30)
        
        try:
            # Make request to deep dive endpoint
            response = requests.post(
                f"{base_url}/deep-dive",
                json={"url": url},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Success!")
                print(f"   Model: {result.get('model_type', 'Unknown')}")
                print(f"   Is Phishing: {result.get('is_phishing', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.2%}")
                print(f"   Probability: {result.get('probability', 0):.2%}")
                
                # Show BERT scores if available
                details = result.get('analysis_details', {})
                bert_scores = details.get('bert_scores', {})
                if bert_scores:
                    print(f"   BERT Phishing Score: {bert_scores.get('phishing', 0):.2%}")
                    print(f"   BERT Legitimate Score: {bert_scores.get('legitimate', 0):.2%}")
                
                # Show text content preview
                text_content = result.get('text_content', '')
                if text_content:
                    preview = text_content[:100] + "..." if len(text_content) > 100 else text_content
                    print(f"   Content Preview: {preview}")
                
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        # Wait between requests
        if i < len(test_urls):
            time.sleep(2)
    
    print("\n" + "=" * 50)
    print("🔬 Deep Dive API Test Complete!")

def test_health_endpoint():
    """Test the health endpoint to ensure the API is running"""
    base_url = "http://localhost:8000"
    
    print("🏥 Testing Health Endpoint...")
    
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is healthy!")
            print(f"   Status: {result.get('status', 'Unknown')}")
            print(f"   Model Loaded: {result.get('model_loaded', 'Unknown')}")
            print(f"   Version: {result.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Deep Dive API Tests")
    print("Make sure the backend server is running on http://localhost:8000")
    print()
    
    # Test health first
    if test_health_endpoint():
        print()
        test_deep_dive_api()
    else:
        print("\n❌ Cannot proceed with tests - API is not healthy")
        print("Please start the backend server first:")
        print("   python backend.py")









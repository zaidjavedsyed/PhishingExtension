#!/usr/bin/env python3
"""
Test Extension Functionality
This script tests the extension with various URLs to ensure it works correctly
"""

import requests
import time

def test_extension_urls():
    """Test various URLs to ensure the extension works correctly"""
    base_url = "http://localhost:8000"
    
    test_cases = [
        # Legitimate sites (should be allowed)
        ("http://www.google.com", "Should be ALLOWED"),
        ("http://github.com", "Should be ALLOWED"),
        ("http://microsoft.com", "Should be ALLOWED"),
        ("http://chrome://extensions/", "Should be ALLOWED"),
        
        # Phishing sites (should be blocked)
        ("http://appleid.apple.com-sa.pm", "Should be BLOCKED"),
        ("http://35.199.84.117", "Should be BLOCKED"),
        ("http://firebasestorage.googleapis.com", "Should be BLOCKED"),
        
        # Edge cases
        ("http://suspicious-site.com", "Should be BLOCKED"),
        ("http://bit.ly/suspicious-test", "Should be BLOCKED")
    ]
    
    print("Testing Extension URLs:")
    print("=" * 50)
    
    for url, expected in test_cases:
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"url": url},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = "BLOCKED" if result['is_phishing'] else "ALLOWED"
                confidence = result['confidence']
                
                status = "✅ CORRECT" if prediction in expected else "❌ WRONG"
                
                print(f"URL: {url}")
                print(f"Expected: {expected}")
                print(f"Predicted: {prediction}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Status: {status}")
                print("-" * 30)
            else:
                print(f"Error for {url}: Status {response.status_code}")
                
        except Exception as e:
            print(f"Error testing {url}: {e}")
        
        time.sleep(0.5)  # Small delay between requests

def main():
    """Main test function"""
    print("Extension Functionality Test")
    print("=" * 50)
    print("This test verifies that the extension correctly identifies:")
    print("- Legitimate sites as ALLOWED")
    print("- Phishing sites as BLOCKED")
    print("- Chrome internal pages as ALLOWED")
    print()
    
    test_extension_urls()
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print("=" * 50)
    print("If all tests show ✅ CORRECT, your extension is working properly!")
    print("If any tests show ❌ WRONG, check the extension logic.")

if __name__ == "__main__":
    main()


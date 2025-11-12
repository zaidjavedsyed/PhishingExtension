import pandas as pd
import requests
import json

# Read the CSV file
df = pd.read_csv('5.urldata.csv')

# Extract safe and phishing URLs
safe_urls = df[df['Label'] == 0]['Domain'].head(10).tolist()
phishing_urls = df[df['Label'] == 1]['Domain'].head(10).tolist()

print("="*80)
print("TESTING PHISHING DETECTOR EXTENSION")
print("="*80)

base_url = "http://localhost:8000"

test_urls = {
    "Safe Websites (Label=0)": safe_urls,
    "Phishing Websites (Label=1)": phishing_urls
}

results = {
    "safe_correct": 0,
    "safe_incorrect": 0,
    "phishing_correct": 0,
    "phishing_incorrect": 0
}

for category, urls in test_urls.items():
    print(f"\n{category}")
    print("-"*80)
    
    for url in urls:
        # Construct full URL
        test_url = f"https://{url}" if not url.startswith('http') else url
        
        try:
            # Test prediction
            response = requests.post(
                f"{base_url}/predict",
                json={"url": test_url},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('is_phishing', False)
                
                # Determine expected label
                expected_label = 0 if category == "Safe Websites (Label=0)" else 1
                expected_text = "PHISHING" if expected_label == 1 else "SAFE"
                predicted_text = "PHISHING" if prediction else "SAFE"
                
                # Check if correct
                is_correct = (prediction and expected_label == 1) or (not prediction and expected_label == 0)
                
                if is_correct:
                    status = "[CORRECT]"
                    if expected_label == 0:
                        results['safe_correct'] += 1
                    else:
                        results['phishing_correct'] += 1
                else:
                    status = "[WRONG!]"
                    if expected_label == 0:
                        results['safe_incorrect'] += 1
                    else:
                        results['phishing_incorrect'] += 1
                
                print(f"{status} {url:40} Expected: {expected_text:8} Got: {predicted_text:8} ({result.get('probability', 0):.1%})")
                
        except Exception as e:
            print(f"[ERROR] {url}: {e}")

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

total_safe = results['safe_correct'] + results['safe_incorrect']
total_phishing = results['phishing_correct'] + results['phishing_incorrect']

print(f"\nSafe Websites:")
print(f"  Correct: {results['safe_correct']}/{total_safe} ({results['safe_correct']/total_safe*100:.1f}%)")
print(f"  Incorrect: {results['safe_incorrect']}/{total_safe} ({results['safe_incorrect']/total_safe*100:.1f}%)")

print(f"\nPhishing Websites:")
print(f"  Correct: {results['phishing_correct']}/{total_phishing} ({results['phishing_correct']/total_phishing*100:.1f}%)")
print(f"  Incorrect: {results['phishing_incorrect']}/{total_phishing} ({results['phishing_incorrect']/total_phishing*100:.1f}%)")

total_correct = results['safe_correct'] + results['phishing_correct']
total_tested = total_safe + total_phishing
print(f"\nOverall Accuracy: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%)")

print("="*80)





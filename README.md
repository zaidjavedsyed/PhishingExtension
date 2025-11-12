# Phishing Website Detector Chrome Extension

A Chrome extension that uses machine learning to detect phishing websites and protect users from malicious URLs.

## Features

- **Machine Learning Detection**: Uses XGBoost model trained on 10,000 URLs
- **Real-time Analysis**: Analyzes URLs as you browse
- **Feature Extraction**: Extracts 16 different features from URLs
- **Visual Warnings**: Shows clear warnings for suspicious sites
- **Whitelist Support**: Allow trusted sites to bypass detection
- **Statistics Tracking**: Track sites checked and threats blocked
- **Settings Panel**: Customize detection behavior

## Installation

### Prerequisites

1. **Python 3.7+** installed on your system
2. **Chrome Browser** (latest version recommended)
3. **XGBoostClassifier.pickle.dat** model file in the project directory

### Step 1: Setup Backend API

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Convert Your XGBoost Model (if needed)**
   
   If you get XGBoost compatibility errors, run the conversion script:
   
   **On Windows:**
   ```cmd
   convert_and_run.bat
   ```
   
   **On Linux/Mac:**
   ```bash
   python convert_model.py
   python backend.py
   ```

3. **Start the FastAPI Backend**
   
   **On Windows:**
   ```cmd
   start_backend.bat
   ```
   
   **On Linux/Mac:**
   ```bash
   chmod +x start_backend.sh
   ./start_backend.sh
   ```
   
   **Or manually:**
   ```bash
   python backend.py
   ```

4. **Verify Backend is Running**
   - Open http://localhost:8000 in your browser
   - You should see the API health status
   - API documentation is available at http://localhost:8000/docs

### Step 2: Install Chrome Extension

1. **Open Chrome Extensions Page**
   - Go to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top right)

2. **Load the Extension**
   - Click "Load unpacked"
   - Select the folder containing the extension files
   - The extension should now appear in your extensions list

3. **Pin the Extension**
   - Click the puzzle piece icon in Chrome toolbar
   - Find "Phishing Website Detector" and click the pin icon

### Step 3: Test the System

1. **Test the API**
   ```bash
   python test_api.py
   ```

2. **Test the Extension**
   - Open the `test.html` file in your browser
   - Click on test links to see the extension in action
   - Check the extension popup for analysis results

## File Structure

```
phishing-detector-extension/
├── manifest.json              # Extension manifest
├── background.js              # Background service worker
├── content.js                 # Content script for page analysis
├── popup.html                 # Extension popup UI
├── popup.js                   # Popup functionality
├── featureExtractor.js        # Feature extraction functions
├── warning.html               # Warning page for phishing sites
├── backend.py                 # FastAPI backend server
├── requirements.txt           # Python dependencies
├── start_backend.sh          # Linux/Mac startup script
├── start_backend.bat          # Windows startup script
├── test_api.py               # API testing script
├── test.html                 # Extension testing page
└── XGBoostClassifier.pickle.dat  # Your trained XGBoost model
```

## How It Works

### 1. Backend API (FastAPI)
- **Model Loading**: Loads your trained XGBoost model from `XGBoostClassifier.pickle.dat`
- **Feature Extraction**: Extracts 16 features from URLs using the same logic as your training
- **Prediction**: Uses the actual XGBoost model to make predictions
- **API Endpoints**: Provides REST API for the Chrome extension to call

### 2. Feature Extraction
The system extracts 16 features from URLs:
- **Address Bar Features**: IP address, @ symbol, URL length, depth, redirection, HTTPS in domain, URL shortening, prefix/suffix
- **Domain Features**: DNS record, web traffic, domain age, domain end
- **HTML/JavaScript Features**: iFrame, mouse over, right click, web forwards

### 3. Machine Learning Model
- **XGBoost Classifier**: Uses your actual trained model (86.4% accuracy)
- **Real-time Prediction**: Makes predictions via API calls
- **Fallback System**: Falls back to rule-based detection if API is unavailable
### 4. Detection Process
1. User navigates to a website
2. Chrome extension sends URL to FastAPI backend
3. Backend extracts features and uses XGBoost model for prediction
4. Extension receives prediction results
5. If phishing detected with high confidence, user is redirected to warning page
6. User can choose to go back or proceed anyway

## Testing the Extension

### Test URLs

**Suspicious URLs (should trigger warnings):**
- `http://bit.ly/suspicious-link`
- `https://suspicious-site.com/verify-account`
- `http://192.168.1.1/login`
- `https://fake-bank.com@real-bank.com`

**Safe URLs (should not trigger warnings):**
- `https://www.google.com`
- `https://www.github.com`
- `https://www.stackoverflow.com`
- `https://www.wikipedia.org`

### Testing Steps

1. **Install the Extension**
   - Follow installation steps above

2. **Test Safe Sites**
   - Navigate to legitimate websites
   - Check that no warnings appear
   - Verify green "Safe Website" indicator in popup

3. **Test Suspicious Sites**
   - Navigate to suspicious URLs
   - Verify warning page appears
   - Test "Go Back" and "Proceed Anyway" buttons

4. **Test Popup Functionality**
   - Click extension icon
   - Verify analysis results display
   - Test scan button
   - Test settings toggles

5. **Test Whitelist Feature**
   - Add a site to whitelist
   - Navigate to that site
   - Verify it's not blocked

## Configuration

### Settings Available

- **Auto-scan enabled**: Automatically analyze URLs as you browse
- **Show notifications**: Display notifications for detected threats
- **Block suspicious sites**: Automatically block access to suspicious sites

### Whitelist Management

- Add trusted domains to bypass detection
- Useful for sites that trigger false positives
- Managed through popup interface

## Troubleshooting

### Common Issues

1. **XGBoost Model Compatibility Error**
   ```
   Check failed: header == serialisation_header_
   ```
   
   **Solution**: Run the model conversion script:
   ```bash
   python convert_model.py
   ```
   
   This will create compatible model formats or a fallback model.

2. **Extension Not Loading**
   - Check that all files are present
   - Verify manifest.json syntax
   - Check Chrome console for errors

3. **Backend API Not Starting**
   - Ensure Python 3.7+ is installed
   - Install dependencies: `pip install -r requirements.txt`
   - Check if port 8000 is available
   - Verify XGBoostClassifier.pickle.dat exists

4. **Warnings Not Appearing**
   - Check extension permissions
   - Verify background script is running
   - Check if site is in whitelist
   - Ensure backend API is running

5. **False Positives**
   - Use whitelist feature to bypass
   - Report false positives through popup
   - Adjust detection threshold if needed

### Debug Mode

1. **Enable Debug Logging**
   - Open Chrome DevTools
   - Go to Console tab
   - Look for extension messages

2. **Check Background Script**
   - Go to `chrome://extensions/`
   - Click "Inspect views: background page"
   - Check console for errors

3. **Test API Endpoints**
   ```bash
   # Test health endpoint
   curl http://localhost:8000/health
   
   # Test prediction endpoint
   curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"url": "https://www.google.com"}'
   ```

### Model Information

- **Original Model**: XGBoost Classifier (86.4% accuracy)
- **Fallback Model**: RandomForest Classifier (if XGBoost fails)
- **Model Formats**: JSON, Binary, Pickle
- **Features**: 16 extracted features from URLs

## Model Information

### Training Data
- **Dataset**: 5,000 legitimate URLs + 5,000 phishing URLs
- **Features**: 16 extracted features
- **Model**: XGBoost Classifier
- **Accuracy**: 86.4% on test set

### Feature Importance
- DNS Record: 15%
- Have IP: 15%
- URL Length: 12%
- TinyURL: 12%
- Domain Age: 12%
- Other features: 34%

## Future Improvements

1. **Model Conversion**: Convert pickle model to JavaScript-compatible format
2. **Real-time Updates**: Update model with new phishing patterns
3. **Enhanced Features**: Add more sophisticated content analysis
4. **User Feedback**: Implement feedback system for model improvement
5. **Performance**: Optimize for faster analysis

## Security Considerations

- Extension only analyzes URLs, doesn't collect personal data
- All analysis happens locally
- No data sent to external servers
- Whitelist stored locally in browser

## Support

For issues or questions:
1. Check troubleshooting section
2. Review Chrome extension documentation
3. Check console for error messages
4. Verify all files are present and correct

## License

This project is for educational purposes. Use responsibly and in accordance with applicable laws and regulations.
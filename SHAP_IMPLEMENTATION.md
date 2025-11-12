# SHAP Explainability Implementation Summary

## What Was Added:

### 1. Backend API Endpoint (`backend.py`)
- Added `/explain` endpoint for SHAP-based explainability
- Uses SHAP TreeExplainer for XGBoost model
- Returns feature importance with contribution values

### 2. Frontend UI Updates
- Added "Why This Prediction?" button in popup.html
- Added explainability section to display SHAP results
- Updated popup.js with explainability handlers
- Updated background.js with explainability message routing

### 3. SHAP Integration
- Feature importance calculation for all 16 URL features
- Sorted by absolute contribution magnitude
- Shows which features increase/decrease phishing risk
- Provides human-readable summary

## How It Works:

1. User clicks "Why This Prediction?" button
2. Frontend sends URL to background script
3. Background script calls `/explain` API endpoint
4. Backend uses SHAP to calculate feature importance
5. Results displayed showing:
   - Top contributing features (increases risk)
   - Protective features (decreases risk)
   - Feature values and SHAP contribution scores

## Key Features:

- **Transparency**: Users can see WHY the model made its prediction
- **Feature Attribution**: Each feature's contribution is quantified
- **Educational**: Helps users understand phishing indicators
- **Research Value**: Demonstrates model interpretability

## Example Output:

For URL: `https://secure-paypal-verification-center.com`

**Top Contributors:**
- URL_Length: increases risk by 6.91 (long URLs are suspicious)
- URL_Depth: increases risk by 0.75 (deep nesting is suspicious)
- Domain_End: decreases risk by 0.51 (certain domain extensions are safer)

This shows users exactly what features flagged this as phishing.


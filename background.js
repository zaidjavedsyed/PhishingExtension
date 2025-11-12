// Background script for Phishing Website Detector Chrome Extension
// This script handles URL analysis and model predictions

// URL Feature Extractor Class
class URLFeatureExtractor {
  constructor() {
    this.featureNames = [
      'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth', 'Redirection',
      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic',
      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over', 'Right_Click', 'Web_Forwards'
    ];
  }

  // Extract features from URL
  extractFeatures(url) {
    try {
      const features = [];
      
      // Basic URL features
      features.push(this.havingIP(url));
      features.push(this.haveAtSign(url));
      features.push(this.getLength(url));
      features.push(this.getDepth(url));
      features.push(this.redirection(url));
      features.push(this.httpDomain(url));
      features.push(this.tinyURL(url));
      features.push(this.prefixSuffix(url));
      
      // Default values for features that require external requests
      features.push(0); // DNS_Record
      features.push(0); // Web_Traffic
      features.push(0); // Domain_Age
      features.push(0); // Domain_End
      features.push(0); // iFrame
      features.push(0); // Mouse_Over
      features.push(0); // Right_Click
      features.push(0); // Web_Forwards
      
      return features;
    } catch (error) {
      console.error('Error extracting features:', error);
      return new Array(16).fill(0);
    }
  }

  havingIP(url) {
    try {
      const domain = url.split('//')[1].split('/')[0];
      return domain.replace(/\./g, '').match(/^\d+$/) ? 1 : 0;
    } catch {
      return 0;
    }
  }

  haveAtSign(url) {
    return url.includes('@') ? 1 : 0;
  }

  getLength(url) {
    return url.length > 75 ? 1 : 0;
  }

  getDepth(url) {
    return url.split('/').length > 4 ? 1 : 0;
  }

  redirection(url) {
    return url.includes('//') && url.split('//').length > 2 ? 1 : 0;
  }

  httpDomain(url) {
    return url.includes('https') ? 0 : 1;
  }

  tinyURL(url) {
    const tinyUrls = ['bit.ly', 'tinyurl', 'short', 't.co'];
    return tinyUrls.some(tiny => url.includes(tiny)) ? 1 : 0;
  }

  prefixSuffix(url) {
    try {
      const domain = url.split('//')[1].split('/')[0];
      return domain.includes('-') ? 1 : 0;
    } catch {
      return 0;
    }
  }
}

class PhishingDetector {
  constructor() {
    this.featureExtractor = new URLFeatureExtractor();
    this.modelLoaded = false;
    // Use Render API URL in production, fallback to localhost for development
    // You'll need to replace this with your actual Render URL after deployment
    this.apiBaseUrl = 'https://phishing-detection-api.onrender.com'; // Render API URL
    this.localApiUrl = 'http://localhost:8000'; // Local development URL
    this.suspiciousThreshold = 0.7; // Threshold for considering a site as phishing
    
    // Whitelist of trusted domains that should never be blocked
    this.trustedDomains = [
      'google.com', 'github.com', 'microsoft.com', 'amazon.com', 'facebook.com',
      'twitter.com', 'linkedin.com', 'youtube.com', 'netflix.com', 'apple.com',
      'chrome.google.com', 'chrome-extension://', 'chrome://', 'moz-extension://',
      'edge://', 'about:', 'chrome-extension://', 'chrome://extensions/',
      'chrome://settings/', 'chrome://newtab/', 'chrome://history/',
      'chrome://bookmarks/', 'chrome://downloads/', 'chrome://help/',
      'chrome://flags/', 'chrome://version/', 'chrome://gpu/',
      'chrome://net-internals/', 'chrome://components/', 'chrome://crashes/',
      'chrome://inspect/', 'chrome://process-internals/', 'chrome://system/',
      'chrome://user-actions/', 'chrome://webrtc-internals/', 'chrome://webrtc-logs/',
      'chrome://accessibility/', 'chrome://appcache-internals/', 'chrome://blob-internals/',
      'chrome://dino/', 'chrome://discards/', 'chrome://domain-reliability/',
      'chrome://download-internals/', 'chrome://extensions/', 'chrome://flags/',
      'chrome://gcm-internals/', 'chrome://histograms/', 'chrome://indexeddb-internals/',
      'chrome://inspect/', 'chrome://interstitials/', 'chrome://invalidations/',
      'chrome://local-state/', 'chrome://media-internals/', 'chrome://net-export/',
      'chrome://network-error/', 'chrome://network-errors/', 'chrome://ntp-tiles-internals/',
      'chrome://omnibox/', 'chrome://password-manager-internals/', 'chrome://policy/',
      'chrome://predictors/', 'chrome://print/', 'chrome://quota-internals/',
      'chrome://serviceworker-internals/', 'chrome://site-engagement/', 'chrome://surveys/',
      'chrome://sync-internals/', 'chrome://task-scheduler/', 'chrome://terms/',
      'chrome://thumbnails/', 'chrome://translate-internals/', 'chrome://ukm/',
      'chrome://usb-internals/', 'chrome://version/', 'chrome://web-app-internals/',
      'chrome://webrtc-internals/', 'chrome://webrtc-logs/', 'chrome://z-internals/'
    ];
    
    this.loadModel();
  }

  // Check if URL is in trusted domains whitelist
  isTrustedDomain(url) {
    try {
      const urlObj = new URL(url);
      const hostname = urlObj.hostname.toLowerCase();
      
      // Check for Chrome internal pages
      if (url.startsWith('chrome://') || url.startsWith('chrome-extension://') || 
          url.startsWith('moz-extension://') || url.startsWith('edge://') || 
          url.startsWith('about:')) {
        return true;
      }
      
      // Check for trusted domains
      return this.trustedDomains.some(trusted => {
        if (trusted.includes('://') || trusted.includes('about:')) {
          return url.startsWith(trusted);
        }
        return hostname === trusted || hostname.endsWith('.' + trusted);
      });
    } catch (error) {
      console.error('Error checking trusted domain:', error);
      return false;
    }
  }

  // Load the model by checking API health
  async loadModel() {
    // Try production API first, then fallback to localhost for development
    const urls = [this.apiBaseUrl, this.localApiUrl];
    
    for (const url of urls) {
      try {
        const response = await fetch(`${url}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        const healthData = await response.json();
        
        if (healthData.model_loaded) {
          // Update apiBaseUrl to the working one
          if (url !== this.apiBaseUrl) {
            this.apiBaseUrl = url;
            console.log('Switched to localhost API for development');
          }
          this.modelLoaded = true;
          console.log('✅ Phishing detection model loaded successfully from API');
          return;
        }
      } catch (error) {
        console.log(`Health check failed for ${url}:`, error.message);
        continue; // Try next URL
      }
    }
    
    console.error('❌ Error connecting to backend API');
    console.log('🔄 Falling back to rule-based detection');
    this.modelLoaded = false;
  }

  // Call the FastAPI backend to get prediction from XGBoost model
  async predictWithAPI(url) {
    // Try production API first, then fallback to localhost for development
    const urls = [this.apiBaseUrl, this.localApiUrl];
    
    for (const urlToTry of urls) {
      try {
        const response = await fetch(`${urlToTry}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ url: url }),
          signal: AbortSignal.timeout(10000) // 10 second timeout
        });

        if (!response.ok) {
          throw new Error(`API request failed: ${response.status}`);
        }

        const result = await response.json();
        
        // Update apiBaseUrl to the working one
        if (urlToTry !== this.apiBaseUrl) {
          this.apiBaseUrl = urlToTry;
          console.log('Switched to localhost API for development');
        }
        
        return {
          prediction: result.is_phishing ? 1 : 0,
          confidence: result.confidence,
          probability: result.probability,
          features: result.features,
          featureNames: result.feature_names
        };
      } catch (error) {
        console.log(`API prediction failed for ${urlToTry}:`, error.message);
        if (urlToTry === urls[urls.length - 1]) {
          // Last URL failed, throw error
          throw error;
        }
        continue; // Try next URL
      }
    }
  }

  // Fallback prediction function based on feature patterns
  // This mimics the XGBoost model behavior using rule-based logic
  predict(features) {
    let phishingScore = 0;
    let totalWeight = 0;

    // Feature weights based on typical XGBoost feature importance
    const featureWeights = [
      0.15, // Have_IP
      0.10, // Have_At
      0.12, // URL_Length
      0.08, // URL_Depth
      0.10, // Redirection
      0.08, // https_Domain
      0.12, // TinyURL
      0.10, // Prefix/Suffix
      0.15, // DNS_Record
      0.10, // Web_Traffic
      0.12, // Domain_Age
      0.10, // Domain_End
      0.08, // iFrame
      0.08, // Mouse_Over
      0.08, // Right_Click
      0.08  // Web_Forwards
    ];

    // Calculate weighted score
    for (let i = 0; i < features.length && i < featureWeights.length; i++) {
      phishingScore += features[i] * featureWeights[i];
      totalWeight += featureWeights[i];
    }

    // Normalize score
    const normalizedScore = totalWeight > 0 ? phishingScore / totalWeight : 0;
    
    // Convert to prediction (1 = phishing, 0 = legitimate)
    const prediction = normalizedScore >= this.suspiciousThreshold ? 1 : 0;
    const confidence = Math.abs(normalizedScore - 0.5) * 2; // Confidence based on distance from 0.5

    return {
      prediction: prediction,
      confidence: Math.min(confidence, 1.0),
      score: normalizedScore
    };
  }

  // Analyze URL for phishing
  async analyzeURL(url) {
    try {
      // First check if URL is trusted - if so, always allow
      if (this.isTrustedDomain(url)) {
        console.log('✅ Trusted domain, allowing:', url);
        return {
          isPhishing: false,
          confidence: 0.95,
          features: new Array(16).fill(0),
          usingAPI: false,
          reason: 'trusted_domain'
        };
      }

      let result;
      let features;
      let usingAPI = false;

      // Try to use API first
      if (this.modelLoaded) {
        try {
          const apiResult = await this.predictWithAPI(url);
          result = apiResult;
          features = apiResult.features;
          usingAPI = true;
          console.log('✅ Using XGBoost model via API');
        } catch (apiError) {
          console.warn('⚠️ API failed, falling back to rule-based detection:', apiError);
          // Fall back to rule-based detection
          features = this.featureExtractor.extractFeatures(url);
          result = this.predict(features);
          usingAPI = false;
        }
      } else {
        // Use rule-based detection
        features = this.featureExtractor.extractFeatures(url);
        result = this.predict(features);
        usingAPI = false;
        console.log('🔄 Using rule-based detection');
      }
      
      return {
        url: url,
        features: features,
        isPhishing: result.prediction === 1,
        confidence: result.confidence,
        score: result.score || result.probability || 0,
        timestamp: Date.now(),
        usingAPI: usingAPI,
        modelType: usingAPI ? 'XGBoost' : 'Rule-based'
      };
    } catch (error) {
      console.error('Error analyzing URL:', error);
      return {
        url: url,
        isPhishing: false,
        confidence: 0,
        error: error.message,
        usingAPI: false,
        modelType: 'Error'
      };
    }
  }

  // Check if URL should be blocked
  shouldBlockURL(url) {
    // Add your blocking logic here
    // For now, we'll block URLs with high phishing scores
    return this.analyzeURL(url).then(result => {
      return result.isPhishing && result.confidence > 0.8;
    });
  }

  // Perform deep dive analysis using BERT model
  async performDeepDiveAnalysis(url) {
    try {
      console.log('🔬 Performing deep dive analysis for:', url);
      
      // Call the BERT deep dive endpoint
      // Try production API first, then fallback to localhost
      const urls = [this.apiBaseUrl, this.localApiUrl];
      let response;
      let workingUrl = null;
      
      for (const urlToTry of urls) {
        try {
          response = await fetch(`${urlToTry}/deep-dive`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url }),
            signal: AbortSignal.timeout(15000) // 15 second timeout for deep dive
          });
          workingUrl = urlToTry;
          break;
        } catch (error) {
          if (urlToTry === urls[urls.length - 1]) throw error;
          continue;
        }
      }
      
      if (workingUrl && workingUrl !== this.apiBaseUrl) {
        this.apiBaseUrl = workingUrl;
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('🔬 Deep dive analysis result:', result);
      
      return result;
      
    } catch (error) {
      console.error('Error performing deep dive analysis:', error);
      throw new Error(`Deep dive analysis failed: ${error.message}`);
    }
  }

  async performExplainabilityAnalysis(url) {
    try {
      console.log('📊 Performing explainability analysis for:', url);
      
      // Try production API first, then fallback to localhost
      const urls = [this.apiBaseUrl, this.localApiUrl];
      let response;
      let workingUrl = null;
      
      for (const urlToTry of urls) {
        try {
          response = await fetch(`${urlToTry}/explain`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url }),
            signal: AbortSignal.timeout(15000) // 15 second timeout
          });
          workingUrl = urlToTry;
          break;
        } catch (error) {
          if (urlToTry === urls[urls.length - 1]) throw error;
          continue;
        }
      }
      
      if (workingUrl && workingUrl !== this.apiBaseUrl) {
        this.apiBaseUrl = workingUrl;
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('📊 Explainability analysis result:', result);
      
      return result;
      
    } catch (error) {
      console.error('Error performing explainability analysis:', error);
      throw new Error(`Explainability analysis failed: ${error.message}`);
    }
  }
}

// Initialize the detector
const phishingDetector = new PhishingDetector();

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzeURL') {
    phishingDetector.analyzeURL(request.url)
      .then(result => {
        sendResponse({ success: true, result: result });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'deepDiveAnalysis') {
    phishingDetector.performDeepDiveAnalysis(request.url)
      .then(result => {
        sendResponse({ success: true, result: result });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'explainPrediction') {
    phishingDetector.performExplainabilityAnalysis(request.url)
      .then(result => {
        sendResponse({ success: true, data: result });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'shouldBlock') {
    phishingDetector.shouldBlockURL(request.url)
      .then(shouldBlock => {
        sendResponse({ shouldBlock: shouldBlock });
      })
      .catch(error => {
        sendResponse({ shouldBlock: false, error: error.message });
      });
    return true; // Keep message channel open for async response
  }
});

// Listen for tab updates to analyze URLs
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && tab.url) {
    try {
      const result = await phishingDetector.analyzeURL(tab.url);
      
      // Only block if it's phishing AND confidence is very high AND it's not a trusted domain
      if (result.isPhishing && result.confidence > 0.9 && result.reason !== 'trusted_domain') {
        console.log('🚨 Blocking suspicious URL:', tab.url, 'Confidence:', result.confidence);
        
        // Check if tab still exists before updating
        try {
          await chrome.tabs.get(tabId);
          // Tab exists, safe to update
          chrome.tabs.update(tabId, {
            url: chrome.runtime.getURL('warning.html') + '?url=' + encodeURIComponent(tab.url)
          });
        } catch (tabError) {
          console.log('Tab no longer exists, skipping redirect:', tabError);
        }
      } else {
        console.log('✅ Allowing URL:', tab.url, 'Reason:', result.reason || 'low_confidence');
      }
    } catch (error) {
      console.error('Error in tab update listener:', error);
    }
  }
});

// Note: webRequest blocking is not available in Manifest V3
// We rely on tab updates and content scripts for detection

// Store analysis results
chrome.storage.local.set({ 'phishingDetector': phishingDetector });

console.log('Phishing Website Detector background script loaded');

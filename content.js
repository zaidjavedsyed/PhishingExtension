// Content script for Phishing Website Detector Chrome Extension
// This script runs on web pages and provides additional analysis

// Feature extraction functions (copied from featureExtractor.js)
class URLFeatureExtractor {
  constructor() {
    // URL shortening services regex pattern
    this.shorteningServices = /bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net/;
  }

  // Helper function to parse URL
  parseURL(url) {
    try {
      return new URL(url);
    } catch (e) {
      return null;
    }
  }

  // 1. Check for IP address in URL
  havingIP(url) {
    try {
      const urlObj = this.parseURL(url);
      if (!urlObj) return 1;
      
      const hostname = urlObj.hostname;
      // Check if hostname is an IP address
      const ipRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
      return ipRegex.test(hostname) ? 1 : 0;
    } catch (e) {
      return 1;
    }
  }

  // 2. Check for @ symbol in URL
  haveAtSign(url) {
    return url.includes('@') ? 1 : 0;
  }

  // 3. Check URL length (>= 54 characters is suspicious)
  getLength(url) {
    return url.length >= 54 ? 1 : 0;
  }

  // 4. Calculate URL depth (number of path segments)
  getDepth(url) {
    try {
      const urlObj = this.parseURL(url);
      if (!urlObj) return 0;
      
      const pathSegments = urlObj.pathname.split('/').filter(segment => segment.length > 0);
      return pathSegments.length;
    } catch (e) {
      return 0;
    }
  }

  // 5. Check for redirection '//' in URL
  redirection(url) {
    const pos = url.lastIndexOf('//');
    if (pos > 6) {
      if (pos > 7) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }

  // 6. Check for 'https' in domain name
  httpDomain(url) {
    try {
      const urlObj = this.parseURL(url);
      if (!urlObj) return 0;
      
      return urlObj.hostname.includes('https') ? 1 : 0;
    } catch (e) {
      return 0;
    }
  }

  // 7. Check for URL shortening services
  tinyURL(url) {
    return this.shorteningServices.test(url) ? 1 : 0;
  }

  // 8. Check for prefix/suffix separated by '-' in domain
  prefixSuffix(url) {
    try {
      const urlObj = this.parseURL(url);
      if (!urlObj) return 0;
      
      return urlObj.hostname.includes('-') ? 1 : 0;
    } catch (e) {
      return 0;
    }
  }

  // 9. DNS Record availability (simplified - assume 0 for legitimate sites)
  dnsRecord(url) {
    try {
      const urlObj = this.parseURL(url);
      if (!urlObj) return 1;
      
      // Check if it's a well-known domain
      const wellKnownDomains = [
        'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
        'apple.com', 'netflix.com', 'youtube.com', 'twitter.com',
        'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com'
      ];
      
      const hostname = urlObj.hostname.toLowerCase();
      return wellKnownDomains.some(domain => hostname.includes(domain)) ? 0 : 1;
    } catch (e) {
      return 1;
    }
  }

  // 10. Web Traffic (simplified - assume legitimate sites have traffic)
  webTraffic(url) {
    try {
      const urlObj = this.parseURL(url);
      if (!urlObj) return 1;
      
      const hostname = urlObj.hostname.toLowerCase();
      const suspiciousPatterns = [
        'bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly',
        'short.ly', 'is.gd', 'v.gd', 'qr.ae'
      ];
      
      return suspiciousPatterns.some(pattern => hostname.includes(pattern)) ? 1 : 0;
    } catch (e) {
      return 1;
    }
  }

  // 11. Domain Age (simplified - assume new domains are suspicious)
  domainAge(url) {
    try {
      const urlObj = this.parseURL(url);
      if (!urlObj) return 1;
      
      const hostname = urlObj.hostname.toLowerCase();
      const establishedDomains = [
        'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
        'apple.com', 'netflix.com', 'youtube.com', 'twitter.com',
        'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
        'wikipedia.org', 'reddit.com', 'pinterest.com', 'tumblr.com'
      ];
      
      return establishedDomains.some(domain => hostname.includes(domain)) ? 0 : 1;
    } catch (e) {
      return 1;
    }
  }

  // 12. Domain End (simplified - assume legitimate domains have longer expiration)
  domainEnd(url) {
    return this.domainAge(url);
  }

  // 13. iFrame Redirection (simplified - would need to fetch page content)
  iframe(url) {
    return 0; // Simplified for content script
  }

  // 14. Mouse Over (simplified - would need to fetch page content)
  mouseOver(url) {
    return 0; // Simplified for content script
  }

  // 15. Right Click (simplified - would need to fetch page content)
  rightClick(url) {
    return 0; // Simplified for content script
  }

  // 16. Web Forwards (simplified - would need to check redirects)
  forwarding(url) {
    return 0; // Simplified for content script
  }

  // Main function to extract all features
  extractFeatures(url) {
    const features = [];
    
    // Address bar based features (8 features)
    features.push(this.havingIP(url));
    features.push(this.haveAtSign(url));
    features.push(this.getLength(url));
    features.push(this.getDepth(url));
    features.push(this.redirection(url));
    features.push(this.httpDomain(url));
    features.push(this.tinyURL(url));
    features.push(this.prefixSuffix(url));
    
    // Domain based features (4 features)
    features.push(this.dnsRecord(url));
    features.push(this.webTraffic(url));
    features.push(this.domainAge(url));
    features.push(this.domainEnd(url));
    
    // HTML & JavaScript based features (4 features)
    features.push(this.iframe(url));
    features.push(this.mouseOver(url));
    features.push(this.rightClick(url));
    features.push(this.forwarding(url));
    
    return features;
  }
}

class ContentAnalyzer {
  constructor() {
    this.featureExtractor = new URLFeatureExtractor();
    this.analysisComplete = false;
    this.init();
  }

  init() {
    // Analyze the current page
    this.analyzeCurrentPage();
    
    // Listen for navigation changes
    this.setupNavigationListener();
    
    // Add visual indicators
    this.addVisualIndicators();
  }

  async analyzeCurrentPage() {
    try {
      const currentURL = window.location.href;
      
      // Send URL to background script for analysis
      chrome.runtime.sendMessage({
        action: 'analyzeURL',
        url: currentURL
      }, (response) => {
        if (response && response.success) {
          this.handleAnalysisResult(response.result);
        }
      });

      // Perform additional content-based analysis
      await this.analyzePageContent();
      
    } catch (error) {
      console.error('Error analyzing current page:', error);
    }
  }

  async analyzePageContent() {
    try {
      // Check for suspicious content patterns
      const suspiciousPatterns = this.detectSuspiciousContent();
      
      // Check for suspicious JavaScript
      const suspiciousJS = this.detectSuspiciousJavaScript();
      
      // Check for form security
      const formSecurity = this.checkFormSecurity();
      
      const contentAnalysis = {
        suspiciousPatterns: suspiciousPatterns,
        suspiciousJavaScript: suspiciousJS,
        formSecurity: formSecurity,
        timestamp: Date.now()
      };

      // Store analysis results
      this.contentAnalysis = contentAnalysis;
      
      // Show warning if suspicious content detected
      if (suspiciousPatterns.length > 0 || suspiciousJS.length > 0) {
        this.showContentWarning(contentAnalysis);
      }
      
    } catch (error) {
      console.error('Error analyzing page content:', error);
    }
  }

  detectSuspiciousContent() {
    const suspiciousPatterns = [];
    
    // Check for common phishing keywords
    const phishingKeywords = [
      'verify your account', 'update your information', 'confirm your identity',
      'urgent action required', 'security alert', 'account suspended',
      'click here immediately', 'limited time offer', 'congratulations you won',
      'free money', 'act now', 'expires today'
    ];
    
    const pageText = document.body.innerText.toLowerCase();
    
    phishingKeywords.forEach(keyword => {
      if (pageText.includes(keyword.toLowerCase())) {
        suspiciousPatterns.push({
          type: 'phishing_keyword',
          keyword: keyword,
          severity: 'medium'
        });
      }
    });
    
    // Check for suspicious forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
      const inputs = form.querySelectorAll('input[type="password"], input[type="email"]');
      if (inputs.length > 0) {
        // Check if form is submitted to suspicious domain
        const action = form.action ? String(form.action) : '';
        if (action && !this.isTrustedDomain(action)) {
          suspiciousPatterns.push({
            type: 'suspicious_form',
            action: action,
            severity: 'high'
          });
        }
      }
    });
    
    return suspiciousPatterns;
  }

  detectSuspiciousJavaScript() {
    const suspiciousJS = [];
    
    // Check for suspicious JavaScript patterns
    const scripts = document.querySelectorAll('script');
    scripts.forEach(script => {
      const content = script.textContent || script.innerHTML;
      
      // Check for obfuscated code
      if (this.isObfuscatedCode(content)) {
        suspiciousJS.push({
          type: 'obfuscated_code',
          severity: 'high'
        });
      }
      
      // Check for suspicious functions
      const suspiciousFunctions = [
        'eval(', 'document.write(', 'innerHTML', 'outerHTML',
        'setTimeout', 'setInterval', 'onload', 'onclick'
      ];
      
      suspiciousFunctions.forEach(func => {
        if (content.includes(func)) {
          suspiciousJS.push({
            type: 'suspicious_function',
            function: func,
            severity: 'medium'
          });
        }
      });
    });
    
    return suspiciousJS;
  }

  checkFormSecurity() {
    const forms = document.querySelectorAll('form');
    const securityIssues = [];
    
    forms.forEach(form => {
      // Check if form uses HTTPS
      const formAction = form.action ? String(form.action) : '';
      if (formAction && !formAction.startsWith('https://')) {
        securityIssues.push({
          type: 'insecure_form',
          issue: 'Form submission not using HTTPS',
          severity: 'high'
        });
      }
      
      // Check for password fields without proper security
      const passwordFields = form.querySelectorAll('input[type="password"]');
      passwordFields.forEach(field => {
        if (!field.hasAttribute('autocomplete') || field.getAttribute('autocomplete') !== 'off') {
          securityIssues.push({
            type: 'password_security',
            issue: 'Password field may be cached by browser',
            severity: 'medium'
          });
        }
      });
    });
    
    return securityIssues;
  }

  isObfuscatedCode(code) {
    // Simple heuristic to detect obfuscated code
    const suspiciousPatterns = [
      /\\x[0-9a-fA-F]{2}/g, // Hex encoded characters
      /\\u[0-9a-fA-F]{4}/g, // Unicode encoded characters
      /String\.fromCharCode/g, // Character code conversion
      /eval\s*\(/g, // Eval usage
      /document\.write\s*\(/g // Document write
    ];
    
    let suspiciousCount = 0;
    suspiciousPatterns.forEach(pattern => {
      const matches = code.match(pattern);
      if (matches) {
        suspiciousCount += matches.length;
      }
    });
    
    return suspiciousCount > 5; // Threshold for considering code obfuscated
  }

  isTrustedDomain(url) {
    const trustedDomains = [
      'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
      'apple.com', 'netflix.com', 'youtube.com', 'twitter.com',
      'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
      'wikipedia.org', 'reddit.com', 'pinterest.com', 'tumblr.com',
      'paypal.com', 'ebay.com', 'amazon.com', 'apple.com'
    ];
    
    try {
      const domain = new URL(url).hostname.toLowerCase();
      return trustedDomains.some(trusted => domain.includes(trusted));
    } catch (e) {
      return false;
    }
  }

  handleAnalysisResult(result) {
    this.analysisComplete = true;
    
    if (result.isPhishing) {
      this.showPhishingWarning(result);
    } else {
      this.showSafeIndicator(result);
    }
  }

  showPhishingWarning(result) {
    // Create warning overlay
    const warningOverlay = document.createElement('div');
    warningOverlay.id = 'phishing-warning-overlay';
    warningOverlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(255, 0, 0, 0.9);
      z-index: 999999;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: Arial, sans-serif;
    `;
    
    const warningContent = document.createElement('div');
    warningContent.style.cssText = `
      background: white;
      padding: 30px;
      border-radius: 10px;
      text-align: center;
      max-width: 500px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    `;
    
    warningContent.innerHTML = `
      <h2 style="color: #d32f2f; margin-bottom: 20px;">⚠️ Phishing Website Detected</h2>
      <p style="margin-bottom: 20px; font-size: 16px;">
        This website has been identified as potentially malicious. 
        Confidence: ${Math.round(result.confidence * 100)}%
      </p>
      <div style="margin-bottom: 20px;">
        <button id="go-back-btn" style="
          background: #d32f2f;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 5px;
          cursor: pointer;
          margin-right: 10px;
        ">Go Back</button>
        <button id="proceed-btn" style="
          background: #666;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 5px;
          cursor: pointer;
        ">Proceed Anyway</button>
      </div>
      <p style="font-size: 12px; color: #666;">
        This warning is provided by the Phishing Website Detector extension.
      </p>
    `;
    
    warningOverlay.appendChild(warningContent);
    document.body.appendChild(warningOverlay);
    
    // Add event listeners
    document.getElementById('go-back-btn').addEventListener('click', () => {
      window.history.back();
      warningOverlay.remove();
    });
    
    document.getElementById('proceed-btn').addEventListener('click', () => {
      warningOverlay.remove();
    });
  }

  showSafeIndicator(result) {
    // Add safe indicator to page
    const safeIndicator = document.createElement('div');
    safeIndicator.id = 'phishing-safe-indicator';
    safeIndicator.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      background: #4caf50;
      color: white;
      padding: 10px 15px;
      border-radius: 5px;
      font-family: Arial, sans-serif;
      font-size: 12px;
      z-index: 1000;
      box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    `;
    
    safeIndicator.innerHTML = `
      ✅ Safe Website<br>
      <small>Confidence: ${Math.round(result.confidence * 100)}%</small>
    `;
    
    document.body.appendChild(safeIndicator);
    
    // Remove indicator after 5 seconds
    setTimeout(() => {
      if (safeIndicator.parentNode) {
        safeIndicator.remove();
      }
    }, 5000);
  }

  showContentWarning(analysis) {
    // Show content-based warning
    const contentWarning = document.createElement('div');
    contentWarning.id = 'content-warning';
    contentWarning.style.cssText = `
      position: fixed;
      top: 50px;
      right: 10px;
      background: #ff9800;
      color: white;
      padding: 15px;
      border-radius: 5px;
      font-family: Arial, sans-serif;
      font-size: 12px;
      z-index: 1000;
      max-width: 300px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    `;
    
    let warningText = '⚠️ Suspicious content detected:<br>';
    analysis.suspiciousPatterns.forEach(pattern => {
      warningText += `• ${pattern.type}<br>`;
    });
    analysis.suspiciousJavaScript.forEach(js => {
      warningText += `• ${js.type}<br>`;
    });
    
    contentWarning.innerHTML = warningText;
    document.body.appendChild(contentWarning);
    
    // Remove warning after 10 seconds
    setTimeout(() => {
      if (contentWarning.parentNode) {
        contentWarning.remove();
      }
    }, 10000);
  }

  addVisualIndicators() {
    // Add visual indicators for suspicious elements
    const suspiciousElements = document.querySelectorAll('a[href*="bit.ly"], a[href*="tinyurl"], a[href*="goo.gl"]');
    suspiciousElements.forEach(element => {
      element.style.border = '2px solid #ff9800';
      element.style.borderRadius = '3px';
      element.title = 'Shortened URL detected - proceed with caution';
    });
  }

  setupNavigationListener() {
    // Listen for navigation changes
    let currentURL = window.location.href;
    
    const checkNavigation = () => {
      if (window.location.href !== currentURL) {
        currentURL = window.location.href;
        this.analyzeCurrentPage();
      }
    };
    
    // Check for navigation changes periodically
    setInterval(checkNavigation, 1000);
    
    // Also listen for popstate events
    window.addEventListener('popstate', checkNavigation);
  }
}

// Initialize the content analyzer when the page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new ContentAnalyzer();
  });
} else {
  new ContentAnalyzer();
}

console.log('Phishing Website Detector content script loaded');

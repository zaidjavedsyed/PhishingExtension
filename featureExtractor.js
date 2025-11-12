// Feature extraction functions based on URLFeatureExtraction.py
// These functions extract the same features used to train the XGBoost model

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
    // In a real implementation, this would check DNS records
    // For now, we'll use a simplified approach
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
    // In a real implementation, this would check Alexa rankings
    // For now, we'll use a simplified approach based on domain recognition
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
    // In a real implementation, this would check WHOIS data
    // For now, we'll use a simplified approach
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
    // In a real implementation, this would check WHOIS expiration data
    // For now, we'll use the same logic as domainAge
    return this.domainAge(url);
  }

  // 13. iFrame Redirection (simplified - would need to fetch page content)
  iframe(url) {
    // This would require fetching the page content and checking for iframes
    // For now, we'll return 0 (legitimate) as we can't fetch content in background
    return 0;
  }

  // 14. Mouse Over (simplified - would need to fetch page content)
  mouseOver(url) {
    // This would require fetching the page content and checking for onmouseover
    // For now, we'll return 0 (legitimate) as we can't fetch content in background
    return 0;
  }

  // 15. Right Click (simplified - would need to fetch page content)
  rightClick(url) {
    // This would require fetching the page content and checking for right-click disabling
    // For now, we'll return 0 (legitimate) as we can't fetch content in background
    return 0;
  }

  // 16. Web Forwards (simplified - would need to check redirects)
  forwarding(url) {
    // This would require checking the number of redirects
    // For now, we'll return 0 (legitimate) as we can't check redirects in background
    return 0;
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
    // Note: These are simplified since we can't fetch page content in background
    features.push(this.iframe(url));
    features.push(this.mouseOver(url));
    features.push(this.rightClick(url));
    features.push(this.forwarding(url));
    
    return features;
  }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = URLFeatureExtractor;
}

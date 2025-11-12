// Popup script for Phishing Website Detector Chrome Extension
// This script handles the popup UI interactions and communicates with the background script

class PopupManager {
  constructor() {
    this.currentTab = null;
    this.analysisResult = null;
    this.stats = {
      sitesChecked: 0,
      threatsBlocked: 0
    };
    
    this.init();
  }

  async init() {
    try {
      // Get current tab
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      this.currentTab = tabs[0];
      
      // Load settings and stats
      await this.loadSettings();
      await this.loadStats();
      
      // Analyze current page
      await this.analyzeCurrentPage();
      
      // Setup event listeners
      this.setupEventListeners();
      
      // Show main content
      this.showMainContent();
      
    } catch (error) {
      console.error('Error initializing popup:', error);
      this.showError('Failed to initialize popup');
    }
  }

  async loadSettings() {
    try {
      const result = await chrome.storage.sync.get([
        'autoScanEnabled',
        'notificationsEnabled',
        'blockSuspiciousSites'
      ]);
      
      // Set default values if not found
      this.settings = {
        autoScanEnabled: result.autoScanEnabled !== undefined ? result.autoScanEnabled : true,
        notificationsEnabled: result.notificationsEnabled !== undefined ? result.notificationsEnabled : true,
        blockSuspiciousSites: result.blockSuspiciousSites !== undefined ? result.blockSuspiciousSites : true
      };
      
      // Update UI
      this.updateSettingsUI();
      
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  }

  async loadStats() {
    try {
      const result = await chrome.storage.local.get(['stats']);
      if (result.stats) {
        this.stats = result.stats;
      }
      
      // Update stats UI
      this.updateStatsUI();
      
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  }

  async analyzeCurrentPage() {
    if (!this.currentTab || !this.currentTab.url) {
      this.showError('No active tab found');
      return;
    }

    try {
      // Update current URL display
      document.getElementById('current-url').textContent = this.currentTab.url;
      
      // Send analysis request to background script
      const response = await chrome.runtime.sendMessage({
        action: 'analyzeURL',
        url: this.currentTab.url
      });

      if (response && response.success) {
        this.analysisResult = response.result;
        this.updateAnalysisUI();
      } else {
        this.showError('Failed to analyze URL');
      }
      
    } catch (error) {
      console.error('Error analyzing current page:', error);
      this.showError('Error analyzing current page');
    }
  }

  async performDeepDiveAnalysis() {
    if (!this.currentTab || !this.currentTab.url) {
      this.showError('No active tab found');
      return;
    }

    try {
      // Show loading state
      const deepDiveBtn = document.getElementById('deep-dive-btn');
      const originalText = deepDiveBtn.textContent;
      deepDiveBtn.textContent = '🔬 Analyzing...';
      deepDiveBtn.disabled = true;

      // Show deep dive section
      const deepDiveSection = document.getElementById('deep-dive-section');
      deepDiveSection.classList.remove('hidden');

      // Show loading in deep dive content
      const deepDiveContent = document.getElementById('deep-dive-content');
      deepDiveContent.innerHTML = `
        <div class="deep-dive-result">
          <div class="deep-dive-score">🔬 Running BERT Analysis...</div>
          <div class="deep-dive-details">This may take a few moments as we analyze the page content with our advanced transformer model.</div>
        </div>
      `;

      // Send deep dive request to background script
      const response = await chrome.runtime.sendMessage({
        action: 'deepDiveAnalysis',
        url: this.currentTab.url
      });

      if (response && response.success) {
        this.deepDiveResult = response.result;
        this.updateDeepDiveUI();
      } else {
        this.showDeepDiveError('Failed to perform deep dive analysis');
      }
      
    } catch (error) {
      console.error('Error performing deep dive analysis:', error);
      this.showDeepDiveError('Error performing deep dive analysis');
    } finally {
      // Restore button state
      const deepDiveBtn = document.getElementById('deep-dive-btn');
      deepDiveBtn.textContent = '🔬 Deep Dive Analysis';
      deepDiveBtn.disabled = false;
    }
  }

  async performExplainabilityAnalysis() {
    if (!this.currentTab) {
      console.error('No current tab available');
      return;
    }

    try {
      const explainBtn = document.getElementById('explain-btn');
      explainBtn.textContent = '📊 Analyzing...';
      explainBtn.disabled = true;

      // Show explainability section
      const explainSection = document.getElementById('explain-section');
      explainSection.classList.remove('hidden');

      // Show loading
      const explainContent = document.getElementById('explain-content');
      explainContent.innerHTML = `
        <div class="deep-dive-result">
          <div class="deep-dive-score">📊 Calculating Feature Importance...</div>
          <div class="deep-dive-details">Using SHAP values to explain the prediction</div>
        </div>
      `;

      // Send message to background script
      const response = await chrome.runtime.sendMessage({
        action: 'explainPrediction',
        url: this.currentTab.url
      });

      if (response && response.success) {
        this.updateExplainabilityUI(response.data);
      } else {
        this.showExplainabilityError(response?.error || 'Failed to get explanation');
      }

    } catch (error) {
      console.error('Explainability analysis error:', error);
      this.showExplainabilityError('Failed to perform explainability analysis');
    } finally {
      const explainBtn = document.getElementById('explain-btn');
      explainBtn.textContent = '📊 Why This Prediction?';
      explainBtn.disabled = false;
    }
  }

  updateExplainabilityUI(data) {
    const explainContent = document.getElementById('explain-content');
    
    let html = `
      <div class="deep-dive-result">
        <div class="deep-dive-score">${data.summary}</div>
      </div>
      <div class="deep-dive-result">
        <div class="deep-dive-score">📊 Feature Analysis:</div>
    `;

    // Display top 5 features
    const topFeatures = data.feature_importance.slice(0, 5);
    topFeatures.forEach(feature => {
      const contribution = feature.contribution > 0 ? '🔴' : '🟢';
      const direction = feature.contribution > 0 ? 'increases' : 'reduces';
      const impact = feature.contribution > 0 ? '+' : '';
      html += `
        <div class="deep-dive-details" style="margin-top: 5px;">
          ${contribution} <strong>${feature.feature}</strong> ${direction} risk by ${impact}${feature.importance.toFixed(2)}
        </div>
      `;
    });

    html += '</div>';

    explainContent.innerHTML = html;
  }

  showExplainabilityError(message) {
    const explainContent = document.getElementById('explain-content');
    explainContent.innerHTML = `
      <div class="deep-dive-result">
        <div class="deep-dive-score" style="color: #dc3545;">❌ Error</div>
        <div class="deep-dive-details">${message}</div>
      </div>
    `;
  }

  updateAnalysisUI() {
    if (!this.analysisResult) return;

    const statusIndicator = document.getElementById('status-indicator');
    const statusIcon = statusIndicator.querySelector('.status-icon');
    const statusText = statusIndicator.querySelector('.status-text');
    
    // Update status indicator
    statusIndicator.className = 'status-indicator';
    
    if (this.analysisResult.isPhishing) {
      statusIndicator.classList.add('status-danger');
      statusIcon.textContent = '🚨';
      statusText.textContent = 'Phishing Detected';
    } else {
      statusIndicator.classList.add('status-safe');
      statusIcon.textContent = '✅';
      statusText.textContent = 'Safe Website';
    }

    // Add model type indicator
    if (this.analysisResult.modelType) {
      const modelIndicator = document.createElement('div');
      modelIndicator.style.cssText = `
        font-size: 10px;
        color: #666;
        margin-top: 5px;
        padding: 2px 6px;
        background: #f0f0f0;
        border-radius: 3px;
        display: inline-block;
      `;
      modelIndicator.textContent = `Model: ${this.analysisResult.modelType}`;
      statusIndicator.appendChild(modelIndicator);
    }

    // Update feature list
    this.updateFeatureList();
  }

  updateDeepDiveUI() {
    if (!this.deepDiveResult) return;

    const deepDiveContent = document.getElementById('deep-dive-content');
    
    // Determine result styling
    const isPhishing = this.deepDiveResult.is_phishing;
    const confidence = this.deepDiveResult.confidence;
    const probability = this.deepDiveResult.probability;
    
    const resultClass = isPhishing ? 'status-danger' : 'status-safe';
    const resultIcon = isPhishing ? '🚨' : '✅';
    const resultText = isPhishing ? 'Phishing Detected' : 'Safe Website';
    
    // Format confidence and probability
    const confidencePercent = (confidence * 100).toFixed(1);
    const probabilityPercent = (probability * 100).toFixed(1);
    
    // Get analysis details
    const details = this.deepDiveResult.analysis_details || {};
    const bertScores = details.bert_scores || {};
    const phishingScore = (bertScores.phishing * 100).toFixed(1);
    const legitimateScore = (bertScores.legitimate * 100).toFixed(1);
    
    deepDiveContent.innerHTML = `
      <div class="deep-dive-result ${resultClass}">
        <div class="deep-dive-score">
          ${resultIcon} ${resultText} (${confidencePercent}% confidence)
        </div>
        <div class="deep-dive-details">
          <strong>BERT Analysis Results:</strong><br>
          • Phishing Score: ${phishingScore}%<br>
          • Legitimate Score: ${legitimateScore}%<br>
          • Model: ${this.deepDiveResult.model_type}<br>
          • Analysis Time: ${details.analysis_timestamp ? new Date(details.analysis_timestamp).toLocaleTimeString() : 'Unknown'}
        </div>
      </div>
      
      ${this.deepDiveResult.text_content ? `
        <div class="deep-dive-result">
          <div class="deep-dive-score">📄 Analyzed Content</div>
          <div class="deep-dive-text">${this.deepDiveResult.text_content}</div>
        </div>
      ` : ''}
      
      <div class="deep-dive-result">
        <div class="deep-dive-score">📊 Analysis Summary</div>
        <div class="deep-dive-details">
          The BERT transformer model analyzed both the URL and page content to provide this assessment. 
          This advanced AI model considers linguistic patterns, context, and semantic meaning to detect 
          phishing attempts with higher accuracy than traditional methods.
        </div>
      </div>
    `;
  }

  showDeepDiveError(message) {
    const deepDiveContent = document.getElementById('deep-dive-content');
    deepDiveContent.innerHTML = `
      <div class="deep-dive-result status-danger">
        <div class="deep-dive-score">❌ Analysis Failed</div>
        <div class="deep-dive-details">${message}</div>
      </div>
    `;
  }

  updateFeatureList() {
    const featureList = document.getElementById('feature-list');
    featureList.innerHTML = '';

    if (!this.analysisResult || !this.analysisResult.features) return;

    const featureNames = [
      'IP Address in URL',
      'At Symbol (@)',
      'URL Length',
      'URL Depth',
      'Redirection',
      'HTTPS in Domain',
      'URL Shortening',
      'Prefix/Suffix',
      'DNS Record',
      'Web Traffic',
      'Domain Age',
      'Domain End',
      'iFrame',
      'Mouse Over',
      'Right Click',
      'Web Forwards'
    ];

    this.analysisResult.features.forEach((feature, index) => {
      if (index < featureNames.length) {
        const li = document.createElement('li');
        li.className = 'feature-item';
        
        const icon = document.createElement('span');
        icon.className = 'feature-icon';
        
        if (feature === 1) {
          icon.classList.add('feature-danger');
          icon.textContent = '⚠️';
        } else {
          icon.classList.add('feature-safe');
          icon.textContent = '✅';
        }
        
        const text = document.createElement('span');
        text.textContent = featureNames[index];
        
        li.appendChild(icon);
        li.appendChild(text);
        featureList.appendChild(li);
      }
    });
  }

  updateSettingsUI() {
    // Update toggle states
    document.getElementById('auto-scan-toggle').classList.toggle('active', this.settings.autoScanEnabled);
    document.getElementById('notifications-toggle').classList.toggle('active', this.settings.notificationsEnabled);
    document.getElementById('block-toggle').classList.toggle('active', this.settings.blockSuspiciousSites);
  }

  updateStatsUI() {
    document.getElementById('sites-checked').textContent = this.stats.sitesChecked;
    document.getElementById('threats-blocked').textContent = this.stats.threatsBlocked;
  }

  setupEventListeners() {
    // Scan button
    document.getElementById('scan-btn').addEventListener('click', () => {
      this.analyzeCurrentPage();
    });

    // Deep dive button
    document.getElementById('deep-dive-btn').addEventListener('click', () => {
      this.performDeepDiveAnalysis();
    });

    // Explainability button
    document.getElementById('explain-btn').addEventListener('click', () => {
      this.performExplainabilityAnalysis();
    });

    // History button
    document.getElementById('history-btn').addEventListener('click', () => {
      this.showHistory();
    });

    // Whitelist button
    document.getElementById('whitelist-btn').addEventListener('click', () => {
      this.whitelistCurrentSite();
    });

    // Report button
    document.getElementById('report-btn').addEventListener('click', () => {
      this.reportFalsePositive();
    });

    // Settings toggles
    document.getElementById('auto-scan-toggle').addEventListener('click', () => {
      this.toggleSetting('autoScanEnabled');
    });

    document.getElementById('notifications-toggle').addEventListener('click', () => {
      this.toggleSetting('notificationsEnabled');
    });

    document.getElementById('block-toggle').addEventListener('click', () => {
      this.toggleSetting('blockSuspiciousSites');
    });
  }

  async toggleSetting(settingName) {
    this.settings[settingName] = !this.settings[settingName];
    
    // Save to storage
    await chrome.storage.sync.set({ [settingName]: this.settings[settingName] });
    
    // Update UI
    this.updateSettingsUI();
    
    // Send setting change to background script
    chrome.runtime.sendMessage({
      action: 'updateSetting',
      setting: settingName,
      value: this.settings[settingName]
    });
  }

  async whitelistCurrentSite() {
    if (!this.currentTab || !this.currentTab.url) return;

    try {
      const url = new URL(this.currentTab.url);
      const domain = url.hostname;
      
      // Get current whitelist
      const result = await chrome.storage.sync.get(['whitelist']);
      const whitelist = result.whitelist || [];
      
      // Add domain to whitelist if not already present
      if (!whitelist.includes(domain)) {
        whitelist.push(domain);
        await chrome.storage.sync.set({ whitelist: whitelist });
        
        // Show success message
        this.showMessage('Site added to whitelist', 'success');
        
        // Re-analyze page
        setTimeout(() => this.analyzeCurrentPage(), 1000);
      } else {
        this.showMessage('Site already in whitelist', 'info');
      }
      
    } catch (error) {
      console.error('Error whitelisting site:', error);
      this.showMessage('Error whitelisting site', 'error');
    }
  }

  async reportFalsePositive() {
    if (!this.analysisResult) return;

    try {
      // Get current reports
      const result = await chrome.storage.local.get(['falsePositives']);
      const reports = result.falsePositives || [];
      
      // Add new report
      reports.push({
        url: this.currentTab.url,
        timestamp: Date.now(),
        analysisResult: this.analysisResult
      });
      
      await chrome.storage.local.set({ falsePositives: reports });
      
      this.showMessage('False positive reported', 'success');
      
    } catch (error) {
      console.error('Error reporting false positive:', error);
      this.showMessage('Error reporting false positive', 'error');
    }
  }

  showHistory() {
    // Open history page in new tab
    chrome.tabs.create({
      url: chrome.runtime.getURL('history.html')
    });
  }

  showMainContent() {
    document.getElementById('loading-section').classList.add('hidden');
    document.getElementById('main-content').classList.remove('hidden');
  }

  showError(message) {
    const statusIndicator = document.getElementById('status-indicator');
    statusIndicator.className = 'status-indicator status-danger';
    statusIndicator.querySelector('.status-icon').textContent = '❌';
    statusIndicator.querySelector('.status-text').textContent = message;
  }

  showMessage(message, type = 'info') {
    // Create temporary message element
    const messageEl = document.createElement('div');
    messageEl.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      right: 10px;
      padding: 10px;
      border-radius: 5px;
      color: white;
      font-size: 12px;
      text-align: center;
      z-index: 1000;
    `;
    
    switch (type) {
      case 'success':
        messageEl.style.background = '#28a745';
        break;
      case 'error':
        messageEl.style.background = '#dc3545';
        break;
      case 'warning':
        messageEl.style.background = '#ffc107';
        messageEl.style.color = '#333';
        break;
      default:
        messageEl.style.background = '#007bff';
    }
    
    messageEl.textContent = message;
    document.body.appendChild(messageEl);
    
    // Remove message after 3 seconds
    setTimeout(() => {
      if (messageEl.parentNode) {
        messageEl.remove();
      }
    }, 3000);
  }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new PopupManager();
});

// Handle popup close
window.addEventListener('beforeunload', () => {
  // Save any pending data
  chrome.storage.local.set({ lastPopupClose: Date.now() });
});

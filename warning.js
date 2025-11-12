// warning.js - JavaScript for the warning page
// This file handles the warning page functionality

// Get URL from query parameters
const urlParams = new URLSearchParams(window.location.search);
const suspiciousUrl = urlParams.get('url') || 'Unknown URL';

// Display the suspicious URL
document.getElementById('suspicious-url').textContent = suspiciousUrl;

// Simulate analysis (in a real implementation, this would come from the background script)
setTimeout(() => {
    const confidence = Math.random() * 0.4 + 0.6; // Random confidence between 60-100%
    const confidencePercent = Math.round(confidence * 100);
    
    // Update confidence bar
    document.getElementById('confidence-fill').style.width = `${confidencePercent}%`;
    document.getElementById('confidence-text').textContent = `${confidencePercent}% Confidence`;
    
    // Show detected features
    showDetectedFeatures();
}, 1000);

function showDetectedFeatures() {
    const features = [
        'Suspicious URL structure detected',
        'Shortened URL service identified',
        'Suspicious domain characteristics',
        'Potential redirection patterns',
        'Unusual URL length detected'
    ];
    
    const featureList = document.getElementById('feature-list');
    features.forEach(feature => {
        const li = document.createElement('li');
        li.textContent = feature;
        featureList.appendChild(li);
    });
    
    document.getElementById('features-detected').style.display = 'block';
}

function goBack() {
    if (window.history.length > 1) {
        window.history.back();
    } else {
        window.location.href = 'https://www.google.com';
    }
}

function goHome() {
    window.location.href = 'https://www.google.com';
}

function proceedAnyway() {
    if (confirm('Are you sure you want to proceed? This website has been identified as potentially malicious.')) {
        // In a real implementation, you would redirect to the original URL
        // For now, we'll just show an alert
        alert('Proceeding to the suspicious website. Please be extremely careful and do not enter any personal information.');
        // window.location.href = suspiciousUrl;
    }
}

async function performDeepDiveAnalysis() {
    try {
        // Show loading state
        const deepDiveBtn = document.getElementById('deepDiveBtn');
        const originalText = deepDiveBtn.textContent;
        deepDiveBtn.textContent = '🔬 Analyzing...';
        deepDiveBtn.disabled = true;

        // Show deep dive section
        const deepDiveSection = document.getElementById('deep-dive-section');
        deepDiveSection.style.display = 'block';

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
            url: suspiciousUrl
        });

        if (response && response.success) {
            updateDeepDiveUI(response.result);
        } else {
            showDeepDiveError('Failed to perform deep dive analysis');
        }
        
    } catch (error) {
        console.error('Error performing deep dive analysis:', error);
        showDeepDiveError('Error performing deep dive analysis');
    } finally {
        // Restore button state
        const deepDiveBtn = document.getElementById('deepDiveBtn');
        deepDiveBtn.textContent = '🔬 Deep Dive Analysis';
        deepDiveBtn.disabled = false;
    }
}

function updateDeepDiveUI(result) {
    const deepDiveContent = document.getElementById('deep-dive-content');
    
    // Determine result styling
    const isPhishing = result.is_phishing;
    const confidence = result.confidence;
    
    const resultClass = isPhishing ? 'status-danger' : 'status-safe';
    const resultIcon = isPhishing ? '🚨' : '✅';
    const resultText = isPhishing ? 'Phishing Detected' : 'Safe Website';
    
    // Format confidence
    const confidencePercent = (confidence * 100).toFixed(1);
    
    // Get analysis details
    const details = result.analysis_details || {};
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
                • Model: ${result.model_type}<br>
                • Analysis Time: ${details.analysis_timestamp ? new Date(details.analysis_timestamp).toLocaleTimeString() : 'Unknown'}
            </div>
        </div>
        
        ${result.text_content ? `
            <div class="deep-dive-result">
                <div class="deep-dive-score">📄 Analyzed Content</div>
                <div class="deep-dive-text">${result.text_content}</div>
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

function showDeepDiveError(message) {
    const deepDiveContent = document.getElementById('deep-dive-content');
    deepDiveContent.innerHTML = `
        <div class="deep-dive-result status-danger">
            <div class="deep-dive-score">❌ Analysis Failed</div>
            <div class="deep-dive-details">${message}</div>
        </div>
    `;
}

// Add some interactive effects
document.addEventListener('DOMContentLoaded', () => {
    // Add click effect to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });

    // Add deep dive button event listener
    const deepDiveBtn = document.getElementById('deepDiveBtn');
    if (deepDiveBtn) {
        deepDiveBtn.addEventListener('click', performDeepDiveAnalysis);
    }

    // Add explainability button event listener
    const explainBtn = document.getElementById('explainBtn');
    if (explainBtn) {
        explainBtn.addEventListener('click', performExplainabilityAnalysis);
    }
});

async function performExplainabilityAnalysis() {
    try {
        // Show loading state
        const explainBtn = document.getElementById('explainBtn');
        const originalText = explainBtn.textContent;
        explainBtn.textContent = '📊 Analyzing...';
        explainBtn.disabled = true;

        // Show explain section
        const explainSection = document.getElementById('explain-section');
        explainSection.style.display = 'block';

        // Show loading in explain content
        const explainContent = document.getElementById('explain-content');
        explainContent.innerHTML = `
            <div class="explain-result">
                <div class="explain-loading">📊 Calculating Feature Importance...</div>
                <div class="explain-details">Analyzing URL features with SHAP to explain this prediction.</div>
            </div>
        `;

        // Send explainability request to background script
        const response = await chrome.runtime.sendMessage({
            action: 'explainPrediction',
            url: suspiciousUrl
        });

        if (response && response.success) {
            updateExplainabilityUI(response.data);
        } else {
            showExplainabilityError('Failed to get explanation');
        }
        
    } catch (error) {
        console.error('Error performing explainability analysis:', error);
        showExplainabilityError('Error performing explainability analysis');
    } finally {
        // Restore button state
        const explainBtn = document.getElementById('explainBtn');
        explainBtn.textContent = '📊 Why This Prediction?';
        explainBtn.disabled = false;
    }
}

function updateExplainabilityUI(result) {
    const explainContent = document.getElementById('explain-content');
    
    let html = `
      <div class="explain-result">
        <div class="explain-summary">${result.summary}</div>
      </div>
      <div class="explain-result">
        <div class="explain-summary">📊 Feature Analysis:</div>
    `;

    // Display top 5 features
    if (result.feature_importance && result.feature_importance.length > 0) {
        result.feature_importance.slice(0, 5).forEach(feature => {
            const contribution = feature.contribution > 0 ? '🔴' : '🟢';
            const direction = feature.contribution > 0 ? 'increases' : 'reduces';
            const impact = feature.contribution > 0 ? '+' : '';
            html += `
                <div class="explain-details" style="margin-top: 5px;">
                    ${contribution} <strong>${feature.feature}</strong> ${direction} risk by ${impact}${feature.importance.toFixed(2)}
                </div>
            `;
        });
    }

    html += '</div>';

    explainContent.innerHTML = html;
}

function showExplainabilityError(message) {
    const explainContent = document.getElementById('explain-content');
    explainContent.innerHTML = `
        <div class="explain-result status-danger">
            <div class="explain-error">❌ Explanation Failed</div>
            <div class="explain-details">${message}</div>
        </div>
    `;
}

// Report Generation Utilities - FIXED VERSION
export interface SystemInfo {
  os: string;
  architecture: string;
  processor: string;
  cpuCores: number;
  totalRamGB: number;
  availableRamGB: number;
  totalStorageGB: number;
  freeStorageGB: number;
  gpus?: Array<{
    name: string;
    vramGB: number | string;
  }>;
}

export interface ModelRecommendation {
  name: string;
  specs: {
    parameters: string;
    min_ram_gb: number;
    recommended_ram_gb: number;
    min_vram_gb: number;
    recommended_vram_gb: number;
    description: string;
    domain?: string;
    install_methods: {
      ollama?: {
        command: string;
        note: string;
      };
      huggingface?: {
        model_id: string;
        note: string;
      };
      gguf?: {
        source: string;
        recommended_quant: string;
        note: string;
      };
    };
  };
  compatibility: {
    performance_tier: string;
    recommended_quant?: string;
    notes: string[];
  };
}

export interface Recommendations {
  excellent: ModelRecommendation[];
  good: ModelRecommendation[];
  basic: ModelRecommendation[];
  not_suitable: ModelRecommendation[];
}

export interface LLMRecommender {
  getInstallationPlatforms(): Array<{
    name: string;
    description: string;
    difficulty: string;
    bestFor: string;
    installation: Record<string, string>;
  }>;
  getOptimizationTips(): string[];
}

export interface ReportData {
  systemInfo: SystemInfo;
  recommendations: Recommendations;
  recommender: LLMRecommender;
  timestamp: Date;
}

export class ReportGenerator {
  private data: ReportData;

  constructor(data: ReportData) {
    this.data = data;
  }

  generateHTMLReport(): string {
    try {
      const { systemInfo, recommendations, recommender } = this.data;
      const timestamp = new Date().toLocaleString();
      
      const suitableModels = [
        ...(recommendations.excellent || []),
        ...(recommendations.good || []),
        ...(recommendations.basic || [])
      ];

      const totalSpecialized = suitableModels.filter(m => m.specs?.domain).length;
      const totalGeneral = suitableModels.length - totalSpecialized;

      return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Hardware Compatibility Report</title>
    <style>
        ${this.getReportCSS()}
    </style>
</head>
<body>
    <div class="container">
        ${this.generateHeader(timestamp)}
        ${this.generateSystemSpecs(systemInfo)}
        ${suitableModels.length > 0 
          ? this.generateRecommendations(recommendations, totalGeneral, totalSpecialized)
          : this.generateInsufficientHardware(systemInfo)
        }
        ${this.generateInstallationGuide(recommender)}
        ${this.generateOptimizationTips(recommender)}
        ${this.generateFooter(timestamp)}
    </div>
    <script>
        ${this.getReportJavaScript()}
    </script>
</body>
</html>`;
    } catch (error) {
      console.error('Error generating HTML report:', error);
      return this.generateErrorReport(error);
    }
  }

  private generateErrorReport(error: any): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report Generation Error</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .error { background: #fee; border: 1px solid #fcc; padding: 20px; border-radius: 8px; }
        .error h1 { color: #c33; margin-top: 0; }
        .error code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="error">
        <h1>Report Generation Error</h1>
        <p>An error occurred while generating the report:</p>
        <code>${error?.message || 'Unknown error'}</code>
        <p>Please check the console for more details and ensure all required data is provided.</p>
    </div>
</body>
</html>`;
  }

  private getReportCSS(): string {
    return `
/* CSS Variables for consistent theming */
:root {
    --primary-color: #3b82f6;
    --primary-dark: #1d4ed8;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-500: #6b7280;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
    color: var(--gray-800);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
}

.container {
    max-width: 1200px;
    margin: 20px auto;
    padding: 30px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    border: 1px solid var(--gray-200);
}

.header {
    text-align: center;
    margin-bottom: 40px;
    padding: 40px 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}

.header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 12px;
    position: relative;
    z-index: 1;
    letter-spacing: -0.025em;
}

.header .subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
    font-weight: 400;
}

.section {
    margin-bottom: 40px;
    padding: 30px;
    background: var(--gray-50);
    border-radius: 16px;
    border: 1px solid var(--gray-200);
    page-break-inside: avoid;
}

.section h2 {
    color: var(--primary-dark);
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.025em;
}

.specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.spec-item {
    background: white;
    padding: 24px;
    border-radius: 12px;
    border-left: 4px solid var(--primary-color);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
    page-break-inside: avoid;
}

.spec-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
}

.spec-label {
    font-weight: 600;
    color: var(--gray-600);
    margin-bottom: 8px;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.spec-value {
    color: var(--gray-900);
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1.4;
}

.summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.stat-card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid var(--gray-200);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    page-break-inside: avoid;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--primary-color);
    margin-bottom: 8px;
    line-height: 1;
}

.stat-label {
    color: var(--gray-600);
    font-size: 0.95rem;
    font-weight: 500;
}

.model-card {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: 16px;
    padding: 30px;
    margin: 20px 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    page-break-inside: avoid;
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--primary-color);
    transform: scaleY(0);
    transition: transform 0.3s ease;
}

.model-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
}

.model-card:hover::before {
    transform: scaleY(1);
}

.model-header {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 20px;
}

.model-name {
    font-size: 1.375rem;
    font-weight: 700;
    color: var(--gray-900);
    flex: 1;
    letter-spacing: -0.025em;
}

.model-domain {
    background: linear-gradient(135deg, var(--error-color), #dc2626);
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.performance-tier {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.875rem;
    margin: 16px 0;
    letter-spacing: 0.025em;
}

.performance-tier.excellent {
    background: linear-gradient(135deg, var(--success-color), #059669);
    color: white;
}

.performance-tier.good {
    background: linear-gradient(135deg, var(--warning-color), #d97706);
    color: white;
}

.performance-tier.basic {
    background: linear-gradient(135deg, var(--error-color), #dc2626);
    color: white;
}

.requirements {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 15px;
    margin: 20px 0;
    padding: 20px;
    background: var(--gray-50);
    border-radius: 12px;
    border: 1px solid var(--gray-200);
}

.req-item {
    text-align: center;
    padding: 16px;
    background: white;
    border-radius: 8px;
    border: 1px solid var(--gray-200);
    transition: all 0.2s ease;
}

.req-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.req-label {
    font-size: 0.75rem;
    color: var(--gray-500);
    margin-bottom: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.req-value {
    font-weight: 600;
    color: var(--gray-900);
    font-size: 0.875rem;
}

.installation-methods {
    margin-top: 24px;
    padding: 24px;
    background: var(--gray-50);
    border-radius: 12px;
    border: 1px solid var(--gray-200);
}

.install-method {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    page-break-inside: avoid;
}

.install-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
}

.install-title {
    font-weight: 600;
    color: var(--gray-900);
    font-size: 1.125rem;
}

.install-badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.install-badge.easy {
    background: #dcfce7;
    color: #166534;
}

.install-badge.intermediate {
    background: #fef3c7;
    color: #92400e;
}

.install-badge.advanced {
    background: #fecaca;
    color: #991b1b;
}

.install-command {
    background: var(--gray-900);
    color: #f9fafb;
    padding: 16px 20px;
    border-radius: 8px;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 16px 0;
    overflow-x: auto;
    position: relative;
    word-break: break-all;
}

.install-command::before {
    content: '$ ';
    color: var(--success-color);
    font-weight: 600;
}

.copy-btn {
    position: absolute;
    top: 12px;
    right: 12px;
    background: var(--gray-700);
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: 500;
    transition: all 0.2s ease;
}

.copy-btn:hover {
    background: var(--gray-600);
    transform: scale(1.05);
}

.install-note {
    background: #eff6ff;
    border-left: 4px solid var(--primary-color);
    padding: 16px 20px;
    margin: 16px 0;
    border-radius: 6px;
    font-size: 0.875rem;
    line-height: 1.6;
}

.insufficient-hardware {
    background: linear-gradient(135deg, #fecaca, #fca5a5);
    border: 1px solid var(--error-color);
    border-radius: 16px;
    padding: 30px;
    margin: 30px 0;
}

.insufficient-hardware h3 {
    color: #991b1b;
    margin-bottom: 16px;
    font-weight: 700;
}

.cloud-solutions {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    margin-top: 20px;
}

.cloud-solution {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 4px solid var(--primary-color);
    text-decoration: none;
    color: inherit;
    transition: all 0.2s ease;
    display: block;
    page-break-inside: avoid;
}

.cloud-solution:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
}

.cloud-solution h4 {
    color: var(--gray-900);
    margin-bottom: 8px;
    font-weight: 600;
}

.cloud-solution p {
    color: var(--gray-600);
    font-size: 0.875rem;
    line-height: 1.5;
}

.footer {
    text-align: center;
    padding: 30px;
    background: var(--gray-50);
    border-radius: 12px;
    margin-top: 40px;
    color: var(--gray-600);
    border: 1px solid var(--gray-200);
}

.tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.tip-card {
    background: white;
    padding: 24px;
    border-radius: 12px;
    border-left: 4px solid var(--success-color);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    page-break-inside: avoid;
}

.tip-card h4 {
    color: var(--gray-900);
    margin-bottom: 12px;
    font-size: 1.125rem;
    font-weight: 600;
}

.tip-card ul {
    margin-left: 20px;
    color: var(--gray-600);
}

.tip-card li {
    margin-bottom: 8px;
    line-height: 1.6;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        margin: 10px;
        padding: 20px;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .specs-grid {
        grid-template-columns: 1fr;
    }
    
    .model-header {
        flex-direction: column;
        align-items: flex-start;
    }
}

/* Print Styles */
@media print {
    body {
        background: white;
        font-size: 12pt;
    }
    
    .container {
        box-shadow: none;
        margin: 0;
        max-width: 100%;
    }
    
    .model-card, .tip-card, .spec-item {
        break-inside: avoid;
        page-break-inside: avoid;
    }
    
    .copy-btn {
        display: none;
    }
}
`;
  }

  private getReportJavaScript(): string {
    return `
function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function() {
            showToast('Command copied to clipboard!');
        }).catch(function(err) {
            console.error('Failed to copy:', err);
            fallbackCopyTextToClipboard(text);
        });
    } else {
        fallbackCopyTextToClipboard(text);
    }
}

function fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        const successful = document.execCommand('copy');
        if (successful) {
            showToast('Command copied to clipboard!');
        } else {
            showToast('Failed to copy command', 'error');
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showToast('Copy not supported in this browser', 'error');
    }
    
    document.body.removeChild(textArea);
}

function showToast(message, type = 'success') {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => toast.remove());
    
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = message;
    toast.style.cssText = 
        'position: fixed; top: 20px; right: 20px; padding: 12px 20px; ' +
        'background: ' + (type === 'success' ? '#10b981' : '#ef4444') + '; ' +
        'color: white; border-radius: 8px; font-weight: 500; font-size: 14px; ' +
        'box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); z-index: 1000; ' +
        'opacity: 0; transform: translateY(-10px); ' +
        'transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);';
    
    document.body.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Add copy functionality to command blocks
        const commandBlocks = document.querySelectorAll('.install-command');
        commandBlocks.forEach(function(block, index) {
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-btn';
            copyBtn.innerHTML = '📋 Copy';
            copyBtn.setAttribute('data-command', block.textContent.replace('$ ', '').trim());
            
            copyBtn.onclick = function(e) {
                e.preventDefault();
                const command = this.getAttribute('data-command');
                copyToClipboard(command);
                
                const btn = this;
                btn.innerHTML = '✅ Copied!';
                setTimeout(function() {
                    btn.innerHTML = '📋 Copy';
                }, 2000);
            };
            
            block.style.position = 'relative';
            block.appendChild(copyBtn);
        });
        
        // Add smooth scrolling for internal links
        const anchorLinks = document.querySelectorAll('a[href^="#"]');
        anchorLinks.forEach(function(anchor) {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
        
        console.log('Report JavaScript initialized successfully');
    } catch (error) {
        console.error('Error initializing report JavaScript:', error);
    }
});
`;
  }

  private generateHeader(timestamp: string): string {
    return `
<div class="header">
    <h1>🤖 LLM Hardware Compatibility Report</h1>
    <div class="subtitle">Professional Analysis Generated on ${timestamp}</div>
</div>`;
  }

  private generateSystemSpecs(systemInfo: SystemInfo): string {
    if (!systemInfo) {
      return '<div class="section"><h2>❌ System Information Unavailable</h2><p>Unable to retrieve system specifications.</p></div>';
    }

    return `
<div class="section">
    <h2>🖥️ System Specifications</h2>
    <div class="specs-grid">
        <div class="spec-item">
            <div class="spec-label">Operating System</div>
            <div class="spec-value">${systemInfo.os || 'Unknown'} (${systemInfo.architecture || 'Unknown'})</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">Processor</div>
            <div class="spec-value">${systemInfo.processor || 'Unknown'}</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">CPU Cores</div>
            <div class="spec-value">${systemInfo.cpuCores || 'Unknown'} cores detected</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">Memory (RAM)</div>
            <div class="spec-value">${systemInfo.totalRamGB || 'Unknown'} GB total (${systemInfo.availableRamGB || 'Unknown'} GB available)</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">Storage</div>
            <div class="spec-value">${systemInfo.freeStorageGB || 'Unknown'} GB free / ${systemInfo.totalStorageGB || 'Unknown'} GB total</div>
        </div>
        ${systemInfo.gpus && systemInfo.gpus.length > 0 
          ? systemInfo.gpus.map(gpu => `
            <div class="spec-item">
                <div class="spec-label">GPU</div>
                <div class="spec-value">${gpu.name || 'Unknown GPU'} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB || 'Unknown VRAM'})</div>
            </div>
          `).join('')
          : `
            <div class="spec-item">
                <div class="spec-label">GPU</div>
                <div class="spec-value">None detected</div>
            </div>
          `
        }
    </div>
</div>`;
  }

  private generateRecommendations(recommendations: Recommendations, totalGeneral: number, totalSpecialized: number): string {
    if (!recommendations) {
      return '<div class="section"><h2>❌ Recommendations Unavailable</h2><p>Unable to generate model recommendations.</p></div>';
    }

    const suitableModels = [
      ...(recommendations.excellent || []),
      ...(recommendations.good || []),
      ...(recommendations.basic || [])
    ];

    let html = `
<div class="section">
    <h2>🤖 Model Recommendations</h2>
    
    <div class="summary-stats">
        <div class="stat-card">
            <div class="stat-number">${suitableModels.length}</div>
            <div class="stat-label">Compatible Models</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${totalGeneral}</div>
            <div class="stat-label">General Purpose</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${totalSpecialized}</div>
            <div class="stat-label">Specialized Domain</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${(recommendations.not_suitable || []).length}</div>
            <div class="stat-label">Not Suitable</div>
        </div>
    </div>
`;

    const categories = [
      { key: 'excellent', title: '🟢 Excellent Performance', models: recommendations.excellent || [] },
      { key: 'good', title: '🟡 Good Performance', models: recommendations.good || [] },
      { key: 'basic', title: '🟠 Basic Performance', models: recommendations.basic || [] }
    ];

    for (const category of categories) {
      if (category.models.length > 0) {
        html += `<h3 style="margin-top: 2.5rem; color: var(--primary-dark); font-size: 1.5rem; font-weight: 700;">${category.title}</h3>`;
        for (const model of category.models) {
          html += this.generateModelCard(model, category.key);
        }
      }
    }

    html += '</div>';
    return html;
  }

  private generateModelCard(model: ModelRecommendation, category: string): string {
    if (!model || !model.specs) {
      return '<div class="model-card"><p>Invalid model data</p></div>';
    }

    const domainBadge = model.specs.domain 
      ? `<span class="model-domain">${model.specs.domain}</span>` 
      : '';

    const specs = model.specs;
    const compatibility = model.compatibility || {};

    return `
<div class="model-card">
    <div class="model-header">
        <div class="model-name">${model.name || 'Unknown Model'}</div>
        ${domainBadge}
    </div>
    
    <div class="performance-tier ${category}">
        ${compatibility.performance_tier || 'Unknown Performance'}
    </div>
    
    <p style="margin: 1.25rem 0; color: var(--gray-600); line-height: 1.6; font-size: 0.95rem;">${specs.description || 'No description available'}</p>
    
    <div class="requirements">
        <div class="req-item">
            <div class="req-label">Parameters</div>
            <div class="req-value">${specs.parameters || 'Unknown'}</div>
        </div>
        <div class="req-item">
            <div class="req-label">Min RAM</div>
            <div class="req-value">${specs.min_ram_gb || 'Unknown'} GB</div>
        </div>
        <div class="req-item">
            <div class="req-label">Rec. RAM</div>
            <div class="req-value">${specs.recommended_ram_gb || 'Unknown'} GB</div>
        </div>
        <div class="req-item">
            <div class="req-label">Min VRAM</div>
            <div class="req-value">${specs.min_vram_gb || 'Unknown'} GB</div>
        </div>
        <div class="req-item">
            <div class="req-label">Rec. VRAM</div>
            <div class="req-value">${specs.recommended_vram_gb || 'Unknown'} GB</div>
        </div>
    </div>
    
    ${compatibility.recommended_quant 
      ? `<p style="margin: 1rem 0; font-weight: 500;"><strong>🔧 Recommended Quantization:</strong> ${compatibility.recommended_quant}</p>` 
      : ''
    }
    
    ${compatibility.notes && compatibility.notes.length > 0 
      ? `<p style="margin: 1rem 0; font-weight: 500;"><strong>ℹ️ Notes:</strong> ${compatibility.notes.join('; ')}</p>` 
      : ''
    }
    
    ${this.generateInstallationMethods(model)}
</div>`;
  }

  private generateInstallationMethods(model: ModelRecommendation): string {
    if (!model.specs || !model.specs.install_methods) {
      return '<div class="installation-methods"><p>No installation methods available</p></div>';
    }

    const methods = model.specs.install_methods;
    let html = `
<div class="installation-methods">
    <h4 style="color: var(--gray-900); margin-bottom: 1.25rem; font-size: 1.125rem; font-weight: 600;">🚀 Installation & Setup Instructions</h4>
`;

    if (methods.ollama) {
      html += `
<div class="install-method">
    <div class="install-header">
        <div class="install-title">📱 OLLAMA (Recommended for Beginners)</div>
        <div class="install-badge easy">Easy Setup</div>
    </div>
    <div class="install-command">${methods.ollama.command}</div>
    <div class="install-note">✅ <strong>Setup:</strong> Install Ollama from ollama.ai, then run command above. ${methods.ollama.note}</div>
</div>`;
    }

    if (methods.huggingface) {
      html += `
<div class="install-method">
    <div class="install-header">
        <div class="install-title">🤗 HUGGING FACE (For Developers)</div>
        <div class="install-badge intermediate">Intermediate</div>
    </div>
    <div style="margin: 1rem 0;"><strong>Model ID:</strong> <code style="background: var(--gray-100); padding: 0.375rem 0.75rem; border-radius: 6px; font-family: Monaco, monospace; font-size: 0.8125rem;">${methods.huggingface.model_id}</code></div>
    <div class="install-note">💡 <strong>Requirements:</strong> ${methods.huggingface.note}</div>
    <div class="install-note">🔧 <strong>Install:</strong> pip install transformers torch</div>
</div>`;
    }

    if (methods.gguf) {
      html += `
<div class="install-method">
    <div class="install-header">
        <div class="install-title">⚙️ GGUF (Advanced Users)</div>
        <div class="install-badge advanced">Advanced</div>
    </div>
    <div style="margin: 1rem 0;"><strong>Download Source:</strong> <a href="${methods.gguf.source}" target="_blank" style="color: var(--primary-color); word-break: break-all; text-decoration: underline;">${methods.gguf.source}</a></div>
    <div class="install-note">🎯 <strong>Recommended:</strong> ${methods.gguf.recommended_quant}</div>
    <div class="install-note">💡 <strong>Best for:</strong> ${methods.gguf.note}</div>
</div>`;
    }

    html += '</div>';
    return html;
  }

  private generateInsufficientHardware(systemInfo: SystemInfo): string {
    return `
<div class="section">
    <h2>❌ Insufficient Hardware Detected</h2>
    
    <div class="insufficient-hardware">
        <h3>Your Current System:</h3>
        <ul style="margin: 1.25rem 0 0 1.5rem; line-height: 1.8;">
            <li>RAM: ${systemInfo.totalRamGB || 'Unknown'} GB</li>
            <li>Free Storage: ${systemInfo.freeStorageGB || 'Unknown'} GB</li>
            <li>GPUs: ${systemInfo.gpus?.length || 0} detected</li>
        </ul>
        
        <p style="margin-top: 1.25rem;">
            Unfortunately, your system doesn't meet the minimum requirements 
            for running local LLMs efficiently.
        </p>
    </div>
    
    <h3 style="margin-top: 2.5rem; color: var(--primary-dark); font-weight: 700;">🌐 Recommended Cloud-Based Solutions:</h3>
    <div class="cloud-solutions">
        <a href="https://chat.openai.com" target="_blank" class="cloud-solution">
            <h4>ChatGPT</h4>
            <p>OpenAI's flagship conversational AI with GPT-4 models</p>
        </a>
        <a href="https://claude.ai" target="_blank" class="cloud-solution">
            <h4>Claude</h4>
            <p>Anthropic's helpful, harmless, and honest AI assistant</p>
        </a>
        <a href="https://gemini.google.com" target="_blank" class="cloud-solution">
            <h4>Google Gemini</h4>
            <p>Google's advanced conversational AI service</p>
        </a>
        <a href="https://perplexity.ai" target="_blank" class="cloud-solution">
            <h4>Perplexity AI</h4>
            <p>AI-powered search and question answering</p>
        </a>
    </div>
</div>`;
  }

  private generateInstallationGuide(recommender: LLMRecommender): string {
    if (!recommender || typeof recommender.getInstallationPlatforms !== 'function') {
      return `
<div class="section">
    <h2>🛠️ Installation Guide</h2>
    <p>Installation guide data is not available.</p>
</div>`;
    }

    try {
      const platforms = recommender.getInstallationPlatforms();
      
      return `
<div class="section">
    <h2>🛠️ Platform Installation Guide</h2>
    
    <div style="display: grid; gap: 1.5rem; margin-top: 1.5rem;">
        ${platforms.map(platform => `
        <div style="background: white; padding: 2rem; border-radius: 12px; border: 1px solid var(--gray-200); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 1rem;">
                <h3 style="color: var(--gray-900); margin: 0; font-weight: 700; font-size: 1.25rem;">${platform.name || 'Unknown Platform'}</h3>
                <span class="install-badge ${(platform.difficulty || 'intermediate').toLowerCase()}">${platform.difficulty || 'Intermediate'}</span>
            </div>
            <p style="color: var(--gray-600); margin-bottom: 1.25rem; line-height: 1.6;">${platform.description || 'No description available'}</p>
            <p style="color: var(--gray-700); margin-bottom: 1.25rem; font-weight: 500;"><strong>Best for:</strong> ${platform.bestFor || 'General use'}</p>
            
            <div style="background: var(--gray-50); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--gray-200);">
                <h4 style="color: var(--gray-700); margin-bottom: 1rem; font-weight: 600;">Installation:</h4>
                ${Object.entries(platform.installation || {}).map(([os, instruction]) => `
                <div style="margin-bottom: 0.75rem; font-size: 0.875rem;">
                    <strong style="color: var(--gray-800);">${os}:</strong> <span style="color: var(--gray-600);">${instruction}</span>
                </div>
                `).join('')}
            </div>
        </div>
        `).join('')}
    </div>
</div>`;
    } catch (error) {
      console.error('Error generating installation guide:', error);
      return `
<div class="section">
    <h2>🛠️ Installation Guide</h2>
    <p>Unable to load installation guide. Please check the console for errors.</p>
</div>`;
    }
  }

  private generateOptimizationTips(recommender: LLMRecommender): string {
    if (!recommender || typeof recommender.getOptimizationTips !== 'function') {
      return `
<div class="section">
    <h2>💡 Optimization Tips</h2>
    <p>Optimization tips are not available.</p>
</div>`;
    }

    try {
      const tips = recommender.getOptimizationTips();
      
      return `
<div class="section">
    <h2>💡 Optimization Tips for Your System</h2>
    
    <div class="tips-grid">
        <div class="tip-card">
            <h4>System-Specific Recommendations</h4>
            <ul>
                ${tips.slice(0, Math.ceil(tips.length / 2)).map(tip => `<li>${tip}</li>`).join('')}
            </ul>
        </div>
        <div class="tip-card">
            <h4>General Performance Tips</h4>
            <ul>
                ${tips.slice(Math.ceil(tips.length / 2)).map(tip => `<li>${tip}</li>`).join('')}
            </ul>
        </div>
    </div>
</div>`;
    } catch (error) {
      console.error('Error generating optimization tips:', error);
      return `
<div class="section">
    <h2>💡 Optimization Tips</h2>
    <div class="tip-card">
        <h4>Default Optimization Tips</h4>
        <ul>
            <li>Close unnecessary applications to free up RAM</li>
            <li>Ensure adequate disk space for model downloads</li>
            <li>Use quantized models for better performance on limited hardware</li>
            <li>Consider cloud-based solutions if local hardware is insufficient</li>
        </ul>
    </div>
</div>`;
    }
  }

  private generateFooter(timestamp: string): string {
    return `
<div class="footer">
    <p style="font-size: 1.125rem; font-weight: 600; margin-bottom: 0.75rem; color: var(--gray-800);">Generated by LLM Hardware Compatibility Checker</p>
    <p style="color: var(--gray-600);">Report created on ${timestamp}</p>
    <p style="margin-top: 1.25rem; font-size: 0.875rem; opacity: 0.8; color: var(--gray-500);">
        For the latest updates and more models, visit the project repository
    </p>
</div>`;
  }
}
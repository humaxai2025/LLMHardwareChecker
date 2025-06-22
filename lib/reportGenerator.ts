// Report Generation Utilities
import { SystemInfo } from './systemAnalyzer';
import { Recommendations, ModelRecommendation } from './llmDatabase';
import { LLMRecommender } from './llmRecommender';

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
    const { systemInfo, recommendations, recommender } = this.data;
    const timestamp = new Date().toLocaleString();
    
    const suitableModels = [
      ...recommendations.excellent,
      ...recommendations.good,
      ...recommendations.basic
    ];

    const totalSpecialized = suitableModels.filter(m => m.specs.domain).length;
    const totalGeneral = suitableModels.length - totalSpecialized;

    return `
<!DOCTYPE html>
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
  }

  private getReportCSS(): string {
    return `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary-50: #eff6ff;
    --primary-100: #dbeafe;
    --primary-500: #3b82f6;
    --primary-600: #2563eb;
    --primary-700: #1d4ed8;
    --primary-900: #1e3a8a;
    
    --success-50: #ecfdf5;
    --success-500: #10b981;
    --success-600: #059669;
    
    --warning-50: #fffbeb;
    --warning-500: #f59e0b;
    --warning-600: #d97706;
    
    --error-50: #fef2f2;
    --error-500: #ef4444;
    --error-600: #dc2626;
    
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-400: #9ca3af;
    --gray-500: #6b7280;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
    
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
    
    --border-radius-sm: 0.375rem;
    --border-radius-md: 0.5rem;
    --border-radius-lg: 0.75rem;
    --border-radius-xl: 1rem;
    --border-radius-2xl: 1.5rem;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-feature-settings: 'cv11', 'ss01';
    font-variation-settings: 'opsz' 32;
    line-height: 1.6;
    color: var(--gray-800);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    min-height: 100vh;
    font-size: 15px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    background: white;
    border-radius: var(--border-radius-2xl);
    box-shadow: var(--shadow-2xl);
    margin-top: 2rem;
    margin-bottom: 2rem;
    border: 1px solid var(--gray-200);
}

.header {
    text-align: center;
    margin-bottom: 3rem;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: var(--border-radius-xl);
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
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}

.header h1 {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
    position: relative;
    z-index: 1;
    letter-spacing: -0.025em;
    line-height: 1.1;
}

.header .subtitle {
    font-size: 1.125rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
    font-weight: 400;
    letter-spacing: 0.025em;
}

.section {
    margin-bottom: 3rem;
    padding: 2.5rem;
    background: var(--gray-50);
    border-radius: var(--border-radius-xl);
    border: 1px solid var(--gray-200);
}

.section h2 {
    color: var(--primary-700);
    font-size: 1.875rem;
    font-weight: 700;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    letter-spacing: -0.025em;
}

.specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.spec-item {
    background: white;
    padding: 1.75rem;
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.spec-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, var(--primary-500), var(--primary-600));
    transform: scaleY(0);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.spec-item:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-200);
}

.spec-item:hover::before {
    transform: scaleY(1);
}

.spec-label {
    font-weight: 600;
    color: var(--gray-600);
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-family: inherit;
}

.spec-value {
    color: var(--gray-900);
    font-size: 1.125rem;
    font-weight: 600;
    line-height: 1.4;
}

.summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.stat-card {
    background: white;
    padding: 2rem;
    border-radius: var(--border-radius-xl);
    text-align: center;
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-sm);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-500), var(--primary-600));
    transform: scaleX(0);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stat-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-xl);
}

.stat-card:hover::before {
    transform: scaleX(1);
}

.stat-number {
    font-size: 3rem;
    font-weight: 800;
    color: var(--primary-600);
    margin-bottom: 0.5rem;
    line-height: 1;
    letter-spacing: -0.05em;
}

.stat-label {
    color: var(--gray-600);
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.025em;
}

.model-card {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: var(--border-radius-xl);
    padding: 2rem;
    margin: 1.5rem 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, var(--primary-500), var(--primary-600));
    transform: scaleY(0);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.model-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-200);
}

.model-card:hover::before {
    transform: scaleY(1);
}

.model-header {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.model-name {
    font-size: 1.375rem;
    font-weight: 700;
    color: var(--gray-900);
    flex: 1;
    letter-spacing: -0.025em;
}

.model-domain {
    background: linear-gradient(135deg, var(--error-500), var(--error-600));
    color: white;
    padding: 0.375rem 1rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: var(--shadow-sm);
}

.performance-tier {
    display: inline-block;
    padding: 0.625rem 1.25rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.875rem;
    margin: 1rem 0;
    letter-spacing: 0.025em;
    box-shadow: var(--shadow-sm);
}

.performance-tier.excellent {
    background: linear-gradient(135deg, var(--success-500), var(--success-600));
    color: white;
}

.performance-tier.good {
    background: linear-gradient(135deg, var(--warning-500), var(--warning-600));
    color: white;
}

.performance-tier.basic {
    background: linear-gradient(135deg, var(--error-500), var(--error-600));
    color: white;
}

.requirements {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
    padding: 1.5rem;
    background: var(--gray-50);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--gray-200);
}

.req-item {
    text-align: center;
    padding: 1rem;
    background: white;
    border-radius: var(--border-radius-md);
    border: 1px solid var(--gray-200);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.req-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.req-label {
    font-size: 0.75rem;
    color: var(--gray-500);
    margin-bottom: 0.375rem;
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
    margin-top: 2rem;
    padding: 2rem;
    background: var(--gray-50);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--gray-200);
}

.install-method {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: var(--border-radius-lg);
    padding: 1.75rem;
    margin: 1.25rem 0;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.install-method:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.install-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.install-title {
    font-weight: 600;
    color: var(--gray-900);
    font-size: 1.125rem;
    letter-spacing: -0.025em;
}

.install-badge {
    padding: 0.375rem 0.875rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.install-badge.easy {
    background: var(--success-50);
    color: var(--success-600);
    border: 1px solid var(--success-200);
}

.install-badge.intermediate {
    background: var(--warning-50);
    color: var(--warning-600);
    border: 1px solid var(--warning-200);
}

.install-badge.advanced {
    background: var(--error-50);
    color: var(--error-600);
    border: 1px solid var(--error-200);
}

.install-command {
    background: var(--gray-900);
    color: var(--gray-50);
    padding: 1rem 1.25rem;
    border-radius: var(--border-radius-md);
    font-family: 'JetBrains Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 1rem 0;
    overflow-x: auto;
    position: relative;
    border: 1px solid var(--gray-700);
    line-height: 1.5;
}

.install-command::before {
    content: '$ ';
    color: var(--success-400);
    font-weight: 600;
}

.copy-btn {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    background: var(--gray-700);
    color: white;
    border: none;
    padding: 0.375rem 0.75rem;
    border-radius: var(--border-radius-sm);
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: 500;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    font-family: inherit;
}

.copy-btn:hover {
    background: var(--gray-600);
    transform: scale(1.05);
}

.install-note {
    background: var(--primary-50);
    border-left: 4px solid var(--primary-500);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    border-radius: var(--border-radius-sm);
    font-size: 0.875rem;
    line-height: 1.6;
}

.quick-start {
    background: linear-gradient(135deg, var(--warning-50), var(--warning-100));
    border: 1px solid var(--warning-200);
    border-radius: var(--border-radius-lg);
    padding: 1.75rem;
    margin-top: 1.5rem;
}

.platform-tabs {
    display: grid;
    gap: 1rem;
    margin-top: 1.25rem;
}

.platform-tab {
    background: white;
    padding: 1.25rem;
    border-radius: var(--border-radius-md);
    border-left: 4px solid var(--warning-500);
    font-size: 0.875rem;
    box-shadow: var(--shadow-sm);
}

.platform-tab code {
    background: var(--gray-100);
    padding: 0.25rem 0.5rem;
    border-radius: var(--border-radius-sm);
    font-family: 'JetBrains Mono', Monaco, monospace;
    font-size: 0.8125rem;
    font-weight: 500;
}

.insufficient-hardware {
    background: linear-gradient(135deg, var(--error-50), var(--error-100));
    border: 1px solid var(--error-200);
    border-radius: var(--border-radius-xl);
    padding: 2.5rem;
    margin: 2rem 0;
}

.insufficient-hardware h3 {
    color: var(--error-700);
    margin-bottom: 1.25rem;
    font-weight: 700;
}

.cloud-solutions {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin-top: 1.5rem;
}

.cloud-solution {
    background: white;
    padding: 1.75rem;
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--gray-200);
    text-decoration: none;
    color: inherit;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-sm);
}

.cloud-solution:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-300);
}

.cloud-solution h4 {
    color: var(--gray-900);
    margin-bottom: 0.75rem;
    font-weight: 600;
    font-size: 1.125rem;
}

.cloud-solution p {
    color: var(--gray-600);
    font-size: 0.875rem;
    line-height: 1.5;
}

.footer {
    text-align: center;
    padding: 2.5rem;
    background: var(--gray-50);
    border-radius: var(--border-radius-lg);
    margin-top: 3rem;
    color: var(--gray-600);
    border: 1px solid var(--gray-200);
}

.footer p {
    margin-bottom: 0.5rem;
}

.footer p:first-child {
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--gray-800);
}

.tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.tip-card {
    background: white;
    padding: 1.75rem;
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.tip-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, var(--success-500), var(--success-600));
}

.tip-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.tip-card h4 {
    color: var(--gray-900);
    margin-bottom: 1rem;
    font-size: 1.125rem;
    font-weight: 600;
    letter-spacing: -0.025em;
}

.tip-card ul {
    margin-left: 1.5rem;
    color: var(--gray-600);
}

.tip-card li {
    margin-bottom: 0.5rem;
    line-height: 1.6;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        margin: 1rem;
        padding: 1.5rem;
    }
    
    .header {
        padding: 2rem 1.5rem;
    }
    
    .header h1 {
        font-size: 2.25rem;
    }
    
    .specs-grid {
        grid-template-columns: 1fr;
    }
    
    .install-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .model-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .section {
        padding: 1.5rem;
    }
}

/* Print Styles */
@media print {
    body {
        background: white;
        font-size: 12pt;
        line-height: 1.4;
    }
    
    .container {
        box-shadow: none;
        margin: 0;
        border: none;
    }
    
    .model-card {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid var(--gray-300);
    }
    
    .install-method {
        break-inside: avoid;
        box-shadow: none;
    }
    
    .header {
        background: var(--gray-100) !important;
        color: var(--gray-900) !important;
    }
    
    .performance-tier {
        background: var(--gray-200) !important;
        color: var(--gray-900) !important;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --gray-50: #1f2937;
        --gray-100: #374151;
        --gray-900: #f9fafb;
        --gray-800: #f3f4f6;
        --gray-700: #e5e7eb;
        --gray-600: #d1d5db;
    }
}
`;
  }

  private getReportJavaScript(): string {
    return `
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Create toast notification
        showToast('Command copied to clipboard!');
    }).catch(function(err) {
        console.error('Failed to copy:', err);
        showToast('Failed to copy command', 'error');
    });
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = \`toast toast-\${type}\`;
    toast.textContent = message;
    toast.style.cssText = \`
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: \${type === 'success' ? '#10b981' : '#ef4444'};
        color: white;
        border-radius: 8px;
        font-weight: 500;
        font-size: 14px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        opacity: 0;
        transform: translateY(-10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    \`;
    
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
        setTimeout(() => document.body.removeChild(toast), 300);
    }, 3000);
}

// Add copy functionality to command blocks
document.addEventListener('DOMContentLoaded', function() {
    const commandBlocks = document.querySelectorAll('.install-command');
    commandBlocks.forEach(block => {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.innerHTML = '📋 Copy';
        copyBtn.onclick = (e) => {
            e.preventDefault();
            const command = block.textContent.replace('$ ', '').trim();
            copyToClipboard(command);
            copyBtn.innerHTML = '✅ Copied!';
            setTimeout(() => {
                copyBtn.innerHTML = '📋 Copy';
            }, 2000);
        };
        block.style.position = 'relative';
        block.appendChild(copyBtn);
    });
    
    // Add smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
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
    
    // Add intersection observer for animation on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe cards for scroll animations
    document.querySelectorAll('.model-card, .spec-item, .stat-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s cubic-bezier(0.4, 0, 0.2, 1), transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
        observer.observe(card);
    });
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
    return `
<div class="section">
    <h2>🖥️ System Specifications</h2>
    <div class="specs-grid">
        <div class="spec-item">
            <div class="spec-label">Operating System</div>
            <div class="spec-value">${systemInfo.os} (${systemInfo.architecture})</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">Processor</div>
            <div class="spec-value">${systemInfo.processor}</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">CPU Cores</div>
            <div class="spec-value">${systemInfo.cpuCores} cores detected</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">Memory (RAM)</div>
            <div class="spec-value">${systemInfo.totalRamGB} GB total (${systemInfo.availableRamGB} GB available)</div>
        </div>
        <div class="spec-item">
            <div class="spec-label">Storage</div>
            <div class="spec-value">${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total</div>
        </div>
        ${systemInfo.gpus && systemInfo.gpus.length > 0 
          ? systemInfo.gpus.map(gpu => `
            <div class="spec-item">
                <div class="spec-label">GPU</div>
                <div class="spec-value">${gpu.name} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB})</div>
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
    const suitableModels = [
      ...recommendations.excellent,
      ...recommendations.good,
      ...recommendations.basic
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
            <div class="stat-number">${recommendations.not_suitable.length}</div>
            <div class="stat-label">Not Suitable</div>
        </div>
    </div>
`;

    const categories = [
      { key: 'excellent', title: '🟢 Excellent Performance', models: recommendations.excellent },
      { key: 'good', title: '🟡 Good Performance', models: recommendations.good },
      { key: 'basic', title: '🟠 Basic Performance', models: recommendations.basic }
    ];

    for (const category of categories) {
      if (category.models.length > 0) {
        html += `<h3 style="margin-top: 2.5rem; color: var(--primary-700); font-size: 1.5rem; font-weight: 700;">${category.title}</h3>`;
        for (const model of category.models) {
          html += this.generateModelCard(model, category.key);
        }
      }
    }

    html += '</div>';
    return html;
  }

  private generateModelCard(model: ModelRecommendation, category: string): string {
    const domainBadge = model.specs.domain 
      ? `<span class="model-domain">${model.specs.domain}</span>` 
      : '';

    return `
<div class="model-card">
    <div class="model-header">
        <div class="model-name">${model.name}</div>
        ${domainBadge}
    </div>
    
    <div class="performance-tier ${category}">
        ${model.compatibility.performance_tier}
    </div>
    
    <p style="margin: 1.25rem 0; color: var(--gray-600); line-height: 1.6; font-size: 0.95rem;">${model.specs.description}</p>
    
    <div class="requirements">
        <div class="req-item">
            <div class="req-label">Parameters</div>
            <div class="req-value">${model.specs.parameters}</div>
        </div>
        <div class="req-item">
            <div class="req-label">Min RAM</div>
            <div class="req-value">${model.specs.min_ram_gb} GB</div>
        </div>
        <div class="req-item">
            <div class="req-label">Rec. RAM</div>
            <div class="req-value">${model.specs.recommended_ram_gb} GB</div>
        </div>
        <div class="req-item">
            <div class="req-label">Min VRAM</div>
            <div class="req-value">${model.specs.min_vram_gb} GB</div>
        </div>
        <div class="req-item">
            <div class="req-label">Rec. VRAM</div>
            <div class="req-value">${model.specs.recommended_vram_gb} GB</div>
        </div>
    </div>
    
    ${model.compatibility.recommended_quant 
      ? `<p style="margin: 1rem 0; font-weight: 500;"><strong>🔧 Recommended Quantization:</strong> ${model.compatibility.recommended_quant}</p>` 
      : ''
    }
    
    ${model.compatibility.notes.length > 0 
      ? `<p style="margin: 1rem 0; font-weight: 500;"><strong>ℹ️ Notes:</strong> ${model.compatibility.notes.join('; ')}</p>` 
      : ''
    }
    
    ${this.generateInstallationMethods(model)}
</div>`;
  }

  private generateInstallationMethods(model: ModelRecommendation): string {
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
    <div style="margin: 1rem 0;"><strong>Model ID:</strong> <code style="background: var(--gray-100); padding: 0.375rem 0.75rem; border-radius: var(--border-radius-sm); font-family: 'JetBrains Mono', Monaco, monospace; font-size: 0.8125rem;">${methods.huggingface.model_id}</code></div>
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
    <div style="margin: 1rem 0;"><strong>Download Source:</strong> <a href="${methods.gguf.source}" target="_blank" style="color: var(--primary-600); word-break: break-all; text-decoration: underline;">${methods.gguf.source}</a></div>
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
            <li>RAM: ${systemInfo.totalRamGB} GB</li>
            <li>Free Storage: ${systemInfo.freeStorageGB} GB</li>
            <li>GPUs: ${systemInfo.gpus?.length || 0} detected</li>
        </ul>
        
        <p style="margin-top: 1.25rem;">
            Unfortunately, your system doesn't meet the minimum requirements 
            for running local LLMs efficiently.
        </p>
    </div>
    
    <h3 style="margin-top: 2.5rem; color: var(--primary-700); font-weight: 700;">🌐 Recommended Cloud-Based Solutions:</h3>
    <div class="cloud-solutions">
        <a href="https://chat.openai.com" target="_blank" class="cloud-solution">
            <h4>ChatGPT</h4>
            <p>OpenAI's flagship conversational AI with GPT-4 models</p>
        </a>
        <a href="https://claude.ai" target="_blank" class="cloud-solution">
            <h4>Claude</h4>
            <p>Anthropic's helpful, harmless, and honest AI assistant</p>
        </a>
        <a href="https://bard.google.com" target="_blank" class="cloud-solution">
            <h4>Google Bard</h4>
            <p>Google's experimental conversational AI service</p>
        </a>
        <a href="https://perplexity.ai" target="_blank" class="cloud-solution">
            <h4>Perplexity AI</h4>
            <p>AI-powered search and question answering</p>
        </a>
    </div>
</div>`;
  }

  private generateInstallationGuide(recommender: LLMRecommender): string {
    const platforms = recommender.getInstallationPlatforms();
    
    return `
<div class="section">
    <h2>🛠️ Platform Installation Guide</h2>
    
    <div style="display: grid; gap: 1.5rem; margin-top: 1.5rem;">
        ${platforms.map(platform => `
        <div style="background: white; padding: 2rem; border-radius: var(--border-radius-lg); border: 1px solid var(--gray-200); box-shadow: var(--shadow-sm);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 1rem;">
                <h3 style="color: var(--gray-900); margin: 0; font-weight: 700; font-size: 1.25rem;">${platform.name}</h3>
                <span class="install-badge ${platform.difficulty.toLowerCase()}">${platform.difficulty}</span>
            </div>
            <p style="color: var(--gray-600); margin-bottom: 1.25rem; line-height: 1.6;">${platform.description}</p>
            <p style="color: var(--gray-700); margin-bottom: 1.25rem; font-weight: 500;"><strong>Best for:</strong> ${platform.bestFor}</p>
            
            <div style="background: var(--gray-50); padding: 1.5rem; border-radius: var(--border-radius-md); border: 1px solid var(--gray-200);">
                <h4 style="color: var(--gray-700); margin-bottom: 1rem; font-weight: 600;">Installation:</h4>
                ${Object.entries(platform.installation).map(([os, instruction]) => `
                <div style="margin-bottom: 0.75rem; font-size: 0.875rem;">
                    <strong style="color: var(--gray-800);">${os}:</strong> <span style="color: var(--gray-600);">${instruction}</span>
                </div>
                `).join('')}
            </div>
        </div>
        `).join('')}
    </div>
</div>`;
  }

  private generateOptimizationTips(recommender: LLMRecommender): string {
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
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import {
  DocumentArrowDownIcon,
  DocumentTextIcon,
  ClipboardDocumentListIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

// Import interfaces - ensure these match your actual imports
interface SystemInfo {
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

interface ModelRecommendation {
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

interface Recommendations {
  excellent: ModelRecommendation[];
  good: ModelRecommendation[];
  basic: ModelRecommendation[];
  not_suitable: ModelRecommendation[];
}

interface LLMRecommender {
  getInstallationPlatforms(): Array<{
    name: string;
    description: string;
    difficulty: string;
    bestFor: string;
    installation: Record<string, string>;
  }>;
  getOptimizationTips(): string[];
}

interface ReportDownloadProps {
  systemInfo: SystemInfo;
  recommendations: Recommendations;
  recommender: LLMRecommender;
}

// Simple ReportGenerator class that can be included inline
class ReportGenerator {
  private data: {
    systemInfo: SystemInfo;
    recommendations: Recommendations;
    recommender: LLMRecommender;
    timestamp: Date;
  };

  constructor(data: {
    systemInfo: SystemInfo;
    recommendations: Recommendations;
    recommender: LLMRecommender;
    timestamp: Date;
  }) {
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

      return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Hardware Compatibility Report</title>
    <style>
        ${this.getCSS()}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 LLM Hardware Compatibility Report</h1>
            <div class="subtitle">Generated on ${timestamp}</div>
        </div>
        
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
                    <div class="spec-value">${systemInfo.cpuCores || 'Unknown'} cores</div>
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
        </div>
        
        ${suitableModels.length > 0 ? this.generateRecommendationsSection(recommendations) : this.generateInsufficientHardwareSection(systemInfo)}
        
        <div class="footer">
            <p><strong>Generated by LLM Hardware Compatibility Checker</strong></p>
            <p>Report created on ${timestamp}</p>
        </div>
    </div>
    
    <script>
        ${this.getJavaScript()}
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
<html>
<head><title>Report Generation Error</title></head>
<body style="font-family: Arial, sans-serif; margin: 40px;">
    <div style="background: #fee; border: 1px solid #fcc; padding: 20px; border-radius: 8px;">
        <h1 style="color: #c33;">Report Generation Error</h1>
        <p>An error occurred: ${error?.message || 'Unknown error'}</p>
    </div>
</body>
</html>`;
  }

  private generateRecommendationsSection(recommendations: Recommendations): string {
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
                <div class="stat-number">${(recommendations.excellent || []).length}</div>
                <div class="stat-label">Excellent</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${(recommendations.good || []).length}</div>
                <div class="stat-label">Good</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${(recommendations.basic || []).length}</div>
                <div class="stat-label">Basic</div>
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
        html += `<h3 class="category-title">${category.title}</h3>`;
        for (const model of category.models) {
          html += this.generateModelCard(model, category.key);
        }
      }
    }

    html += '</div>';
    return html;
  }

  private generateModelCard(model: ModelRecommendation, category: string): string {
    const specs = model.specs || {};
    const compatibility = model.compatibility || {};

    return `
    <div class="model-card">
        <div class="model-header">
            <div class="model-name">${model.name || 'Unknown Model'}</div>
            ${specs.domain ? `<span class="model-domain">${specs.domain}</span>` : ''}
        </div>
        
        <div class="performance-tier ${category}">
            ${compatibility.performance_tier || 'Unknown Performance'}
        </div>
        
        <p class="model-description">${specs.description || 'No description available'}</p>
        
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
        
        ${this.generateInstallationMethods(model)}
    </div>`;
  }

  private generateInstallationMethods(model: ModelRecommendation): string {
    const methods = model.specs?.install_methods;
    if (!methods) return '';

    let html = '<div class="installation-methods"><h4>🚀 Installation Methods</h4>';

    if (methods.ollama) {
      html += `
      <div class="install-method">
          <div class="install-header">
              <span class="install-title">📱 OLLAMA</span>
              <span class="install-badge easy">Easy</span>
          </div>
          <div class="install-command">${methods.ollama.command}</div>
          <div class="install-note">${methods.ollama.note}</div>
      </div>`;
    }

    if (methods.huggingface) {
      html += `
      <div class="install-method">
          <div class="install-header">
              <span class="install-title">🤗 HUGGING FACE</span>
              <span class="install-badge intermediate">Intermediate</span>
          </div>
          <div class="model-id">Model ID: ${methods.huggingface.model_id}</div>
          <div class="install-note">${methods.huggingface.note}</div>
      </div>`;
    }

    if (methods.gguf) {
      html += `
      <div class="install-method">
          <div class="install-header">
              <span class="install-title">⚙️ GGUF</span>
              <span class="install-badge advanced">Advanced</span>
          </div>
          <div class="install-note">Download: <a href="${methods.gguf.source}" target="_blank">${methods.gguf.source}</a></div>
          <div class="install-note">Recommended: ${methods.gguf.recommended_quant}</div>
      </div>`;
    }

    html += '</div>';
    return html;
  }

  private generateInsufficientHardwareSection(systemInfo: SystemInfo): string {
    return `
    <div class="section">
        <h2>❌ Insufficient Hardware</h2>
        <div class="insufficient-hardware">
            <h3>Your System:</h3>
            <ul>
                <li>RAM: ${systemInfo.totalRamGB || 'Unknown'} GB</li>
                <li>Free Storage: ${systemInfo.freeStorageGB || 'Unknown'} GB</li>
                <li>GPUs: ${systemInfo.gpus?.length || 0} detected</li>
            </ul>
            <p>Your system doesn't meet minimum requirements for local LLMs.</p>
            
            <h3>Recommended Cloud Solutions:</h3>
            <div class="cloud-solutions">
                <a href="https://chat.openai.com" target="_blank" class="cloud-solution">
                    <h4>ChatGPT</h4>
                    <p>OpenAI's flagship AI</p>
                </a>
                <a href="https://claude.ai" target="_blank" class="cloud-solution">
                    <h4>Claude</h4>
                    <p>Anthropic's AI assistant</p>
                </a>
                <a href="https://gemini.google.com" target="_blank" class="cloud-solution">
                    <h4>Google Gemini</h4>
                    <p>Google's AI service</p>
                </a>
            </div>
        </div>
    </div>`;
  }

  private getCSS(): string {
    return `
    :root {
        --primary-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-600: #4b5563;
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
        padding: 20px;
        -webkit-font-smoothing: antialiased;
        print-color-adjust: exact;
    }

    .container {
        max-width: 1000px;
        margin: 0 auto;
        background: white;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }

    .header {
        text-align: center;
        margin-bottom: 30px;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        print-color-adjust: exact;
    }

    .header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }

    .section {
        margin-bottom: 30px;
        padding: 25px;
        background: var(--gray-50);
        border-radius: 12px;
        border: 1px solid var(--gray-200);
        page-break-inside: avoid;
    }

    .section h2 {
        color: var(--primary-color);
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .specs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
    }

    .spec-item {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        page-break-inside: avoid;
    }

    .spec-label {
        font-weight: 600;
        color: var(--gray-600);
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }

    .spec-value {
        color: var(--gray-900);
        font-size: 1rem;
        font-weight: 600;
    }

    .summary-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }

    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid var(--gray-200);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        page-break-inside: avoid;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 5px;
    }

    .stat-label {
        color: var(--gray-600);
        font-size: 0.9rem;
        font-weight: 500;
    }

    .category-title {
        margin: 25px 0 15px 0;
        color: var(--primary-color);
        font-size: 1.3rem;
        font-weight: 700;
    }

    .model-card {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        page-break-inside: avoid;
    }

    .model-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 15px;
        flex-wrap: wrap;
        gap: 10px;
    }

    .model-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--gray-900);
    }

    .model-domain {
        background: linear-gradient(135deg, var(--error-color), #dc2626);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        print-color-adjust: exact;
    }

    .performance-tier {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin: 10px 0;
        print-color-adjust: exact;
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

    .model-description {
        margin: 15px 0;
        color: var(--gray-600);
        line-height: 1.5;
    }

    .requirements {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 10px;
        margin: 15px 0;
        padding: 15px;
        background: var(--gray-50);
        border-radius: 8px;
    }

    .req-item {
        text-align: center;
        padding: 10px;
        background: white;
        border-radius: 6px;
        border: 1px solid var(--gray-200);
    }

    .req-label {
        font-size: 0.7rem;
        color: var(--gray-600);
        margin-bottom: 3px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .req-value {
        font-weight: 600;
        color: var(--gray-900);
        font-size: 0.8rem;
    }

    .installation-methods {
        margin-top: 20px;
        padding: 20px;
        background: var(--gray-50);
        border-radius: 8px;
    }

    .installation-methods h4 {
        color: var(--gray-900);
        margin-bottom: 15px;
        font-size: 1rem;
    }

    .install-method {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        page-break-inside: avoid;
    }

    .install-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        flex-wrap: wrap;
        gap: 8px;
    }

    .install-title {
        font-weight: 600;
        color: var(--gray-900);
    }

    .install-badge {
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
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
        padding: 12px 15px;
        border-radius: 6px;
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 0.8rem;
        margin: 10px 0;
        word-break: break-all;
        position: relative;
        print-color-adjust: exact;
    }

    .install-command::before {
        content: '$ ';
        color: var(--success-color);
        font-weight: 600;
    }

    .copy-btn {
        position: absolute;
        top: 8px;
        right: 8px;
        background: var(--gray-700);
        color: white;
        border: none;
        padding: 4px 8px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.7rem;
        font-weight: 500;
    }

    .copy-btn:hover {
        background: var(--gray-600);
    }

    .install-note {
        background: #eff6ff;
        border-left: 3px solid var(--primary-color);
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.8rem;
        line-height: 1.4;
    }

    .model-id {
        margin: 10px 0;
        font-size: 0.9rem;
    }

    .model-id code {
        background: var(--gray-100);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
    }

    .insufficient-hardware {
        background: linear-gradient(135deg, #fecaca, #fca5a5);
        border: 1px solid var(--error-color);
        border-radius: 12px;
        padding: 25px;
        print-color-adjust: exact;
    }

    .insufficient-hardware h3 {
        color: #991b1b;
        margin-bottom: 15px;
    }

    .insufficient-hardware ul {
        margin: 15px 0 0 20px;
        line-height: 1.6;
    }

    .cloud-solutions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }

    .cloud-solution {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        text-decoration: none;
        color: inherit;
        display: block;
        page-break-inside: avoid;
    }

    .cloud-solution h4 {
        color: var(--gray-900);
        margin-bottom: 5px;
    }

    .cloud-solution p {
        color: var(--gray-600);
        font-size: 0.85rem;
    }

    .footer {
        text-align: center;
        padding: 25px;
        background: var(--gray-50);
        border-radius: 8px;
        margin-top: 30px;
        color: var(--gray-600);
        border: 1px solid var(--gray-200);
    }

    /* Print styles */
    @media print {
        body {
            background: white !important;
            padding: 0 !important;
        }
        
        .container {
            box-shadow: none !important;
            max-width: 100% !important;
        }
        
        .copy-btn {
            display: none !important;
        }
        
        .model-card, .spec-item, .install-method {
            break-inside: avoid;
            page-break-inside: avoid;
        }
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .container {
            padding: 20px;
        }
        
        .header h1 {
            font-size: 1.8rem;
        }
        
        .specs-grid {
            grid-template-columns: 1fr;
        }
        
        .model-header {
            flex-direction: column;
            align-items: flex-start;
        }
    }
    `;
  }

  private getJavaScript(): string {
    return `
    function copyToClipboard(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(function() {
                showToast('Copied to clipboard!');
            }).catch(function(err) {
                fallbackCopy(text);
            });
        } else {
            fallbackCopy(text);
        }
    }

    function fallbackCopy(text) {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.left = "-999999px";
        textArea.style.top = "-999999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showToast('Copied to clipboard!');
        } catch (err) {
            showToast('Copy failed', 'error');
        }
        
        document.body.removeChild(textArea);
    }

    function showToast(message, type = 'success') {
        const existingToasts = document.querySelectorAll('.toast');
        existingToasts.forEach(toast => toast.remove());
        
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toast.style.cssText = 
            'position: fixed; top: 20px; right: 20px; padding: 10px 20px; ' +
            'background: ' + (type === 'success' ? '#10b981' : '#ef4444') + '; ' +
            'color: white; border-radius: 6px; font-size: 14px; z-index: 1000; ' +
            'opacity: 0; transition: opacity 0.3s;';
        
        document.body.appendChild(toast);
        
        setTimeout(() => { toast.style.opacity = '1'; }, 10);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    document.addEventListener('DOMContentLoaded', function() {
        try {
            const commandBlocks = document.querySelectorAll('.install-command');
            commandBlocks.forEach(function(block) {
                const copyBtn = document.createElement('button');
                copyBtn.className = 'copy-btn';
                copyBtn.innerHTML = '📋';
                copyBtn.onclick = function(e) {
                    e.preventDefault();
                    const command = block.textContent.replace('$ ', '').trim();
                    copyToClipboard(command);
                    copyBtn.innerHTML = '✅';
                    setTimeout(() => { copyBtn.innerHTML = '📋'; }, 2000);
                };
                
                block.style.position = 'relative';
                block.appendChild(copyBtn);
            });
        } catch (error) {
            console.error('Error initializing JavaScript:', error);
        }
    });
    `;
  }
}

const ReportDownload: React.FC<ReportDownloadProps> = ({
  systemInfo,
  recommendations,
  recommender
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedReports, setGeneratedReports] = useState<{
    html?: string;
    pdf?: boolean;
  }>({});

  // Validate props
  const isDataValid = systemInfo && recommendations && recommender;

  const generateHTMLReport = async () => {
    if (!isDataValid) {
      toast.error('Missing required data for report generation');
      return;
    }

    setIsGenerating(true);
    const toastId = toast.loading('Generating HTML report...');

    try {
      const reportGenerator = new ReportGenerator({
        systemInfo,
        recommendations,
        recommender,
        timestamp: new Date()
      });

      const htmlContent = reportGenerator.generateHTMLReport();
      
      // Create and download blob
      const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `llm-compatibility-report-${new Date().toISOString().split('T')[0]}.html`;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setGeneratedReports(prev => ({ ...prev, html: htmlContent }));
      toast.success('HTML report downloaded successfully!', { id: toastId });
    } catch (error) {
      console.error('HTML report generation failed:', error);
      toast.error('Failed to generate HTML report. Please try again.', { id: toastId });
    } finally {
      setIsGenerating(false);
    }
  };

  const generatePDFReport = async () => {
    if (!isDataValid) {
      toast.error('Missing required data for report generation');
      return;
    }

    setIsGenerating(true);
    const toastId = toast.loading('Generating PDF report... This may take a moment.');

    try {
      // Generate optimized HTML for PDF
      const reportGenerator = new ReportGenerator({
        systemInfo,
        recommendations,
        recommender,
        timestamp: new Date()
      });

      let htmlContent = reportGenerator.generateHTMLReport();
      
      // Create optimized iframe for PDF rendering
      const iframe = document.createElement('iframe');
      iframe.style.cssText = `
        position: fixed;
        top: -9999px;
        left: -9999px;
        width: 794px;
        height: 1123px;
        border: none;
        background: white;
        visibility: hidden;
      `;
      
      document.body.appendChild(iframe);
      
      // Wait for iframe to be ready
      await new Promise<void>((resolve) => {
        iframe.onload = () => resolve();
        iframe.src = 'about:blank';
      });

      const iframeDoc = iframe.contentDocument!;
      iframeDoc.open();
      iframeDoc.write(htmlContent);
      iframeDoc.close();

      // Wait for fonts and content to load
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Wait for fonts to be ready
      if (iframeDoc.fonts) {
        await iframeDoc.fonts.ready;
      }

      // Additional wait to ensure everything is rendered
      await new Promise(resolve => setTimeout(resolve, 1000));

      const targetElement = iframeDoc.body;
      
      if (!targetElement) {
        throw new Error('Could not access iframe content for PDF generation');
      }

      // Generate canvas with high quality settings
      const canvas = await html2canvas(targetElement, {
        scale: 2,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff',
        width: 794,
        height: targetElement.scrollHeight,
        logging: false,
        imageTimeout: 15000,
        foreignObjectRendering: true,
        onclone: (clonedDoc) => {
          // Remove copy buttons and other interactive elements
          const copyButtons = clonedDoc.querySelectorAll('.copy-btn');
          copyButtons.forEach(btn => btn.remove());
          
          // Ensure all styles are applied
          const style = clonedDoc.createElement('style');
          style.textContent = `
            * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
            body { background: white !important; }
          `;
          clonedDoc.head.appendChild(style);
        }
      });

      // Create PDF with proper dimensions
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      const imgData = canvas.toDataURL('image/jpeg', 0.95);
      const pdfWidth = 210; // A4 width in mm
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      // Handle multiple pages if content is long
      let position = 0;
      const pageHeight = 297; // A4 height in mm
      let remainingHeight = pdfHeight;

      // Add first page
      pdf.addImage(imgData, 'JPEG', 0, position, pdfWidth, pdfHeight);
      remainingHeight -= pageHeight;

      // Add additional pages if needed
      while (remainingHeight > 0) {
        position = remainingHeight - pdfHeight;
        pdf.addPage();
        pdf.addImage(imgData, 'JPEG', 0, position, pdfWidth, pdfHeight);
        remainingHeight -= pageHeight;
      }

      // Save the PDF
      const fileName = `llm-compatibility-report-${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(fileName);

      // Cleanup
      document.body.removeChild(iframe);

      setGeneratedReports(prev => ({ ...prev, pdf: true }));
      toast.success('PDF report generated and downloaded successfully!', { id: toastId });
    } catch (error) {
      console.error('PDF generation failed:', error);
      toast.error(
        'PDF generation failed. This might be due to browser limitations. Please try the HTML format instead.',
        { id: toastId, duration: 6000 }
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const copyReportSummary = async () => {
    if (!isDataValid) {
      toast.error('No report data available to copy');
      return;
    }

    try {
      const suitableModels = [
        ...(recommendations.excellent || []),
        ...(recommendations.good || []),
        ...(recommendations.basic || [])
      ];

      const summary = `
LLM Hardware Compatibility Report
Generated: ${new Date().toLocaleDateString()}

SYSTEM SPECIFICATIONS:
- OS: ${systemInfo.os || 'Unknown'} (${systemInfo.architecture || 'Unknown'})
- CPU: ${systemInfo.processor || 'Unknown'} (${systemInfo.cpuCores || 'Unknown'} cores)
- RAM: ${systemInfo.totalRamGB || 'Unknown'} GB total, ${systemInfo.availableRamGB || 'Unknown'} GB available
- Storage: ${systemInfo.freeStorageGB || 'Unknown'} GB free / ${systemInfo.totalStorageGB || 'Unknown'} GB total
- GPUs: ${systemInfo.gpus?.length || 0} detected
${systemInfo.gpus?.map(gpu => `  - ${gpu.name || 'Unknown GPU'} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB || 'Unknown VRAM'})`).join('\n') || ''}

COMPATIBILITY RESULTS:
- Compatible Models: ${suitableModels.length}
- Excellent Performance: ${(recommendations.excellent || []).length}
- Good Performance: ${(recommendations.good || []).length}  
- Basic Performance: ${(recommendations.basic || []).length}
- Not Suitable: ${(recommendations.not_suitable || []).length}

RECOMMENDED MODELS:
${suitableModels.slice(0, 5).map(model => 
  `- ${model.name || 'Unknown'}: ${model.specs?.parameters || 'Unknown'} parameters (${model.compatibility?.performance_tier || 'Unknown'})`
).join('\n')}

NEXT STEPS:
1. Install Ollama from https://ollama.ai
2. Run: ollama run ${suitableModels[0]?.name?.toLowerCase().replace(/[^a-z0-9]/g, '') || 'llama3.2:3b'}
3. Start chatting with your local LLM!

For detailed installation instructions and optimization tips, download the full report.
`.trim();

      await navigator.clipboard.writeText(summary);
      toast.success('Report summary copied to clipboard!');
    } catch (error) {
      console.error('Failed to copy summary:', error);
      toast.error('Failed to copy to clipboard. Please try selecting and copying manually.');
    }
  };

  const suitableModelsCount = [
    ...(recommendations?.excellent || []),
    ...(recommendations?.good || []),
    ...(recommendations?.basic || [])
  ].length;

  if (!isDataValid) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-lg border border-red-200 overflow-hidden"
      >
        <div className="bg-red-50 px-6 py-4 border-b border-red-200">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-8 w-8 text-red-600 mr-3" />
            <h2 className="text-2xl font-bold text-red-800">Report Generation Error</h2>
          </div>
        </div>
        <div className="p-6">
          <p className="text-red-700 mb-4">
            Unable to generate reports due to missing or invalid data. Please ensure:
          </p>
          <ul className="list-disc list-inside text-red-600 space-y-2">
            <li>System information has been properly detected</li>
            <li>Model recommendations have been generated</li>
            <li>All required data is available</li>
          </ul>
          <p className="text-red-700 mt-4">
            Please run the compatibility check again or contact support if the issue persists.
          </p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-4">
        <div className="flex items-center">
          <DocumentArrowDownIcon className="h-8 w-8 text-white mr-3" />
          <h2 className="text-2xl font-bold text-white">Download Detailed Report</h2>
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Report Options */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              📄 Report Formats Available
            </h3>
            
            <div className="space-y-4">
              {/* HTML Report */}
              <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <DocumentTextIcon className="h-6 w-6 text-blue-600 mr-3" />
                    <div>
                      <h4 className="font-semibold text-gray-900">HTML Report</h4>
                      <p className="text-sm text-gray-600">Interactive web report with copy functionality</p>
                    </div>
                  </div>
                  {generatedReports.html && (
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  )}
                </div>
                
                <div className="flex flex-wrap gap-2 mb-3">
                  <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">
                    Interactive
                  </span>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                    Copy Commands
                  </span>
                  <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs font-medium">
                    Searchable
                  </span>
                  <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs font-medium">
                    Reliable
                  </span>
                </div>
                
                <button
                  onClick={generateHTMLReport}
                  disabled={isGenerating}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {isGenerating ? 'Generating...' : 'Download HTML Report'}
                </button>
              </div>

              {/* PDF Report */}
              <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <DocumentArrowDownIcon className="h-6 w-6 text-red-600 mr-3" />
                    <div>
                      <h4 className="font-semibold text-gray-900">PDF Report</h4>
                      <p className="text-sm text-gray-600">High-quality printable document</p>
                    </div>
                  </div>
                  {generatedReports.pdf && (
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  )}
                </div>
                
                <div className="flex flex-wrap gap-2 mb-3">
                  <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs font-medium">
                    Printable
                  </span>
                  <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-xs font-medium">
                    Shareable
                  </span>
                  <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs font-medium">
                    Offline
                  </span>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                    Professional
                  </span>
                </div>
                
                <button
                  onClick={generatePDFReport}
                  disabled={isGenerating}
                  className="w-full bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {isGenerating ? 'Generating PDF...' : 'Download PDF Report'}
                </button>
                
                <p className="text-xs text-gray-500 mt-2">
                  ✨ Optimized rendering for professional PDF output
                </p>
              </div>

              {/* Quick Summary */}
              <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center mb-3">
                  <ClipboardDocumentListIcon className="h-6 w-6 text-green-600 mr-3" />
                  <div>
                    <h4 className="font-semibold text-gray-900">Quick Summary</h4>
                    <p className="text-sm text-gray-600">Copy key findings to clipboard</p>
                  </div>
                </div>
                
                <button
                  onClick={copyReportSummary}
                  disabled={isGenerating}
                  className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors font-medium"
                >
                  Copy Summary to Clipboard
                </button>
              </div>
            </div>
          </div>

          {/* Report Preview */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              📋 Report Contents Preview
            </h3>
            
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-4">
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">🖥️ System Specifications</h4>
                <p className="text-sm text-gray-600">Complete hardware analysis and detection results</p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">🤖 Model Recommendations</h4>
                <p className="text-sm text-gray-600">
                  {suitableModelsCount} compatible models with performance tiers and installation instructions
                </p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">🛠️ Installation Guide</h4>
                <p className="text-sm text-gray-600">Step-by-step setup for Ollama, HuggingFace, and GGUF formats</p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">💡 Optimization Tips</h4>
                <p className="text-sm text-gray-600">System-specific recommendations for best performance</p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">📊 Compatibility Matrix</h4>
                <p className="text-sm text-gray-600">Detailed breakdown of which models work with your hardware</p>
              </div>
            </div>
            
            {/* File Info */}
            <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-800 mb-2">📁 File Information</h4>
              <div className="text-sm text-blue-700 space-y-1">
                <p>• HTML: ~200-500 KB, works in any web browser</p>
                <p>• PDF: ~1-3 MB, optimized fonts and professional layout</p>
                <p>• Both formats include all analysis data and instructions</p>
                <p>• Generated files are completely offline and private</p>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Options */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            🔗 Usage Recommendations
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-800 mb-2">💾 Save for Later</h4>
              <p className="text-sm text-yellow-700">
                Download reports to reference installation steps offline
              </p>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-800 mb-2">📤 Share Results</h4>
              <p className="text-sm text-green-700">
                Share compatibility results with team members or communities
              </p>
            </div>
            
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h4 className="font-semibold text-purple-800 mb-2">🔄 Compare Systems</h4>
              <p className="text-sm text-purple-700">
                Run analysis on multiple systems and compare capabilities
              </p>
            </div>
          </div>
        </div>

        {/* Status Messages */}
        {isGenerating && (
          <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-3"></div>
              <p className="text-blue-800 font-medium">
                Generating report... Please wait, this may take a few moments.
              </p>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ReportDownload;
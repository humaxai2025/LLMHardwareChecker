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
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #1f2937;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    border-radius: 16px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    margin-top: 20px;
    margin-bottom: 20px;
}

.header {
    text-align: center;
    margin-bottom: 40px;
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
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
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="20" cy="80" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
    opacity: 0.1;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 8px;
    position: relative;
    z-index: 1;
}

.header .subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

.section {
    margin-bottom: 40px;
    padding: 30px;
    background: #f8fafc;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.section h2 {
    color: #1e40af;
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.spec-item {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}

.spec-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.spec-label {
    font-weight: 600;
    color: #374151;
    margin-bottom: 8px;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.spec-value {
    color: #1f2937;
    font-size: 1.1rem;
    font-weight: 500;
}

.summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.stat-card {
    background: white;
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #3b82f6;
    margin-bottom: 8px;
    line-height: 1;
}

.stat-label {
    color: #6b7280;
    font-size: 0.9rem;
    font-weight: 500;
}

.model-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: #3b82f6;
    transform: scaleY(0);
    transition: transform 0.3s ease;
}

.model-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

.model-card:hover::before {
    transform: scaleY(1);
}

.model-header {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
}

.model-name {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1f2937;
    flex: 1;
}

.model-domain {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.performance-tier {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 12px 0;
}

.performance-tier.excellent {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
}

.performance-tier.good {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
}

.performance-tier.basic {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: white;
}

.requirements {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin: 20px 0;
    padding: 20px;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

.req-item {
    text-align: center;
    padding: 12px;
    background: white;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}

.req-label {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 4px;
    font-weight: 500;
}

.req-value {
    font-weight: 600;
    color: #1f2937;
    font-size: 0.9rem;
}

.installation-methods {
    margin-top: 24px;
    padding: 24px;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

.install-method {
    background: white;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
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
    color: #1f2937;
    font-size: 1.1rem;
}

.install-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
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
    background: #1f2937;
    color: #f9fafb;
    padding: 12px 16px;
    border-radius: 6px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.9rem;
    margin: 12px 0;
    overflow-x: auto;
    position: relative;
}

.install-command::before {
    content: '$ ';
    color: #10b981;
    font-weight: 600;
}

.copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: #374151;
    color: white;
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.75rem;
    transition: background 0.2s;
}

.copy-btn:hover {
    background: #4b5563;
}

.install-note {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 4px;
    font-size: 0.9rem;
}

.quick-start {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
}

.platform-tabs {
    display: grid;
    gap: 12px;
    margin-top: 16px;
}

.platform-tab {
    background: white;
    padding: 16px;
    border-radius: 6px;
    border-left: 4px solid #f59e0b;
    font-size: 0.9rem;
}

.platform-tab code {
    background: #f3f4f6;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.85rem;
}

.insufficient-hardware {
    background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 30px;
    margin: 30px 0;
}

.insufficient-hardware h3 {
    color: #991b1b;
    margin-bottom: 16px;
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
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
    text-decoration: none;
    color: inherit;
    transition: transform 0.2s;
}

.cloud-solution:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.cloud-solution h4 {
    color: #1f2937;
    margin-bottom: 8px;
}

.cloud-solution p {
    color: #6b7280;
    font-size: 0.9rem;
}

.footer {
    text-align: center;
    padding: 30px;
    background: #f8fafc;
    border-radius: 8px;
    margin-top: 40px;
    color: #6b7280;
    border: 1px solid #e2e8f0;
}

.tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.tip-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #10b981;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.tip-card h4 {
    color: #1f2937;
    margin-bottom: 12px;
    font-size: 1.1rem;
}

.tip-card ul {
    margin-left: 20px;
    color: #4b5563;
}

.tip-card li {
    margin-bottom: 6px;
}

@media (max-width: 768px) {
    .container {
        margin: 10px;
        padding: 16px;
    }
    
    .header h1 {
        font-size: 2rem;
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
}

@media print {
    body {
        background: white;
    }
    
    .container {
        box-shadow: none;
        margin: 0;
    }
    
    .model-card {
        break-inside: avoid;
    }
    
    .install-method {
        break-inside: avoid;
    }
}
`;
  }

  private getReportJavaScript(): string {
    return `
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Could add toast notification here
        console.log('Copied to clipboard:', text);
    });
}

// Add copy functionality to command blocks
document.addEventListener('DOMContentLoaded', function() {
    const commandBlocks = document.querySelectorAll('.install-command');
    commandBlocks.forEach(block => {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.textContent = '📋';
        copyBtn.onclick = () => copyToClipboard(block.textContent.replace('$ ', ''));
        block.style.position = 'relative';
        block.appendChild(copyBtn);
    });
});
`;
  }

  private generateHeader(timestamp: string): string {
    return `
<div class="header">
    <h1>🤖 LLM Hardware Compatibility Report</h1>
    <div class="subtitle">Generated on ${timestamp}</div>
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
        html += `<h3 style="margin-top: 40px; color: #1e40af; font-size: 1.4rem;">${category.title}</h3>`;
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
    
    <p style="margin: 16px 0; color: #4b5563; line-height: 1.6;">${model.specs.description}</p>
    
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
      ? `<p><strong>🔧 Recommended Quantization:</strong> ${model.compatibility.recommended_quant}</p>` 
      : ''
    }
    
    ${model.compatibility.notes.length > 0 
      ? `<p><strong>ℹ️ Notes:</strong> ${model.compatibility.notes.join('; ')}</p>` 
      : ''
    }
    
    ${this.generateInstallationMethods(model)}
</div>`;
  }

  private generateInstallationMethods(model: ModelRecommendation): string {
    const methods = model.specs.install_methods;
    let html = `
<div class="installation-methods">
    <h4 style="color: #1f2937; margin-bottom: 16px; font-size: 1.1rem;">🚀 Installation & Setup Instructions</h4>
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
    <div style="margin: 12px 0;"><strong>Model ID:</strong> <code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px; font-family: Monaco, monospace;">${methods.huggingface.model_id}</code></div>
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
    <div style="margin: 12px 0;"><strong>Download Source:</strong> <a href="${methods.gguf.source}" target="_blank" style="color: #3b82f6; word-break: break-all;">${methods.gguf.source}</a></div>
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
        <ul style="margin: 16px 0 0 20px; line-height: 1.8;">
            <li>RAM: ${systemInfo.totalRamGB} GB</li>
            <li>Free Storage: ${systemInfo.freeStorageGB} GB</li>
            <li>GPUs: ${systemInfo.gpus?.length || 0} detected</li>
        </ul>
        
        <p style="margin-top: 16px;">
            Unfortunately, your system doesn't meet the minimum requirements 
            for running local LLMs efficiently.
        </p>
    </div>
    
    <h3 style="margin-top: 30px; color: #1e40af;">🌐 Recommended Cloud-Based Solutions:</h3>
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
    
    <div style="display: grid; gap: 20px; margin-top: 20px;">
        ${platforms.map(platform => `
        <div style="background: white; padding: 24px; border-radius: 8px; border: 1px solid #e5e7eb;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 12px;">
                <h3 style="color: #1f2937; margin: 0;">${platform.name}</h3>
                <span class="install-badge ${platform.difficulty.toLowerCase()}">${platform.difficulty}</span>
            </div>
            <p style="color: #4b5563; margin-bottom: 16px;">${platform.description}</p>
            <p style="color: #374151; margin-bottom: 16px;"><strong>Best for:</strong> ${platform.bestFor}</p>
            
            <div style="background: #f8fafc; padding: 16px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <h4 style="color: #374151; margin-bottom: 12px;">Installation:</h4>
                ${Object.entries(platform.installation).map(([os, instruction]) => `
                <div style="margin-bottom: 8px;">
                    <strong>${os}:</strong> ${instruction}
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
    <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 8px;">Generated by LLM Hardware Compatibility Checker</p>
    <p>Report created on ${timestamp}</p>
    <p style="margin-top: 16px; font-size: 0.9rem; opacity: 0.8;">
        For the latest updates and more models, visit the project repository
    </p>
</div>`;
  }
}
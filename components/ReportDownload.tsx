import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  DocumentArrowDownIcon,
  DocumentTextIcon,
  ClipboardDocumentListIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { SystemInfo } from '../lib/systemAnalyzer';
import { Recommendations } from '../lib/llmDatabase';
import { LLMRecommender } from '../lib/llmRecommender';
import { ReportGenerator } from '../lib/reportGenerator';

interface ReportDownloadProps {
  systemInfo: SystemInfo;
  recommendations: Recommendations;
  recommender: LLMRecommender;
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

  const generateHTMLReport = async () => {
    setIsGenerating(true);
    toast.loading('Generating HTML report...', { id: 'html-report' });

    try {
      const reportGenerator = new ReportGenerator({
        systemInfo,
        recommendations,
        recommender,
        timestamp: new Date()
      });

      const htmlContent = reportGenerator.generateHTMLReport();

      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `llm-compatibility-report-${new Date().toISOString().split('T')[0]}.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setGeneratedReports(prev => ({ ...prev, html: htmlContent }));
      toast.success('HTML report downloaded!', { id: 'html-report' });
    } catch (error) {
      console.error('Failed to generate HTML report:', error);
      toast.error('Failed to generate HTML report', { id: 'html-report' });
    } finally {
      setIsGenerating(false);
    }
  };

  const generatePDFReport = async () => {
    setIsGenerating(true);
    toast.loading('Generating comprehensive PDF report...', { id: 'pdf-report' });

    try {
      const { jsPDF } = await import('jspdf');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;
      const contentWidth = pageWidth - margin * 2;
      let y = margin;

      // Helper functions for consistent formatting
      const heading = (text: string, size = 16, color: [number, number, number] = [0, 0, 0]) => {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(size);
        pdf.setTextColor(color[0], color[1], color[2]);
        pdf.text(text, margin, y);
        y += size * 0.7 + 2;
      };

      const subheading = (text: string, size = 13, color: [number, number, number] = [0, 0, 0]) => {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(size);
        pdf.setTextColor(color[0], color[1], color[2]);
        pdf.text(text, margin, y);
        y += size * 0.7 + 1;
      };

      const para = (text: string, size = 10, indent = 0) => {
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(size);
        pdf.setTextColor(0, 0, 0);
        const lines = pdf.splitTextToSize(text, contentWidth - indent);
        pdf.text(lines, margin + indent, y);
        y += lines.length * size * 0.4 + 2;
      };

      const code = (text: string, size = 9) => {
        pdf.setFont('courier', 'normal');
        pdf.setFontSize(size);
        pdf.setFillColor(240, 240, 240);
        pdf.setTextColor(0, 0, 0);
        const lines = pdf.splitTextToSize(text, contentWidth - 10);
        const height = lines.length * size * 0.4 + 4;
        pdf.rect(margin + 5, y - 2, contentWidth - 10, height, 'F');
        pdf.text(lines, margin + 8, y + 2);
        y += height + 2;
      };

      const ensureSpace = (rows: number) => {
        if (y > pageHeight - (rows * 6 + margin)) {
          pdf.addPage();
          y = margin;
        }
      };

      const drawSeparator = () => {
        pdf.setDrawColor(200, 200, 200);
        pdf.line(margin, y, pageWidth - margin, y);
        y += 4;
      };

      // TITLE PAGE
      pdf.setFillColor(59, 130, 246);
      pdf.rect(0, 0, pageWidth, 25, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(20);
      pdf.setFont('helvetica', 'bold');
      pdf.text('🤖 LLM Hardware Compatibility Report', margin, 16);
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(12);
      pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, 21);
      y = 35;
      pdf.setTextColor(0, 0, 0);

      // EXECUTIVE SUMMARY
      heading('📋 Executive Summary', 14, [30, 64, 175]);
      const suitableModels = [
        ...recommendations.excellent,
        ...recommendations.good,
        ...recommendations.basic
      ];
      para(`✅ Compatible Models Found: ${suitableModels.length}`);
      para(`🟢 Excellent Performance: ${recommendations.excellent.length} models`);
      para(`🟡 Good Performance: ${recommendations.good.length} models`);
      para(`🟠 Basic Performance: ${recommendations.basic.length} models`);
      para(`❌ Not Suitable: ${recommendations.not_suitable.length} models`);
      
      if (suitableModels.length > 0) {
        para(`🎯 Top Recommendation: ${suitableModels[0].name} (${suitableModels[0].specs.parameters})`);
        const ollamaCommand = suitableModels[0].specs.install_methods.ollama?.command?.replace('ollama run ', '') || 'model-name';
        para(`🚀 Quick Start: Install Ollama, then run: ollama run ${ollamaCommand}`);
      }
      y += 5;

      // SYSTEM SPECIFICATIONS
      ensureSpace(10);
      heading('🖥️ System Specifications', 14, [30, 64, 175]);
      para(`Operating System: ${systemInfo.os} (${systemInfo.architecture})`);
      para(`Processor: ${systemInfo.processor} (${systemInfo.cpuCores} cores)`);
      para(`Memory: ${systemInfo.totalRamGB} GB total, ${systemInfo.availableRamGB} GB available`);
      para(`Storage: ${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total`);
      
      if (systemInfo.gpus && systemInfo.gpus.length > 0) {
        systemInfo.gpus.forEach((gpu, i) => {
          para(`GPU ${i + 1}: ${gpu.name} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB})`);
        });
      } else {
        para('GPU: None detected (CPU-only processing)');
      }
      y += 5;

      // MODEL RECOMMENDATIONS BY CATEGORY
      const categories = [
        { title: '🟢 Excellent Performance Models', models: recommendations.excellent, color: [16, 185, 129] as [number, number, number] },
        { title: '🟡 Good Performance Models', models: recommendations.good, color: [245, 158, 11] as [number, number, number] },
        { title: '🟠 Basic Performance Models', models: recommendations.basic, color: [239, 68, 68] as [number, number, number] }
      ];

      for (const category of categories) {
        if (category.models.length > 0) {
          ensureSpace(8);
          heading(category.title, 13, category.color);
          
          category.models.forEach((model, idx) => {
            ensureSpace(15);
            
            // Model header
            subheading(`${idx + 1}. ${model.name}`, 12);
            if (model.specs.domain) {
              para(`🎯 Specialized Domain: ${model.specs.domain}`, 9, 5);
            }
            para(`📊 Performance Tier: ${model.compatibility.performance_tier}`, 9, 5);
            para(`📝 ${model.specs.description}`, 9, 5);
            
            // Requirements
            para(`💾 RAM Requirements: ${model.specs.min_ram_gb}-${model.specs.recommended_ram_gb} GB (Min-Recommended)`, 9, 5);
            para(`🎮 VRAM Requirements: ${model.specs.min_vram_gb || 'N/A'}-${model.specs.recommended_vram_gb || 'N/A'} GB`, 9, 5);
            para(`⚙️ CPU-Only Support: ${model.specs.cpu_only ? 'Yes' : 'No'}`, 9, 5);
            
            if (model.compatibility.recommended_quant) {
              para(`🔧 Recommended Quantization: ${model.compatibility.recommended_quant}`, 9, 5);
            }
            
            if (model.compatibility.notes.length > 0) {
              para(`ℹ️ Notes: ${model.compatibility.notes.join('; ')}`, 9, 5);
            }
            
            // Installation methods
            para('🚀 Installation Methods:', 10, 5);
            
            if (model.specs.install_methods.ollama) {
              para('• OLLAMA (Recommended for Beginners):', 9, 10);
              code(`$ ${model.specs.install_methods.ollama.command}`);
              para(`  ${model.specs.install_methods.ollama.note}`, 8, 10);
            }
            
            if (model.specs.install_methods.lm_studio) {
              para('• LM STUDIO (GUI Option):', 9, 10);
              if (model.specs.install_methods.lm_studio.command) {
                code(`${model.specs.install_methods.lm_studio.command}`);
              }
              para(`  ${model.specs.install_methods.lm_studio.note}`, 8, 10);
            }
            
            if (model.specs.install_methods.huggingface) {
              para('• HUGGING FACE (For Developers):', 9, 10);
              if (model.specs.install_methods.huggingface.model_id) {
                para(`  Model ID: ${model.specs.install_methods.huggingface.model_id}`, 8, 10);
              }
              if (model.specs.install_methods.huggingface.command) {
                code(`${model.specs.install_methods.huggingface.command}`);
              }
              para(`  ${model.specs.install_methods.huggingface.note}`, 8, 10);
            }
            
            if (model.specs.install_methods.gguf) {
              para('• GGUF/llama.cpp (Advanced Users):', 9, 10);
              if (model.specs.install_methods.gguf.source) {
                para(`  Download: ${model.specs.install_methods.gguf.source}`, 8, 10);
              }
              if (model.specs.install_methods.gguf.recommended_quant) {
                para(`  Recommended: ${model.specs.install_methods.gguf.recommended_quant}`, 8, 10);
              }
              para(`  ${model.specs.install_methods.gguf.note}`, 8, 10);
            }
            
            if (model.specs.install_methods.llamacpp) {
              para('• LLAMA.CPP (Command Line):', 9, 10);
              if (model.specs.install_methods.llamacpp.command) {
                code(`${model.specs.install_methods.llamacpp.command}`);
              }
              if (model.specs.install_methods.llamacpp.download_url) {
                para(`  Download: ${model.specs.install_methods.llamacpp.download_url}`, 8, 10);
              }
              para(`  ${model.specs.install_methods.llamacpp.note}`, 8, 10);
            }
            
            drawSeparator();
          });
        }
      }

      // NOT SUITABLE MODELS (if any)
      if (recommendations.not_suitable.length > 0) {
        ensureSpace(8);
        heading('❌ Models Not Suitable for Your Hardware', 13, [185, 28, 28]);
        recommendations.not_suitable.forEach((model, idx) => {
          para(`${idx + 1}. ${model.name} - Requires ${model.specs.recommended_ram_gb}GB RAM, ${model.specs.recommended_vram_gb}GB VRAM`);
        });
        y += 3;
      }

      // PLATFORM INSTALLATION GUIDE
      ensureSpace(15);
      heading('🛠️ Platform Installation Guide', 14, [30, 64, 175]);
      
      subheading('1. Ollama (Recommended for Beginners)', 12);
      para('Ollama is the easiest way to run LLMs locally with automatic model management.', 10, 5);
      para('• Website: https://ollama.ai', 9, 5);
      para('• Windows: Download installer from website', 9, 5);
      para('• macOS: Download installer or use: brew install ollama', 9, 5);
      para('• Linux: curl -fsSL https://ollama.ai/install.sh | sh', 9, 5);
      para('• Usage: ollama run <model-name>', 9, 5);
      y += 2;

      subheading('2. LM Studio (GUI Application)', 12);
      para('User-friendly graphical interface for running LLMs without command line.', 10, 5);
      para('• Website: https://lmstudio.ai', 9, 5);
      para('• Available for Windows, macOS, and Linux', 9, 5);
      para('• Features: Model browser, chat interface, API server', 9, 5);
      y += 2;

      subheading('3. llama.cpp (Advanced Users)', 12);
      para('High-performance C++ implementation for maximum efficiency.', 10, 5);
      para('• Repository: https://github.com/ggerganov/llama.cpp', 9, 5);
      para('• Requires compilation from source or pre-built binaries', 9, 5);
      para('• Best performance for CPU-only inference', 9, 5);
      para('• Supports GGUF quantized models', 9, 5);
      y += 2;

      subheading('4. HuggingFace Transformers (Developers)', 12);
      para('Python library for machine learning developers and researchers.', 10, 5);
      para('• Installation: pip install transformers torch', 9, 5);
      para('• Requires Python programming knowledge', 9, 5);
      para('• Access to largest model repository', 9, 5);
      para('• Full customization and fine-tuning capabilities', 9, 5);
      y += 5;

      // OPTIMIZATION TIPS
      ensureSpace(15);
      heading('💡 System Optimization Tips', 14, [30, 64, 175]);
      const tips = recommender.getOptimizationTips();
      tips.forEach((tip, idx) => {
        const cleanTip = tip.replace(/^[🔧💾🎮💻🍎🐧🪟📱⚡🗑️❌✅🚀💡⚙️🔋☁️📁]+\s*/, '');
        para(`${idx + 1}. ${cleanTip}`, 9, 5);
      });
      y += 5;

      // QUICK START GUIDE
      if (suitableModels.length > 0) {
        ensureSpace(10);
        heading('🚀 Quick Start Guide', 14, [30, 64, 175]);
        const topModel = suitableModels[0];
        para('Follow these steps to get started with your first local LLM:', 10);
        para('1. Download and install Ollama from https://ollama.ai', 9, 5);
        para('2. Open terminal/command prompt', 9, 5);
        para(`3. Run the command: ${topModel.specs.install_methods.ollama?.command || 'ollama run model-name'}`, 9, 5);
        para('4. Wait for the model to download (first time only)', 9, 5);
        para('5. Start chatting with your local LLM!', 9, 5);
        y += 3;
        
        para('💡 Pro Tips:', 10);
        para('• Models are cached locally after first download', 9, 5);
        para('• Use "ollama list" to see installed models', 9, 5);
        para('• Use "ollama rm <model>" to free up space', 9, 5);
        para('• Check "ollama --help" for more commands', 9, 5);
      } else {
        ensureSpace(8);
        heading('☁️ Recommended Cloud Solutions', 14, [30, 64, 175]);
        para('Your hardware doesn\'t meet minimum requirements for local LLMs. Consider these cloud alternatives:', 10);
        para('• ChatGPT (OpenAI): https://chat.openai.com', 9, 5);
        para('• Claude (Anthropic): https://claude.ai', 9, 5);
        para('• Gemini (Google): https://gemini.google.com', 9, 5);
        para('• Perplexity AI: https://perplexity.ai', 9, 5);
      }

      // TROUBLESHOOTING
      ensureSpace(10);
      heading('🔧 Troubleshooting Common Issues', 14, [30, 64, 175]);
      para('Out of Memory Errors:', 10);
      para('• Try smaller quantized models (Q4_K_M instead of Q8_0)', 9, 5);
      para('• Close other applications to free RAM', 9, 5);
      para('• Use CPU-only mode for large models', 9, 5);
      y += 2;

      para('Slow Performance:', 10);
      para('• Enable GPU acceleration if available', 9, 5);
      para('• Use quantized models appropriate for your hardware', 9, 5);
      para('• Ensure sufficient free storage space', 9, 5);
      y += 2;

      para('Installation Issues:', 10);
      para('• Check system requirements match recommended specs', 9, 5);
      para('• Verify internet connection for model downloads', 9, 5);
      para('• Consult platform-specific documentation', 9, 5);

      // DETAILED COMPATIBILITY MATRIX
      pdf.addPage();
      y = margin;
      heading('📊 Detailed Compatibility Matrix', 14, [30, 64, 175]);
      
      // Create a comprehensive table of all models
      const allModels = [
        ...recommendations.excellent,
        ...recommendations.good,
        ...recommendations.basic,
        ...recommendations.not_suitable,
      ];

      // Table headers
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(8);
      const colWidths = [35, 15, 20, 15, 15, 25, 20, 25];
      let x = margin;
      ['Model Name', 'Params', 'Performance', 'Min RAM', 'Rec RAM', 'Min VRAM', 'Rec VRAM', 'Status'].forEach((header, i) => {
        pdf.text(header, x, y);
        x += colWidths[i];
      });
      y += 5;

      // Table rows
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(7);
      allModels.forEach(model => {
        ensureSpace(1);
        x = margin;
        const status = [...recommendations.excellent, ...recommendations.good, ...recommendations.basic].includes(model) ? '✅ Compatible' : '❌ Not Suitable';
        [
          model.name.substring(0, 20),
          model.specs.parameters,
          model.compatibility.performance_tier.substring(0, 15),
          `${model.specs.min_ram_gb}GB`,
          `${model.specs.recommended_ram_gb}GB`,
          `${model.specs.min_vram_gb || '-'}GB`,
          `${model.specs.recommended_vram_gb || '-'}GB`,
          status
        ].forEach((cell, i) => {
          pdf.text(String(cell), x, y, { maxWidth: colWidths[i] - 2 });
          x += colWidths[i];
        });
        y += 4;
      });

      // Final footer
      y = pageHeight - 20;
      pdf.setFontSize(8);
      pdf.setTextColor(120, 120, 120);
      pdf.text(
        'Generated by LLM Hardware Compatibility Checker | Complete analysis of your system capabilities',
        margin,
        y
      );
      pdf.text(
        `Comprehensive report created on ${new Date().toLocaleString()}`,
        margin,
        y + 5
      );

      // Save the PDF
      pdf.save(`llm-compatibility-report-comprehensive-${new Date().toISOString().split('T')[0]}.pdf`);
      setGeneratedReports(prev => ({ ...prev, pdf: true }));
      toast.success('Comprehensive PDF report downloaded!', { id: 'pdf-report' });
    } catch (error) {
      console.error('Failed to generate comprehensive PDF report:', error);
      toast.error('Failed to generate PDF report. Try HTML format instead.', { id: 'pdf-report' });
    } finally {
      setIsGenerating(false);
    }
  };

  const copyReportSummary = async () => {
    const suitableModels = [
      ...recommendations.excellent,
      ...recommendations.good,
      ...recommendations.basic
    ];

    const summary = `
LLM Hardware Compatibility Report
Generated: ${new Date().toLocaleDateString()}

SYSTEM SPECIFICATIONS:
- OS: ${systemInfo.os} (${systemInfo.architecture})
- CPU: ${systemInfo.processor} (${systemInfo.cpuCores} cores)
- RAM: ${systemInfo.totalRamGB} GB total, ${systemInfo.availableRamGB} GB available
- Storage: ${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total
- GPUs: ${systemInfo.gpus?.length || 0} detected
${systemInfo.gpus?.map(gpu => `  - ${gpu.name} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB})`).join('\n') || ''}

COMPATIBILITY RESULTS:
- Compatible Models: ${suitableModels.length}
- Excellent Performance: ${recommendations.excellent.length}
- Good Performance: ${recommendations.good.length}  
- Basic Performance: ${recommendations.basic.length}
- Not Suitable: ${recommendations.not_suitable.length}

RECOMMENDED MODELS:
${suitableModels.slice(0, 5).map(model => 
  `- ${model.name}: ${model.specs.parameters} parameters (${model.compatibility.performance_tier})`
).join('\n')}

NEXT STEPS:
1. Install Ollama from https://ollama.ai
2. Run: ollama run ${suitableModels[0]?.name.toLowerCase().replace(/[^a-z0-9]/g, '') || 'llama3.2:3b'}
3. Start chatting with your local LLM!

For detailed installation instructions and optimization tips, download the full report.
`.trim();

    try {
      await navigator.clipboard.writeText(summary);
      toast.success('Report summary copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const suitableModelsCount = [
    ...recommendations.excellent,
    ...recommendations.good,
    ...recommendations.basic
  ].length;

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
                      <p className="text-sm text-gray-600">Interactive web report with all details</p>
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
                </div>
                
                <button
                  onClick={generateHTMLReport}
                  disabled={isGenerating}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
                      <p className="text-sm text-gray-600">Printable document format</p>
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
                </div>
                
                <button
                  onClick={generatePDFReport}
                  disabled={isGenerating}
                  className="w-full bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isGenerating ? 'Generating...' : 'Download PDF Report'}
                </button>
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
                  className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors"
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
                <p className="text-sm text-gray-600">Step-by-step setup for Ollama, LM Studio, llama.cpp, and HuggingFace</p>
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
                <p>• PDF: ~1-3 MB, compatible with all PDF viewers</p>
                <p>• Reports include all analysis data and instructions</p>
                <p>• Generated files are completely offline and private</p>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Options */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            🔗 Share & Save Options
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-800 mb-2">💾 Save for Later</h4>
              <p className="text-sm text-yellow-700 mb-3">
                Download reports to reference installation steps offline
              </p>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-800 mb-2">📤 Share Results</h4>
              <p className="text-sm text-green-700 mb-3">
                Share compatibility results with team members or forums
              </p>
            </div>
            
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h4 className="font-semibold text-purple-800 mb-2">🔄 Compare Systems</h4>
              <p className="text-sm text-purple-700 mb-3">
                Run analysis on multiple systems and compare capabilities
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ReportDownload;
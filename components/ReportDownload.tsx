import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
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
      
      // Create blob and download
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
    toast.loading('Generating PDF report...', { id: 'pdf-report' });

    try {
      // Create a proper PDF using jsPDF directly instead of html2canvas
      const { jsPDF } = await import('jspdf');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      // PDF styling
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 20;
      const contentWidth = pageWidth - (margin * 2);
      let yPosition = margin;

      // Helper function to add text with proper wrapping
      const addText = (text: string, fontSize = 10, isBold = false) => {
        pdf.setFontSize(fontSize);
        pdf.setFont('helvetica', isBold ? 'bold' : 'normal');
        const lines = pdf.splitTextToSize(text, contentWidth);
        
        // Check if we need a new page
        if (yPosition + (lines.length * fontSize * 0.5) > pageHeight - margin) {
          pdf.addPage();
          yPosition = margin;
        }
        
        pdf.text(lines, margin, yPosition);
        yPosition += lines.length * fontSize * 0.5 + 5;
        return yPosition;
      };

      // Helper function to add a line
      const addLine = () => {
        if (yPosition > pageHeight - margin - 10) {
          pdf.addPage();
          yPosition = margin;
        }
        pdf.setDrawColor(200, 200, 200);
        pdf.line(margin, yPosition, pageWidth - margin, yPosition);
        yPosition += 5;
      };

      // Title
      pdf.setFillColor(59, 130, 246); // Blue background
      pdf.rect(0, 0, pageWidth, 40, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(24);
      pdf.setFont('helvetica', 'bold');
      pdf.text('LLM Hardware Compatibility Report', margin, 25);
      
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Generated on ${new Date().toLocaleDateString()}`, margin, 35);
      
      yPosition = 50;
      pdf.setTextColor(0, 0, 0);

      // System Specifications Section
      addText('SYSTEM SPECIFICATIONS', 16, true);
      addLine();
      
      addText(`Operating System: ${systemInfo.os} (${systemInfo.architecture})`, 12);
      addText(`Processor: ${systemInfo.processor}`, 12);
      addText(`CPU Cores: ${systemInfo.cpuCores} cores`, 12);
      addText(`Memory (RAM): ${systemInfo.totalRamGB} GB total (${systemInfo.availableRamGB} GB available)`, 12);
      addText(`Storage: ${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total`, 12);
      
      if (systemInfo.gpus && systemInfo.gpus.length > 0) {
        systemInfo.gpus.forEach((gpu, index) => {
          const vramInfo = typeof gpu.vramGB === 'number' ? `${gpu.vramGB} GB VRAM` : gpu.vramGB;
          addText(`GPU ${index + 1}: ${gpu.name} (${vramInfo})`, 12);
        });
      } else {
        addText('GPU: None detected', 12);
      }

      yPosition += 10;

      // Compatibility Results Section
      const suitableModels = [
        ...recommendations.excellent,
        ...recommendations.good,
        ...recommendations.basic
      ];

      addText('COMPATIBILITY RESULTS', 16, true);
      addLine();
      
      addText(`Compatible Models: ${suitableModels.length}`, 12);
      addText(`Excellent Performance: ${recommendations.excellent.length}`, 12);
      addText(`Good Performance: ${recommendations.good.length}`, 12);
      addText(`Basic Performance: ${recommendations.basic.length}`, 12);
      addText(`Not Suitable: ${recommendations.not_suitable.length}`, 12);

      yPosition += 10;

      // Recommended Models Section
      if (suitableModels.length > 0) {
        addText('RECOMMENDED MODELS', 16, true);
        addLine();

        const categories = [
          { title: 'EXCELLENT PERFORMANCE', models: recommendations.excellent, emoji: '🟢' },
          { title: 'GOOD PERFORMANCE', models: recommendations.good, emoji: '🟡' },
          { title: 'BASIC PERFORMANCE', models: recommendations.basic, emoji: '🟠' }
        ];

        categories.forEach(category => {
          if (category.models.length > 0) {
            addText(`${category.emoji} ${category.title}`, 14, true);
            
            category.models.slice(0, 5).forEach((model, index) => {
              addText(`${index + 1}. ${model.name}`, 12, true);
              addText(`   Parameters: ${model.specs.parameters}`, 10);
              addText(`   Description: ${model.specs.description}`, 10);
              addText(`   RAM Required: ${model.specs.min_ram_gb}-${model.specs.recommended_ram_gb} GB`, 10);
              addText(`   VRAM Required: ${model.specs.min_vram_gb}-${model.specs.recommended_vram_gb} GB`, 10);
              addText(`   Performance: ${model.compatibility.performance_tier}`, 10);
              
              // Installation command
              if (model.specs.install_methods.ollama) {
                addText(`   Quick Install: ${model.specs.install_methods.ollama.command}`, 10);
              }
              
              if (model.compatibility.recommended_quant) {
                addText(`   Recommended Quantization: ${model.compatibility.recommended_quant}`, 10);
              }
              
              yPosition += 5;
            });
            
            if (category.models.length > 5) {
              addText(`   ... and ${category.models.length - 5} more models`, 10);
            }
            
            yPosition += 5;
          }
        });
      } else {
        // Insufficient Hardware Section
        addText('INSUFFICIENT HARDWARE DETECTED', 16, true);
        addLine();
        
        addText('Your system does not meet the minimum requirements for running local LLMs efficiently.', 12);
        addText('', 12);
        addText('RECOMMENDED CLOUD-BASED SOLUTIONS:', 14, true);
        addText('• ChatGPT: https://chat.openai.com', 12);
        addText('• Claude: https://claude.ai', 12);
        addText('• Google Bard: https://bard.google.com', 12);
        addText('• Perplexity AI: https://perplexity.ai', 12);
      }

      // Installation Platforms Section
      if (yPosition < pageHeight - 60) {
        yPosition += 10;
        addText('INSTALLATION PLATFORMS', 16, true);
        addLine();
        
        addText('1. OLLAMA (Recommended for Beginners)', 12, true);
        addText('   • Download from: https://ollama.ai', 11);
        addText('   • Easy installation, automatic model management', 11);
        addText('   • Usage: ollama run [model-name]', 11);
        addText('', 11);
        
        addText('2. LM STUDIO (GUI Option)', 12, true);
        addText('   • Download from: https://lmstudio.ai', 11);
        addText('   • User-friendly graphical interface', 11);
        addText('   • No command line needed', 11);
        addText('', 11);
        
        addText('3. LLAMA.CPP (Advanced Users)', 12, true);
        addText('   • Best for CPU optimization', 11);
        addText('   • Requires technical knowledge', 11);
        addText('   • Download GGUF models manually', 11);
      }

      // System Capability Level
      const capabilityLevel = recommender.getSystemCapabilityLevel();
      const levelLabels = {
        low: 'Entry Level',
        medium: 'Mid Range', 
        high: 'High End',
        premium: 'Premium'
      };

      // Add new page for optimization tips
      pdf.addPage();
      yPosition = margin;
      
      addText('OPTIMIZATION TIPS FOR YOUR SYSTEM', 16, true);
      addLine();
      
      addText(`System Capability Level: ${levelLabels[capabilityLevel] || 'Unknown'}`, 14, true);
      yPosition += 5;

      // Get and add optimization tips
      const tips = recommender.getOptimizationTips();
      tips.slice(0, 10).forEach((tip, index) => {
        // Remove emoji and format tip
        const cleanTip = tip.replace(/^[🔧💾🎮💻🍎🐧🪟📱⚡🗑️❌✅🚀💡⚙️🔋☁️📁]+\s*/, '');
        addText(`• ${cleanTip}`, 11);
      });

      // Footer
      pdf.setFontSize(8);
      pdf.setTextColor(128, 128, 128);
      const footerY = pageHeight - 10;
      pdf.text('Generated by LLM Hardware Compatibility Checker', margin, footerY);
      pdf.text(`Report created on ${new Date().toLocaleString()}`, pageWidth - margin - 60, footerY);

      // Save the PDF
      pdf.save(`llm-compatibility-report-${new Date().toISOString().split('T')[0]}.pdf`);

      setGeneratedReports(prev => ({ ...prev, pdf: true }));
      toast.success('PDF report downloaded!', { id: 'pdf-report' });

    } catch (error) {
      console.error('Failed to generate PDF report:', error);
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
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
    toast.loading('Generating high-quality PDF report...', { id: 'pdf-report' });

    try {
      // Create PDF with proper formatting
      const { jsPDF } = await import('jspdf');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      // Define colors and fonts
      const colors = {
        primary: [59, 130, 246], // blue-500
        secondary: [107, 114, 128], // gray-500
        success: [16, 185, 129], // green-500
        warning: [245, 158, 11], // yellow-500
        danger: [239, 68, 68], // red-500
        text: [31, 41, 55], // gray-800
        lightGray: [249, 250, 251] // gray-50
      };

      let yPosition = 20;
      const pageWidth = 210; // A4 width in mm
      const marginLeft = 20;
      const marginRight = 20;
      const contentWidth = pageWidth - marginLeft - marginRight;

      // Helper functions
      const addText = (text: string, x: number, y: number, options: any = {}) => {
        const fontSize = options.fontSize || 10;
        const fontStyle = options.fontStyle || 'normal';
        const align = options.align || 'left';
        const color = options.color || colors.text;
        
        pdf.setFontSize(fontSize);
        pdf.setFont('helvetica', fontStyle);
        pdf.setTextColor(color[0], color[1], color[2]);
        pdf.text(text, x, y, { align });
        return y + (fontSize * 0.35) + (options.spacing || 2);
      };

      const addHeading = (text: string, level: number = 1) => {
        const fontSize = level === 1 ? 18 : level === 2 ? 14 : 12;
        const spacing = level === 1 ? 8 : 6;
        yPosition = addText(text, marginLeft, yPosition, {
          fontSize,
          fontStyle: 'bold',
          color: colors.primary,
          spacing
        });
        return yPosition;
      };

      const addLine = () => {
        pdf.setDrawColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
        pdf.setLineWidth(0.2);
        pdf.line(marginLeft, yPosition, pageWidth - marginRight, yPosition);
        yPosition += 5;
      };

      const checkPageBreak = (spaceNeeded: number = 20) => {
        if (yPosition + spaceNeeded > 280) { // Near bottom of A4
          pdf.addPage();
          yPosition = 20;
        }
      };

      const addSection = (title: string, content: string[]) => {
        checkPageBreak(30);
        addHeading(title, 2);
        addLine();
        
        content.forEach(line => {
          checkPageBreak();
          if (line.startsWith('•')) {
            yPosition = addText(line, marginLeft + 5, yPosition);
          } else if (line.startsWith('  -')) {
            yPosition = addText(line, marginLeft + 10, yPosition, { fontSize: 9 });
          } else {
            yPosition = addText(line, marginLeft, yPosition);
          }
        });
        yPosition += 5;
      };

      // Title Page
      pdf.setFillColor(colors.primary[0], colors.primary[1], colors.primary[2]);
      pdf.rect(0, 0, pageWidth, 60, 'F');
      
      yPosition = addText('🤖 LLM Hardware Compatibility Report', pageWidth/2, 25, {
        fontSize: 20,
        fontStyle: 'bold',
        color: [255, 255, 255],
        align: 'center'
      });
      
      yPosition = addText(`Generated on ${new Date().toLocaleDateString()}`, pageWidth/2, yPosition + 5, {
        fontSize: 12,
        color: [255, 255, 255],
        align: 'center'
      });

      yPosition = 80;

      // System Specifications
      addHeading('System Specifications');
      addLine();
      
      const systemSpecs = [
        `Operating System: ${systemInfo.os} (${systemInfo.architecture})`,
        `Processor: ${systemInfo.processor}`,
        `CPU Cores: ${systemInfo.cpuCores}`,
        `Memory (RAM): ${systemInfo.totalRamGB} GB total (${systemInfo.availableRamGB} GB available)`,
        `Storage: ${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total`,
      ];

      if (systemInfo.gpus && systemInfo.gpus.length > 0) {
        systemInfo.gpus.forEach((gpu, index) => {
          const vramInfo = typeof gpu.vramGB === 'number' ? `${gpu.vramGB} GB VRAM` : gpu.vramGB;
          systemSpecs.push(`GPU ${index + 1}: ${gpu.name} (${vramInfo})`);
        });
      } else {
        systemSpecs.push('GPU: None detected');
      }

      systemSpecs.forEach(spec => {
        checkPageBreak();
        yPosition = addText(`• ${spec}`, marginLeft, yPosition);
      });

      yPosition += 10;

      // Compatibility Summary
      const suitableModels = [
        ...recommendations.excellent,
        ...recommendations.good,
        ...recommendations.basic
      ];

      addHeading('Compatibility Summary');
      addLine();

      const summaryStats = [
        `Compatible Models: ${suitableModels.length}`,
        `Excellent Performance: ${recommendations.excellent.length}`,
        `Good Performance: ${recommendations.good.length}`,
        `Basic Performance: ${recommendations.basic.length}`,
        `Not Suitable: ${recommendations.not_suitable.length}`
      ];

      summaryStats.forEach(stat => {
        checkPageBreak();
        yPosition = addText(`• ${stat}`, marginLeft, yPosition);
      });

      yPosition += 10;

      // Model Recommendations
      if (suitableModels.length > 0) {
        const categories = [
          { name: 'Excellent Performance', models: recommendations.excellent, color: colors.success },
          { name: 'Good Performance', models: recommendations.good, color: colors.warning },
          { name: 'Basic Performance', models: recommendations.basic, color: colors.danger }
        ];

        categories.forEach(category => {
          if (category.models.length > 0) {
            checkPageBreak(40);
            yPosition = addText(category.name, marginLeft, yPosition, {
              fontSize: 14,
              fontStyle: 'bold',
              color: category.color
            });
            addLine();

            category.models.slice(0, 5).forEach(model => { // Limit to 5 models per category for space
              checkPageBreak(25);
              
              // Model name
              yPosition = addText(model.name, marginLeft, yPosition, {
                fontSize: 12,
                fontStyle: 'bold'
              });

              // Model details
              const details = [
                `  Parameters: ${model.specs.parameters}`,
                `  RAM Required: ${model.specs.min_ram_gb}-${model.specs.recommended_ram_gb} GB`,
                `  VRAM Required: ${model.specs.min_vram_gb}-${model.specs.recommended_vram_gb} GB`,
                `  Description: ${model.specs.description}`
              ];

              if (model.specs.domain) {
                details.push(`  Domain: ${model.specs.domain}`);
              }

              // Installation command
              if (model.specs.install_methods.ollama) {
                details.push(`  Quick Install: ${model.specs.install_methods.ollama.command}`);
              }

              details.forEach(detail => {
                checkPageBreak();
                yPosition = addText(detail, marginLeft, yPosition, { fontSize: 9 });
              });

              yPosition += 3;
            });

            if (category.models.length > 5) {
              yPosition = addText(`... and ${category.models.length - 5} more models`, marginLeft, yPosition, {
                fontSize: 9,
                color: colors.secondary
              });
            }

            yPosition += 5;
          }
        });
      } else {
        // Insufficient hardware section
        addSection('Insufficient Hardware Detected', [
          'Your system does not meet the minimum requirements for running local LLMs efficiently.',
          '',
          'Recommended cloud-based solutions:',
          '• ChatGPT: https://chat.openai.com',
          '• Claude: https://claude.ai',
          '• Google Bard: https://bard.google.com',
          '• Perplexity AI: https://perplexity.ai'
        ]);
      }

      // Installation Platforms
      checkPageBreak(50);
      addHeading('Installation Platforms');
      addLine();

      const platforms = recommender.getInstallationPlatforms();
      platforms.forEach(platform => {
        checkPageBreak(20);
        yPosition = addText(platform.name, marginLeft, yPosition, {
          fontSize: 12,
          fontStyle: 'bold'
        });
        yPosition = addText(`  ${platform.description}`, marginLeft, yPosition, { fontSize: 9 });
        yPosition = addText(`  Best for: ${platform.bestFor}`, marginLeft, yPosition, { fontSize: 9 });
        yPosition += 2;
      });

      // Optimization Tips
      checkPageBreak(50);
      addHeading('Optimization Tips');
      addLine();

      const tips = recommender.getOptimizationTips();
      tips.slice(0, 10).forEach(tip => { // Limit tips for space
        checkPageBreak();
        yPosition = addText(`• ${tip.replace(/^[🔧💾🎮💻🍎🐧🪟📱⚡🗑️❌✅🚀💡⚙️🔋☁️📁]+\s*/, '')}`, marginLeft, yPosition, { fontSize: 9 });
      });

      // Footer
      checkPageBreak(30);
      yPosition = Math.max(yPosition + 20, 250);
      pdf.setDrawColor(colors.secondary[0], colors.secondary[1], colors.secondary[2]);
      pdf.setLineWidth(0.5);
      pdf.line(marginLeft, yPosition, pageWidth - marginRight, yPosition);
      yPosition += 10;
      
      yPosition = addText('Generated by LLM Hardware Compatibility Checker', pageWidth/2, yPosition, {
        fontSize: 10,
        align: 'center',
        color: colors.secondary
      });
      
      yPosition = addText(`Report created on ${new Date().toLocaleString()}`, pageWidth/2, yPosition, {
        fontSize: 8,
        align: 'center',
        color: colors.secondary
      });

      // Save the PDF
      pdf.save(`llm-compatibility-report-${new Date().toISOString().split('T')[0]}.pdf`);

      setGeneratedReports(prev => ({ ...prev, pdf: true }));
      toast.success('High-quality PDF report downloaded!', { id: 'pdf-report' });
    } catch (error) {
      console.error('Failed to generate PDF report:', error);
      toast.error('Failed to generate PDF report. Please try again.', { id: 'pdf-report' });
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
                      <p className="text-sm text-gray-600">High-quality document with selectable text</p>
                    </div>
                  </div>
                  {generatedReports.pdf && (
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  )}
                </div>
                
                <div className="flex flex-wrap gap-2 mb-3">
                  <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs font-medium">
                    High Quality
                  </span>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                    Selectable Text
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
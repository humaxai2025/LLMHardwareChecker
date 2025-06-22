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

  // ---- UPDATED PDF GENERATOR FUNCTION ----
  const generatePDFReport = async () => {
    setIsGenerating(true);
    toast.loading('Generating PDF report...', { id: 'pdf-report' });

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

      // Helper to add headings
      const heading = (text, size = 16) => {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(size);
        pdf.text(text, margin, y);
        y += size * 0.7;
      };
      // Helper to add subheadings
      const subheading = (text, size = 13) => {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(size);
        pdf.text(text, margin, y);
        y += size * 0.7;
      };
      // Helper to add normal text
      const para = (text, size = 11) => {
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(size);
        const lines = pdf.splitTextToSize(text, contentWidth);
        pdf.text(lines, margin, y);
        y += lines.length * size * 0.5 + 2;
      };
      // Helper to add a table row
      const tableRow = (columns, colWidths, size = 10, bold = false) => {
        pdf.setFont('helvetica', bold ? 'bold' : 'normal');
        pdf.setFontSize(size);
        let x = margin;
        columns.forEach((col, i) => {
          pdf.text(String(col), x, y, { maxWidth: colWidths[i] - 2 });
          x += colWidths[i];
        });
        y += size * 0.6;
      };
      // New page if needed
      const ensureSpace = rows => {
        if (y > pageHeight - (rows * 8 + margin)) {
          pdf.addPage();
          y = margin;
        }
      };

      // TITLE & DATE
      pdf.setFillColor(59, 130, 246); // blue bar
      pdf.rect(0, 0, pageWidth, 20, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(18);
      pdf.setFont('helvetica', 'bold');
      pdf.text('LLM Hardware Compatibility Report', margin, 13);
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(11);
      pdf.text(`Generated: ${new Date().toLocaleString()}`, pageWidth - margin - 70, 13);
      y = 25;
      pdf.setTextColor(0, 0, 0);

      // SYSTEM SPECS
      heading('🖥️ System Specifications');
      para(`OS: ${systemInfo.os} (${systemInfo.architecture})`);
      para(`Processor: ${systemInfo.processor} (${systemInfo.cpuCores} cores)`);
      para(`RAM: ${systemInfo.totalRamGB} GB total (${systemInfo.availableRamGB} GB available)`);
      para(`Storage: ${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total`);
      if (systemInfo.gpus && systemInfo.gpus.length > 0) {
        systemInfo.gpus.forEach((gpu, i) => {
          para(`GPU ${i + 1}: ${gpu.name} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB})`);
        });
      } else {
        para('GPU: None detected');
      }
      y += 2;

      // MODEL COMPATIBILITY MATRIX (as table)
      heading('📊 Compatibility Matrix');
      ensureSpace(6);
      // Table header
      tableRow(
        ['Model', 'Params', 'Perf.', 'RAM GB', 'VRAM GB', 'Install'],
        [40, 22, 18, 20, 22, 38],
        10,
        true
      );
      pdf.setDrawColor(200, 200, 200);
      pdf.line(margin, y - 3, pageWidth - margin, y - 3);

      const allModels = [
        ...recommendations.excellent,
        ...recommendations.good,
        ...recommendations.basic,
        ...recommendations.not_suitable,
      ];

      for (const model of allModels) {
        ensureSpace(1);
        tableRow(
          [
            model.name,
            model.specs.parameters,
            model.compatibility.performance_tier,
            `${model.specs.min_ram_gb}-${model.specs.recommended_ram_gb}`,
            `${model.specs.min_vram_gb || '-'}-${model.specs.recommended_vram_gb || '-'}`,
            (model.specs.install_methods.ollama?.command || '').replace('ollama run ', 'run ')
          ],
          [40, 22, 18, 20, 22, 38],
          9,
          false
        );
      }
      y += 4;

      // RECOMMENDATIONS
      heading('🤖 Model Recommendations');
      const cats = [
        { title: '🟢 Excellent', arr: recommendations.excellent },
        { title: '🟡 Good', arr: recommendations.good },
        { title: '🟠 Basic', arr: recommendations.basic },
      ];
      for (const cat of cats) {
        if (cat.arr.length > 0) {
          subheading(cat.title, 12);
          cat.arr.forEach((model, idx) => {
            para(`${idx + 1}. ${model.name} (${model.specs.parameters}) - ${model.compatibility.performance_tier}`);
            para(`   RAM: ${model.specs.min_ram_gb}-${model.specs.recommended_ram_gb}GB, VRAM: ${model.specs.min_vram_gb || '-'}-${model.specs.recommended_vram_gb || '-'}`);
            if (model.specs.install_methods.ollama) {
              para(`   Ollama: ${model.specs.install_methods.ollama.command}`);
            }
            if (model.specs.install_methods.lm_studio) {
              para(`   LM Studio: ${model.specs.install_methods.lm_studio.command || model.specs.install_methods.lm_studio.download_url || ''}`);
            }
            if (model.specs.install_methods.llamacpp) {
              para(`   llama.cpp: ${model.specs.install_methods.llamacpp.command || model.specs.install_methods.llamacpp.download_url || ''}`);
            }
            if (model.specs.install_methods.huggingface) {
              para(`   HuggingFace: ${model.specs.install_methods.huggingface.command || model.specs.install_methods.huggingface.download_url || ''}`);
            }
            if (model.compatibility.recommended_quant) {
              para(`   Quantization: ${model.compatibility.recommended_quant}`);
            }
            y += 2;
          });
        }
      }
      if (
        recommendations.excellent.length +
          recommendations.good.length +
          recommendations.basic.length ===
        0
      ) {
        para('No compatible models for your specs. Try cloud-based LLMs like ChatGPT or Claude.');
      }
      y += 4;

      // INSTALLATION GUIDE
      heading('🛠️ Installation Guide');
      subheading('1. Ollama (Recommended for Beginners)');
      para('• Download: https://ollama.ai');
      para('• Simple CLI tool, automatic model download & management.');
      para('• Usage: ollama run <model>');
      y += 2;

      subheading('2. LM Studio (GUI Option)');
      para('• Download: https://lmstudio.ai');
      para('• User-friendly graphical interface, no CLI needed.');

      subheading('3. llama.cpp (Advanced Users)');
      para('• Download and usage: https://github.com/ggerganov/llama.cpp');
      para('• Best for CPU, needs GGUF models, requires more technical steps.');

      subheading('4. HuggingFace (Advanced/Developers)');
      para('• Website: https://huggingface.co/models');
      para('• Model downloads and instructions available per model page.');

      y += 6;

      // OPTIMIZATION TIPS
      heading('💡 Optimization Tips');
      const tips = recommender.getOptimizationTips();
      tips.forEach((tip, idx) => {
        para(`${idx + 1}. ${tip.replace(/^[🔧💾🎮💻🍎🐧🪟📱⚡🗑️❌✅🚀💡⚙️🔋☁️📁]+\s*/, '')}`);
      });
      y += 4;

      // FOOTER
      if (y > pageHeight - 20) {
        pdf.addPage();
        y = margin;
      }
      pdf.setFontSize(8);
      pdf.setTextColor(120, 120, 120);
      pdf.text(
        'Generated by LLM Hardware Compatibility Checker | https://your-app-link-here',
        margin,
        pageHeight - 10
      );
      pdf.text(
        `Report created on ${new Date().toLocaleString()}`,
        pageWidth - margin - 70,
        pageHeight - 10
      );

      // Save
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
  // ---- END UPDATED PDF GENERATOR ----

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

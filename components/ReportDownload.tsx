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

  // Function to wait for fonts to load
  const waitForFontsToLoad = async (): Promise<void> => {
    if ('fonts' in document) {
      try {
        await document.fonts.ready;
        // Additional wait to ensure fonts are fully rendered
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.warn('Font loading check failed:', error);
        // Fallback wait
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    } else {
      // Fallback for browsers without document.fonts
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
  };

  const generatePDFReportOptimized = async () => {
    setIsGenerating(true);
    toast.loading('Generating PDF report...', { id: 'pdf-report' });

    try {
      const reportGenerator = new ReportGenerator({
        systemInfo,
        recommendations,
        recommender,
        timestamp: new Date()
      });

      // Generate HTML content with PDF-optimized styles
      let htmlContent = reportGenerator.generateHTMLReport();
      
      // Replace the CSS with PDF-optimized version
      htmlContent = htmlContent.replace(
        /<style>[\s\S]*?<\/style>/,
        `<style>${getPDFOptimizedCSS()}</style>`
      );

      // Create temporary iframe for better rendering
      const iframe = document.createElement('iframe');
      iframe.style.position = 'fixed';
      iframe.style.top = '-9999px';
      iframe.style.left = '-9999px';
      iframe.style.width = '794px'; // A4 width
      iframe.style.height = '1123px'; // A4 height
      iframe.style.border = 'none';
      iframe.style.backgroundColor = 'white';
      
      document.body.appendChild(iframe);
      
      // Write content to iframe
      iframe.contentDocument?.open();
      iframe.contentDocument?.write(htmlContent);
      iframe.contentDocument?.close();

      // Wait for fonts and content to load
      await waitForFontsToLoad();
      
      // Additional wait for iframe content
      await new Promise(resolve => setTimeout(resolve, 2000));

      const iframeDocument = iframe.contentDocument;
      const iframeBody = iframeDocument?.body;

      if (!iframeBody) {
        throw new Error('Failed to access iframe content');
      }

      // Generate PDF using html2canvas with optimized settings
      const canvas = await html2canvas(iframeBody, {
        scale: 2,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff',
        width: 794,
        height: iframeBody.scrollHeight,
        logging: false,
        imageTimeout: 15000,
        foreignObjectRendering: true,
        ignoreElements: (element) => {
          // Ignore certain elements that cause issues
          return element.classList?.contains('copy-btn') || false;
        }
      });

      // Calculate dimensions for PDF
      const imgWidth = 210; // A4 width in mm
      const pageHeight = 297; // A4 height in mm
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      let heightLeft = imgHeight;

      const pdf = new jsPDF('p', 'mm', 'a4');
      let position = 0;

      // Add image to PDF
      const imgData = canvas.toDataURL('image/jpeg', 0.95);
      pdf.addImage(imgData, 'JPEG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      // Add additional pages if needed
      while (heightLeft >= 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(imgData, 'JPEG', 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      // Save PDF
      pdf.save(`llm-compatibility-report-${new Date().toISOString().split('T')[0]}.pdf`);

      // Cleanup
      document.body.removeChild(iframe);

      setGeneratedReports(prev => ({ ...prev, pdf: true }));
      toast.success('PDF report downloaded!', { id: 'pdf-report' });
    } catch (error) {
      console.error('Failed to generate PDF report:', error);
      toast.error('Failed to generate PDF report. Try HTML format instead.', { id: 'pdf-report' });
    } finally {
      setIsGenerating(false);
    }
  };

  // PDF-optimized CSS that replaces Google Fonts with system fonts
  const getPDFOptimizedCSS = (): string => {
    return `
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
    
    --border-radius-sm: 0.375rem;
    --border-radius-md: 0.5rem;
    --border-radius-lg: 0.75rem;
    --border-radius-xl: 1rem;
    --border-radius-2xl: 1.5rem;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
    color: #1f2937;
    background: white;
    font-size: 14px;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

.container {
    max-width: 100%;
    margin: 0;
    padding: 20px;
    background: white;
}

.header {
    text-align: center;
    margin-bottom: 30px;
    padding: 30px 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -0.025em;
    line-height: 1.1;
}

.header .subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 400;
    letter-spacing: 0.025em;
}

.section {
    margin-bottom: 30px;
    padding: 25px;
    background: #f9fafb;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    page-break-inside: avoid;
}

.section h2 {
    color: #1d4ed8;
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.025em;
}

.specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.spec-item {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
    border: 1px solid #e5e7eb;
    page-break-inside: avoid;
}

.spec-label {
    font-weight: 600;
    color: #4b5563;
    margin-bottom: 8px;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.spec-value {
    color: #111827;
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1.4;
}

.summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.stat-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e5e7eb;
    page-break-inside: avoid;
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 800;
    color: #2563eb;
    margin-bottom: 8px;
    line-height: 1;
    letter-spacing: -0.05em;
}

.stat-label {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.025em;
}

.model-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    page-break-inside: avoid;
}

.model-header {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 15px;
}

.model-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #111827;
    flex: 1;
    letter-spacing: -0.025em;
}

.model-domain {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

.performance-tier {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.875rem;
    margin: 12px 0;
    letter-spacing: 0.025em;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

.performance-tier.excellent {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
}

.performance-tier.good {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
}

.performance-tier.basic {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
}

.requirements {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 10px;
    margin: 15px 0;
    padding: 15px;
    background: #f9fafb;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

.req-item {
    text-align: center;
    padding: 12px;
    background: white;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}

.req-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.req-value {
    font-weight: 600;
    color: #111827;
    font-size: 0.875rem;
}

.installation-methods {
    margin-top: 20px;
    padding: 20px;
    background: #f9fafb;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

.install-method {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 15px;
    margin: 12px 0;
    page-break-inside: avoid;
}

.install-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 12px;
}

.install-title {
    font-weight: 600;
    color: #111827;
    font-size: 1.1rem;
    letter-spacing: -0.025em;
}

.install-badge {
    padding: 4px 12px;
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
    background: #1f2937;
    color: #f9fafb;
    padding: 12px 16px;
    border-radius: 6px;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 12px 0;
    overflow-wrap: break-word;
    word-break: break-all;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

.install-command::before {
    content: '$ ';
    color: #10b981;
    font-weight: 600;
}

.install-note {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 4px;
    font-size: 0.875rem;
    line-height: 1.6;
}

.insufficient-hardware {
    background: linear-gradient(135deg, #fecaca, #fca5a5);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 25px;
    margin: 25px 0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

.insufficient-hardware h3 {
    color: #991b1b;
    margin-bottom: 15px;
    font-weight: 700;
}

.cloud-solutions {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 15px;
}

.cloud-solution {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
    text-decoration: none;
    color: inherit;
    display: block;
    page-break-inside: avoid;
}

.cloud-solution h4 {
    color: #111827;
    margin-bottom: 8px;
    font-weight: 600;
}

.cloud-solution p {
    color: #6b7280;
    font-size: 0.875rem;
    line-height: 1.5;
}

.footer {
    text-align: center;
    padding: 25px;
    background: #f9fafb;
    border-radius: 8px;
    margin-top: 30px;
    color: #6b7280;
    border: 1px solid #e5e7eb;
}

.tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.tip-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #10b981;
    border: 1px solid #e5e7eb;
    page-break-inside: avoid;
}

.tip-card h4 {
    color: #111827;
    margin-bottom: 12px;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.025em;
}

.tip-card ul {
    margin-left: 20px;
    color: #4b5563;
}

.tip-card li {
    margin-bottom: 6px;
    line-height: 1.6;
}

/* Hide copy buttons in PDF */
.copy-btn {
    display: none !important;
}

/* Ensure proper page breaks */
.model-card, .install-method, .tip-card, .spec-item {
    break-inside: avoid;
    page-break-inside: avoid;
}

/* Print-specific styles */
@media print {
    body { 
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
    
    * {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
}
`;
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
                    High Quality
                  </span>
                </div>
                
                <button
                  onClick={generatePDFReportOptimized}
                  disabled={isGenerating}
                  className="w-full bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isGenerating ? 'Generating PDF...' : 'Download PDF Report'}
                </button>
                
                <p className="text-xs text-gray-500 mt-2">
                  ✨ Optimized fonts and formatting for professional PDFs
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
                <p>• PDF: ~1-3 MB, optimized fonts and formatting</p>
                <p>• Professional layout suitable for printing</p>
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
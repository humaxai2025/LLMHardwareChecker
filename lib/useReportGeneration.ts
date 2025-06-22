// useReportGeneration.ts - React hook for managing report generation

import { useState, useCallback, useRef } from 'react';
import { toast } from 'react-hot-toast';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { 
  ReportUtils, 
  ReportGenerationError, 
  PDFGenerationError,
  ReportGenerationProgress,
  REPORT_CONSTANTS
} from './reportUtils';

interface UseReportGenerationProps {
  systemInfo: any;
  recommendations: any;
  recommender: any;
  onSuccess?: (type: 'html' | 'pdf' | 'summary') => void;
  onError?: (error: Error, type: 'html' | 'pdf' | 'summary') => void;
}

interface ReportGenerationState {
  isGenerating: boolean;
  progress: number;
  stage: string;
  error: string | null;
  generatedReports: {
    html?: string;
    pdf?: boolean;
    summary?: boolean;
  };
}

export const useReportGeneration = ({
  systemInfo,
  recommendations,
  recommender,
  onSuccess,
  onError
}: UseReportGenerationProps) => {
  const [state, setState] = useState<ReportGenerationState>({
    isGenerating: false,
    progress: 0,
    stage: 'Ready',
    error: null,
    generatedReports: {}
  });

  const progressTracker = useRef(new ReportGenerationProgress());
  const abortController = useRef<AbortController | null>(null);

  // Update state when progress changes
  useState(() => {
    progressTracker.current.addListener((progress, stage) => {
      setState(prev => ({ ...prev, progress, stage }));
    });
  });

  const updateState = useCallback((updates: Partial<ReportGenerationState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  const validateData = useCallback(() => {
    const validation = ReportUtils.validateReportData({
      systemInfo,
      recommendations,
      recommender
    });

    if (!validation.isValid) {
      throw new ReportGenerationError(
        `Data validation failed: ${validation.errors.join(', ')}`,
        'validation'
      );
    }

    if (validation.warnings.length > 0) {
      console.warn('Report generation warnings:', validation.warnings);
      toast.custom((t) => (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 shadow-lg">
          <h4 className="text-yellow-800 font-semibold mb-2">⚠️ Generation Warnings</h4>
          <ul className="text-yellow-700 text-sm space-y-1">
            {validation.warnings.map((warning, idx) => (
              <li key={idx}>• {warning}</li>
            ))}
          </ul>
        </div>
      ), { duration: 5000 });
    }

    return validation;
  }, [systemInfo, recommendations, recommender]);

  const generateHTMLReport = useCallback(async () => {
    if (state.isGenerating) return;

    updateState({ isGenerating: true, error: null });
    progressTracker.current.reset();
    abortController.current = new AbortController();

    const toastId = toast.loading('Generating HTML report...');
    const resources: any = {};

    try {
      progressTracker.current.updateProgress(10, 'Validating data...');
      validateData();

      progressTracker.current.updateProgress(30, 'Generating HTML content...');
      
      // Dynamic import to avoid bundling issues
      const { ReportGenerator } = await import('./reportGenerator');
      
      const reportGenerator = new ReportGenerator({
        systemInfo,
        recommendations,
        recommender,
        timestamp: new Date()
      });

      const htmlContent = reportGenerator.generateHTMLReport();
      
      if (abortController.current?.signal.aborted) {
        throw new ReportGenerationError('Report generation was cancelled', 'generation');
      }

      progressTracker.current.updateProgress(70, 'Creating download...');
      
      // Create blob with proper encoding
      const blob = new Blob([htmlContent], { 
        type: REPORT_CONSTANTS.HTML_MIME_TYPE 
      });
      resources.blob = blob;

      // Generate filename
      const filename = ReportUtils.generateFilename(
        REPORT_CONSTANTS.DEFAULT_HTML_FILENAME,
        'html'
      );

      progressTracker.current.updateProgress(90, 'Downloading file...');

      // Create and trigger download
      const url = URL.createObjectURL(blob);
      resources.url = url;
      
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.style.display = 'none';
      resources.link = link;
      
      document.body.appendChild(link);
      link.click();

      progressTracker.current.updateProgress(100, 'Complete!');

      updateState({
        isGenerating: false,
        generatedReports: { ...state.generatedReports, html: htmlContent }
      });

      toast.success('HTML report downloaded successfully!', { id: toastId });
      onSuccess?.('html');

    } catch (error) {
      console.error('HTML generation failed:', error);
      
      const errorMessage = error instanceof ReportGenerationError 
        ? error.message 
        : 'Failed to generate HTML report. Please try again.';
      
      updateState({ isGenerating: false, error: errorMessage });
      toast.error(errorMessage, { id: toastId });
      onError?.(error as Error, 'html');
    } finally {
      ReportUtils.cleanup(resources);
      abortController.current = null;
    }
  }, [state, systemInfo, recommendations, recommender, validateData, updateState, onSuccess, onError]);

  const generatePDFReport = useCallback(async () => {
    if (state.isGenerating) return;

    // Check browser compatibility first
    const compatibility = ReportUtils.checkBrowserCompatibility();
    if (!compatibility.canGeneratePDF) {
      const error = new PDFGenerationError('PDF generation not supported in this browser');
      updateState({ error: error.message });
      toast.error('PDF generation requires a modern browser with Canvas support');
      onError?.(error, 'pdf');
      return;
    }

    updateState({ isGenerating: true, error: null });
    progressTracker.current.reset();
    abortController.current = new AbortController();

    const toastId = toast.loading('Generating PDF report... This may take a moment.');
    const resources: any = {};

    try {
      progressTracker.current.updateProgress(5, 'Validating data...');
      validateData();

      progressTracker.current.updateProgress(15, 'Generating HTML content...');
      
      // Dynamic import to avoid bundling issues
      const { ReportGenerator } = await import('./reportGenerator');
      
      const reportGenerator = new ReportGenerator({
        systemInfo,
        recommendations,
        recommender,
        timestamp: new Date()
      });

      let htmlContent = reportGenerator.generateHTMLReport();

      if (abortController.current?.signal.aborted) {
        throw new PDFGenerationError('PDF generation was cancelled');
      }

      progressTracker.current.updateProgress(25, 'Creating rendering environment...');

      // Create iframe for isolated rendering
      const iframe = ReportUtils.createRenderingIframe();
      resources.iframe = iframe;

      // Wait for iframe to be ready
      await new Promise<void>((resolve) => {
        iframe.onload = () => resolve();
        iframe.src = 'about:blank';
      });

      const iframeDoc = iframe.contentDocument!;
      iframeDoc.open();
      iframeDoc.write(htmlContent);
      iframeDoc.close();

      progressTracker.current.updateProgress(40, 'Loading fonts and styles...');

      // Wait for fonts and content to load
      await ReportUtils.waitForFonts(REPORT_CONSTANTS.FONT_LOAD_TIMEOUT_MS);

      if (abortController.current?.signal.aborted) {
        throw new PDFGenerationError('PDF generation was cancelled');
      }

      progressTracker.current.updateProgress(60, 'Rendering to canvas...');

      const targetElement = iframeDoc.body;
      if (!targetElement) {
        throw new PDFGenerationError('Could not access document content for rendering');
      }

      // Generate high-quality canvas
      const canvas = await html2canvas(targetElement, {
        scale: REPORT_CONSTANTS.PDF_SCALE,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff',
        width: REPORT_CONSTANTS.CANVAS_MAX_WIDTH,
        height: Math.min(targetElement.scrollHeight, REPORT_CONSTANTS.CANVAS_MAX_HEIGHT),
        logging: false,
        imageTimeout: 15000,
        foreignObjectRendering: true,
        onclone: (clonedDoc) => {
          // Remove interactive elements for PDF
          const interactiveElements = clonedDoc.querySelectorAll('.copy-btn, button, input, select');
          interactiveElements.forEach(el => el.remove());
          
          // Ensure print styles are applied
          const style = clonedDoc.createElement('style');
          style.textContent = `
            * { 
              -webkit-print-color-adjust: exact !important; 
              print-color-adjust: exact !important; 
            }
            body { background: white !important; }
            .copy-btn { display: none !important; }
          `;
          clonedDoc.head.appendChild(style);
        }
      });

      if (abortController.current?.signal.aborted) {
        throw new PDFGenerationError('PDF generation was cancelled');
      }

      progressTracker.current.updateProgress(80, 'Creating PDF document...');

      // Create PDF with optimal settings
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
        compress: true
      });

      const imgData = canvas.toDataURL('image/jpeg', REPORT_CONSTANTS.PDF_QUALITY);
      const pdfWidth = 210; // A4 width in mm
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      // Handle multiple pages
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

      progressTracker.current.updateProgress(95, 'Downloading PDF...');

      // Generate filename and save
      const filename = ReportUtils.generateFilename(
        REPORT_CONSTANTS.DEFAULT_PDF_FILENAME,
        'pdf'
      );
      
      pdf.save(filename);

      progressTracker.current.updateProgress(100, 'Complete!');

      updateState({
        isGenerating: false,
        generatedReports: { ...state.generatedReports, pdf: true }
      });

      toast.success('PDF report generated and downloaded successfully!', { id: toastId });
      onSuccess?.('pdf');

    } catch (error) {
      console.error('PDF generation failed:', error);
      
      let errorMessage = 'PDF generation failed. ';
      
      if (error instanceof PDFGenerationError) {
        errorMessage += error.message;
      } else if (error instanceof Error) {
        errorMessage += error.message.includes('canvas') 
          ? 'Canvas rendering failed. Try the HTML format instead.'
          : 'Please try the HTML format instead.';
      } else {
        errorMessage += 'Please try again or use the HTML format.';
      }

      updateState({ isGenerating: false, error: errorMessage });
      toast.error(errorMessage, { id: toastId, duration: 6000 });
      onError?.(error as Error, 'pdf');
    } finally {
      ReportUtils.cleanup(resources);
      abortController.current = null;
    }
  }, [state, systemInfo, recommendations, recommender, validateData, updateState, onSuccess, onError]);

  const copyReportSummary = useCallback(async () => {
    if (state.isGenerating) return;

    try {
      validateData();

      const suitableModels = [
        ...(recommendations?.excellent || []),
        ...(recommendations?.good || []),
        ...(recommendations?.basic || [])
      ];

      const summary = `
LLM Hardware Compatibility Report
Generated: ${new Date().toLocaleDateString()}

SYSTEM SPECIFICATIONS:
- OS: ${systemInfo?.os || 'Unknown'} (${systemInfo?.architecture || 'Unknown'})
- CPU: ${systemInfo?.processor || 'Unknown'} (${systemInfo?.cpuCores || 'Unknown'} cores)
- RAM: ${systemInfo?.totalRamGB || 'Unknown'} GB total, ${systemInfo?.availableRamGB || 'Unknown'} GB available
- Storage: ${systemInfo?.freeStorageGB || 'Unknown'} GB free / ${systemInfo?.totalStorageGB || 'Unknown'} GB total
- GPUs: ${systemInfo?.gpus?.length || 0} detected
${systemInfo?.gpus?.map((gpu: any) => 
  `  - ${gpu.name || 'Unknown GPU'} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB || 'Unknown VRAM'})`
).join('\n') || ''}

COMPATIBILITY RESULTS:
- Compatible Models: ${suitableModels.length}
- Excellent Performance: ${(recommendations?.excellent || []).length}
- Good Performance: ${(recommendations?.good || []).length}  
- Basic Performance: ${(recommendations?.basic || []).length}
- Not Suitable: ${(recommendations?.not_suitable || []).length}

RECOMMENDED MODELS:
${suitableModels.slice(0, 5).map((model: any) => 
  `- ${model.name || 'Unknown'}: ${model.specs?.parameters || 'Unknown'} parameters (${model.compatibility?.performance_tier || 'Unknown'})`
).join('\n')}

NEXT STEPS:
1. Install Ollama from https://ollama.ai
2. Run: ollama run ${suitableModels[0]?.name?.toLowerCase().replace(/[^a-z0-9]/g, '') || 'llama3.2:3b'}
3. Start chatting with your local LLM!

For detailed installation instructions and optimization tips, download the full report.
      `.trim();

      await navigator.clipboard.writeText(summary);
      
      updateState({
        generatedReports: { ...state.generatedReports, summary: true }
      });
      
      toast.success('Report summary copied to clipboard!');
      onSuccess?.('summary');

    } catch (error) {
      console.error('Failed to copy summary:', error);
      
      const errorMessage = error instanceof ReportGenerationError
        ? error.message
        : 'Failed to copy summary to clipboard';
      
      updateState({ error: errorMessage });
      toast.error(errorMessage);
      onError?.(error as Error, 'summary');
    }
  }, [systemInfo, recommendations, validateData, state.generatedReports, updateState, onSuccess, onError]);

  const cancelGeneration = useCallback(() => {
    if (abortController.current) {
      abortController.current.abort();
      updateState({ isGenerating: false, error: 'Generation cancelled by user' });
      toast.error('Report generation cancelled');
    }
  }, [updateState]);

  const resetState = useCallback(() => {
    setState({
      isGenerating: false,
      progress: 0,
      stage: 'Ready',
      error: null,
      generatedReports: {}
    });
    progressTracker.current.reset();
  }, []);

  const getDebugInfo = useCallback(() => {
    return ReportUtils.generateDebugInfo();
  }, []);

  const estimateFileSize = useCallback(() => {
    try {
      return ReportUtils.estimateReportSize({ systemInfo, recommendations });
    } catch (error) {
      return { htmlSize: 'Unknown', pdfSize: 'Unknown' };
    }
  }, [systemInfo, recommendations]);

  const checkCompatibility = useCallback(() => {
    return ReportUtils.checkBrowserCompatibility();
  }, []);

  return {
    // State
    ...state,
    
    // Actions
    generateHTMLReport,
    generatePDFReport,
    copyReportSummary,
    cancelGeneration,
    resetState,
    
    // Utilities
    getDebugInfo,
    estimateFileSize,
    checkCompatibility,
    
    // Computed properties
    canGenerate: !state.isGenerating && Boolean(systemInfo && recommendations && recommender),
    hasAnyReports: Object.values(state.generatedReports).some(Boolean)
  };
};
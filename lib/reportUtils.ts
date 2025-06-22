// reportUtils.ts - Utility functions for report generation

export interface ReportGenerationOptions {
  includeSystemInfo: boolean;
  includeRecommendations: boolean;
  includeInstallationGuide: boolean;
  includeOptimizationTips: boolean;
  format: 'html' | 'pdf';
  quality: 'standard' | 'high';
}

export interface ReportValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

export class ReportUtils {
  /**
   * Validates report data before generation
   */
  static validateReportData(data: {
    systemInfo?: any;
    recommendations?: any;
    recommender?: any;
  }): ReportValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Validate system info
    if (!data.systemInfo) {
      errors.push('System information is missing');
    } else {
      if (!data.systemInfo.os) warnings.push('Operating system information is missing');
      if (!data.systemInfo.processor) warnings.push('Processor information is missing');
      if (!data.systemInfo.totalRamGB) warnings.push('RAM information is missing');
      if (!data.systemInfo.cpuCores) warnings.push('CPU core count is missing');
    }

    // Validate recommendations
    if (!data.recommendations) {
      errors.push('Model recommendations are missing');
    } else {
      const totalModels = [
        ...(data.recommendations.excellent || []),
        ...(data.recommendations.good || []),
        ...(data.recommendations.basic || []),
        ...(data.recommendations.not_suitable || [])
      ].length;
      
      if (totalModels === 0) {
        warnings.push('No model recommendations found');
      }
    }

    // Validate recommender
    if (!data.recommender) {
      errors.push('Recommender service is missing');
    } else {
      if (typeof data.recommender.getInstallationPlatforms !== 'function') {
        warnings.push('Installation platforms method is not available');
      }
      if (typeof data.recommender.getOptimizationTips !== 'function') {
        warnings.push('Optimization tips method is not available');
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Sanitizes filename for download
   */
  static sanitizeFilename(filename: string): string {
    return filename
      .replace(/[^a-z0-9.-]/gi, '_')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '');
  }

  /**
   * Generates a timestamp-based filename
   */
  static generateFilename(prefix: string, extension: string): string {
    const timestamp = new Date().toISOString().split('T')[0];
    const sanitizedPrefix = this.sanitizeFilename(prefix);
    return `${sanitizedPrefix}-${timestamp}.${extension}`;
  }

  /**
   * Checks browser compatibility for report generation
   */
  static checkBrowserCompatibility(): {
    canGenerateHTML: boolean;
    canGeneratePDF: boolean;
    canCopyToClipboard: boolean;
    issues: string[];
  } {
    const issues: string[] = [];
    
    // Check HTML blob support
    const canGenerateHTML = typeof Blob !== 'undefined' && typeof URL !== 'undefined';
    if (!canGenerateHTML) {
      issues.push('HTML report generation not supported: Blob API unavailable');
    }

    // Check PDF generation support
    const canGeneratePDF = typeof HTMLCanvasElement !== 'undefined' && 
                          typeof document !== 'undefined' &&
                          'createElement' in document;
    if (!canGeneratePDF) {
      issues.push('PDF report generation not supported: Canvas API unavailable');
    }

    // Check clipboard support
    const canCopyToClipboard = typeof navigator !== 'undefined' && 
                              'clipboard' in navigator &&
                              typeof navigator.clipboard.writeText === 'function';
    if (!canCopyToClipboard) {
      issues.push('Clipboard functionality limited: Modern Clipboard API unavailable');
    }

    return {
      canGenerateHTML,
      canGeneratePDF,
      canCopyToClipboard,
      issues
    };
  }

  /**
   * Waits for fonts to load before PDF generation
   */
  static async waitForFonts(timeout: number = 5000): Promise<boolean> {
    if (!document.fonts) {
      // Fallback wait for older browsers
      await new Promise(resolve => setTimeout(resolve, 2000));
      return false;
    }

    try {
      await Promise.race([
        document.fonts.ready,
        new Promise(resolve => setTimeout(resolve, timeout))
      ]);
      
      // Additional wait to ensure rendering
      await new Promise(resolve => setTimeout(resolve, 500));
      return true;
    } catch (error) {
      console.warn('Font loading check failed:', error);
      await new Promise(resolve => setTimeout(resolve, 2000));
      return false;
    }
  }

  /**
   * Creates a safe iframe for PDF rendering
   */
  static createRenderingIframe(): HTMLIFrameElement {
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
      z-index: -1;
    `;
    
    // Add sandbox attributes for security
    iframe.setAttribute('sandbox', 'allow-same-origin');
    
    document.body.appendChild(iframe);
    return iframe;
  }

  /**
   * Cleans up resources after report generation
   */
  static cleanup(resources: {
    iframe?: HTMLIFrameElement;
    blob?: Blob;
    url?: string;
    link?: HTMLAnchorElement;
  }): void {
    try {
      if (resources.iframe && resources.iframe.parentNode) {
        resources.iframe.parentNode.removeChild(resources.iframe);
      }
      
      if (resources.url) {
        URL.revokeObjectURL(resources.url);
      }
      
      if (resources.link && resources.link.parentNode) {
        resources.link.parentNode.removeChild(resources.link);
      }
    } catch (error) {
      console.warn('Cleanup error:', error);
    }
  }

  /**
   * Estimates report file size
   */
  static estimateReportSize(data: {
    systemInfo: any;
    recommendations: any;
  }): { htmlSize: string; pdfSize: string } {
    const baseHTMLSize = 50; // KB
    const baseStylesSize = 30; // KB
    const baseJSSize = 10; // KB
    
    const modelCount = [
      ...(data.recommendations?.excellent || []),
      ...(data.recommendations?.good || []),
      ...(data.recommendations?.basic || [])
    ].length;
    
    const estimatedContentSize = modelCount * 2; // KB per model
    const totalHTMLSize = baseHTMLSize + baseStylesSize + baseJSSize + estimatedContentSize;
    
    // PDF is typically 3-5x larger than HTML
    const totalPDFSize = totalHTMLSize * 4;
    
    const formatSize = (sizeKB: number): string => {
      if (sizeKB < 1024) {
        return `${Math.round(sizeKB)} KB`;
      } else {
        return `${(sizeKB / 1024).toFixed(1)} MB`;
      }
    };
    
    return {
      htmlSize: formatSize(totalHTMLSize),
      pdfSize: formatSize(totalPDFSize)
    };
  }

  /**
   * Generates a debug report for troubleshooting
   */
  static generateDebugInfo(): string {
    const compatibility = this.checkBrowserCompatibility();
    
    return `
LLM Compatibility Checker - Debug Information
Generated: ${new Date().toISOString()}

BROWSER INFORMATION:
- User Agent: ${navigator.userAgent}
- Platform: ${navigator.platform}
- Language: ${navigator.language}
- Cookie Enabled: ${navigator.cookieEnabled}
- Online: ${navigator.onLine}

FEATURE SUPPORT:
- HTML Generation: ${compatibility.canGenerateHTML ? 'Supported' : 'Not Supported'}
- PDF Generation: ${compatibility.canGeneratePDF ? 'Supported' : 'Not Supported'}
- Clipboard API: ${compatibility.canCopyToClipboard ? 'Supported' : 'Limited'}

DOCUMENT INFORMATION:
- Ready State: ${document.readyState}
- Document Mode: ${(document as any).documentMode || 'Standard'}
- Character Set: ${document.characterSet}

COMPATIBILITY ISSUES:
${compatibility.issues.length > 0 ? compatibility.issues.map(issue => `- ${issue}`).join('\n') : '- None detected'}

SCREEN INFORMATION:
- Screen Size: ${screen.width}x${screen.height}
- Available Size: ${screen.availWidth}x${screen.availHeight}
- Color Depth: ${screen.colorDepth}

MEMORY (if available):
- Used JS Heap: ${(performance as any).memory?.usedJSHeapSize || 'Not available'}
- Total JS Heap: ${(performance as any).memory?.totalJSHeapSize || 'Not available'}
- Heap Limit: ${(performance as any).memory?.jsHeapSizeLimit || 'Not available'}
    `.trim();
  }
}

/**
 * Custom error classes for report generation
 */
export class ReportGenerationError extends Error {
  constructor(
    message: string,
    public readonly phase: 'validation' | 'generation' | 'download' | 'pdf-rendering',
    public readonly originalError?: Error
  ) {
    super(message);
    this.name = 'ReportGenerationError';
  }
}

export class PDFGenerationError extends ReportGenerationError {
  constructor(message: string, originalError?: Error) {
    super(message, 'pdf-rendering', originalError);
    this.name = 'PDFGenerationError';
  }
}

/**
 * Progress tracking for report generation
 */
export class ReportGenerationProgress {
  private listeners: Array<(progress: number, stage: string) => void> = [];
  private currentProgress = 0;
  private currentStage = 'Starting';

  addListener(callback: (progress: number, stage: string) => void): void {
    this.listeners.push(callback);
  }

  removeListener(callback: (progress: number, stage: string) => void): void {
    const index = this.listeners.indexOf(callback);
    if (index > -1) {
      this.listeners.splice(index, 1);
    }
  }

  updateProgress(progress: number, stage: string): void {
    this.currentProgress = Math.max(0, Math.min(100, progress));
    this.currentStage = stage;
    this.listeners.forEach(listener => {
      try {
        listener(this.currentProgress, this.currentStage);
      } catch (error) {
        console.error('Progress listener error:', error);
      }
    });
  }

  getCurrentProgress(): { progress: number; stage: string } {
    return { progress: this.currentProgress, stage: this.currentStage };
  }

  reset(): void {
    this.currentProgress = 0;
    this.currentStage = 'Starting';
  }
}

/**
 * Configuration for report generation
 */
export const DEFAULT_REPORT_CONFIG: ReportGenerationOptions = {
  includeSystemInfo: true,
  includeRecommendations: true,
  includeInstallationGuide: true,
  includeOptimizationTips: true,
  format: 'html',
  quality: 'standard'
};

/**
 * Constants for report generation
 */
export const REPORT_CONSTANTS = {
  // File size limits
  MAX_HTML_SIZE_MB: 10,
  MAX_PDF_SIZE_MB: 50,
  
  // Timeouts
  FONT_LOAD_TIMEOUT_MS: 5000,
  PDF_GENERATION_TIMEOUT_MS: 30000,
  
  // PDF settings
  PDF_DPI: 96,
  PDF_SCALE: 2,
  PDF_QUALITY: 0.95,
  
  // Canvas settings
  CANVAS_MAX_WIDTH: 794,
  CANVAS_MAX_HEIGHT: 10000,
  
  // File naming
  DEFAULT_HTML_FILENAME: 'llm-compatibility-report',
  DEFAULT_PDF_FILENAME: 'llm-compatibility-report',
  
  // MIME types
  HTML_MIME_TYPE: 'text/html;charset=utf-8',
  PDF_MIME_TYPE: 'application/pdf'
} as const;
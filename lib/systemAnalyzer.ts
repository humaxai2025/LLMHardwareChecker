'use client';

// Client-side system analysis utilities
export interface SystemInfo {
  os: string;
  architecture: string;
  processor: string;
  cpuCores: number;
  totalRamGB: number;
  availableRamGB: number;
  totalStorageGB: number;
  freeStorageGB: number;
  gpus: GPU[];
  userAgent: string;
  screenResolution: string;
  colorDepth: number;
  language: string;
  timezone: string;
}

export interface GPU {
  name: string;
  vramGB: number | string;
  type: string;
  renderer?: string;
}

export class SystemAnalyzer {
  private systemInfo: Partial<SystemInfo> = {};

  async analyzeSystem(): Promise<SystemInfo> {
    try {
      // Basic browser/OS detection
      this.detectPlatform();
      
      // Hardware detection
      await this.detectHardware();
      
      // GPU detection
      await this.detectGPUs();
      
      // Storage estimation
      this.estimateStorage();
      
      return this.systemInfo as SystemInfo;
    } catch (error) {
      console.error('System analysis failed:', error);
      return this.getDefaultSystemInfo();
    }
  }

  private detectPlatform(): void {
    const userAgent = navigator.userAgent;
    this.systemInfo.userAgent = userAgent;
    
    // OS Detection
    if (userAgent.includes('Windows')) {
      this.systemInfo.os = 'Windows';
    } else if (userAgent.includes('Mac')) {
      this.systemInfo.os = 'macOS';
    } else if (userAgent.includes('Linux')) {
      this.systemInfo.os = 'Linux';
    } else if (userAgent.includes('Android')) {
      this.systemInfo.os = 'Android';
    } else if (userAgent.includes('iOS')) {
      this.systemInfo.os = 'iOS';
    } else {
      this.systemInfo.os = 'Unknown';
    }

    // Architecture detection (limited in browsers)
    this.systemInfo.architecture = navigator.platform || 'Unknown';
    
    // Processor detection (very limited)
    this.systemInfo.processor = this.estimateProcessor(userAgent);
    
    // Screen and locale info
    this.systemInfo.screenResolution = `${screen.width}x${screen.height}`;
    this.systemInfo.colorDepth = screen.colorDepth;
    this.systemInfo.language = navigator.language;
    this.systemInfo.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  }

  private estimateProcessor(userAgent: string): string {
    // Limited processor detection from user agent
    if (userAgent.includes('Intel')) return 'Intel Processor';
    if (userAgent.includes('AMD')) return 'AMD Processor';
    if (userAgent.includes('ARM')) return 'ARM Processor';
    if (userAgent.includes('Apple')) return 'Apple Silicon';
    return 'Unknown Processor';
  }

  private async detectHardware(): Promise<void> {
    // CPU cores detection
    this.systemInfo.cpuCores = navigator.hardwareConcurrency || 4;
    
    // Memory detection (very limited in browsers)
    const memory = (navigator as any).deviceMemory;
    if (memory) {
      this.systemInfo.totalRamGB = memory;
      this.systemInfo.availableRamGB = memory * 0.8; // Estimate 80% available
    } else {
      // Fallback estimation based on other factors
      this.systemInfo.totalRamGB = this.estimateRAM();
      this.systemInfo.availableRamGB = this.systemInfo.totalRamGB * 0.7;
    }
  }

  private estimateRAM(): number {
    // Estimate RAM based on various factors
    const cores = this.systemInfo.cpuCores || 4;
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    if (isMobile) {
      return cores <= 4 ? 4 : 8; // Mobile devices typically have 4-8GB
    }
    
    // Desktop estimation based on CPU cores
    if (cores <= 2) return 4;
    if (cores <= 4) return 8;
    if (cores <= 8) return 16;
    return 32; // High-end systems
  }

  private async detectGPUs(): Promise<void> {
    const gpus: GPU[] = [];
    
    try {
      // WebGL-based GPU detection
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') as WebGLRenderingContext | null;
      
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
          const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
          
          gpus.push({
            name: this.parseGPUName(renderer),
            vramGB: this.estimateVRAM(renderer),
            type: this.detectGPUType(renderer, vendor),
            renderer: renderer
          });
        }
      }
      
      // Try experimental GPU API if available
      if ('gpu' in navigator) {
        try {
          const adapter = await (navigator as any).gpu.requestAdapter();
          if (adapter) {
            gpus.push({
              name: 'WebGPU Compatible GPU',
              vramGB: 'Unknown',
              type: 'Modern GPU'
            });
          }
        } catch (e) {
          // WebGPU not available or failed
        }
      }
      
      this.systemInfo.gpus = gpus.length > 0 ? gpus : [];
      
    } catch (error) {
      console.warn('GPU detection failed:', error);
      this.systemInfo.gpus = [];
    }
  }

  private parseGPUName(renderer: string): string {
    // Clean up the renderer string to get a more readable GPU name
    const cleanName = renderer
      .replace(/OpenGL Engine/i, '')
      .replace(/Direct3D/i, '')
      .replace(/\([^)]*\)/g, '') // Remove parentheses
      .replace(/\s+/g, ' ')
      .trim();
    
    return cleanName || 'Unknown GPU';
  }

  private estimateVRAM(renderer: string): number | string {
    const rendererLower = renderer.toLowerCase();
    
    // NVIDIA cards
    if (rendererLower.includes('rtx 4090')) return 24;
    if (rendererLower.includes('rtx 4080')) return 16;
    if (rendererLower.includes('rtx 4070')) return 12;
    if (rendererLower.includes('rtx 4060')) return 8;
    if (rendererLower.includes('rtx 3090')) return 24;
    if (rendererLower.includes('rtx 3080')) return 10;
    if (rendererLower.includes('rtx 3070')) return 8;
    if (rendererLower.includes('rtx 3060')) return 12;
    if (rendererLower.includes('gtx 1660')) return 6;
    if (rendererLower.includes('gtx 1080')) return 8;
    if (rendererLower.includes('gtx 1070')) return 8;
    if (rendererLower.includes('gtx 1060')) return 6;
    
    // AMD cards
    if (rendererLower.includes('rx 7900')) return 20;
    if (rendererLower.includes('rx 6900')) return 16;
    if (rendererLower.includes('rx 6800')) return 16;
    if (rendererLower.includes('rx 6700')) return 12;
    if (rendererLower.includes('rx 580')) return 8;
    if (rendererLower.includes('rx 570')) return 4;
    
    // Intel cards
    if (rendererLower.includes('arc a770')) return 16;
    if (rendererLower.includes('arc a750')) return 8;
    
    // Apple Silicon
    if (rendererLower.includes('apple')) return 'Unified Memory';
    
    // Generic estimation
    if (rendererLower.includes('integrated') || rendererLower.includes('intel')) return 'Shared';
    
    return 'Unknown';
  }

  private detectGPUType(renderer: string, vendor: string): string {
    const rendererLower = renderer.toLowerCase();
    const vendorLower = vendor.toLowerCase();
    
    if (vendorLower.includes('nvidia') || rendererLower.includes('nvidia') || rendererLower.includes('geforce') || rendererLower.includes('rtx') || rendererLower.includes('gtx')) {
      return 'NVIDIA';
    }
    if (vendorLower.includes('amd') || rendererLower.includes('amd') || rendererLower.includes('radeon') || rendererLower.includes('rx ')) {
      return 'AMD';
    }
    if (vendorLower.includes('intel') || rendererLower.includes('intel')) {
      return 'Intel';
    }
    if (vendorLower.includes('apple') || rendererLower.includes('apple')) {
      return 'Apple';
    }
    return 'Unknown';
  }

  private estimateStorage(): void {
    // Browser storage estimation (very limited)
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      navigator.storage.estimate().then(estimate => {
        const quota = estimate.quota || 0;
        const usage = estimate.usage || 0;
        
        // Convert to GB and provide rough estimates
        this.systemInfo.totalStorageGB = Math.round((quota / (1024 ** 3)) * 10) / 10;
        this.systemInfo.freeStorageGB = Math.round(((quota - usage) / (1024 ** 3)) * 10) / 10;
      }).catch(() => {
        this.setDefaultStorage();
      });
    } else {
      this.setDefaultStorage();
    }
  }

  private setDefaultStorage(): void {
    // Default storage estimates based on system type
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    if (isMobile) {
      this.systemInfo.totalStorageGB = 64;
      this.systemInfo.freeStorageGB = 32;
    } else {
      this.systemInfo.totalStorageGB = 500;
      this.systemInfo.freeStorageGB = 250;
    }
  }

  private getDefaultSystemInfo(): SystemInfo {
    return {
      os: 'Unknown',
      architecture: 'Unknown',
      processor: 'Unknown Processor',
      cpuCores: 4,
      totalRamGB: 8,
      availableRamGB: 6,
      totalStorageGB: 500,
      freeStorageGB: 250,
      gpus: [],
      userAgent: navigator.userAgent,
      screenResolution: `${screen.width}x${screen.height}`,
      colorDepth: screen.colorDepth,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    };
  }
}

export const analyzeSystem = async (): Promise<SystemInfo> => {
  const analyzer = new SystemAnalyzer();
  return await analyzer.analyzeSystem();
};
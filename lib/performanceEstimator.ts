import { SystemInfo } from './systemAnalyzer';
import { LLMModel } from './llmDatabase';

export interface PerformanceEstimate {
  tokensPerSecond: number;
  speedRating: 'Very Fast' | 'Fast' | 'Moderate' | 'Slow' | 'Very Slow';
  loadTimeSeconds: number;
  memoryUsagePercent: number;
  gpuUtilizationPercent: number;
  powerConsumptionWatts: number;
  notes: string[];
}

export class PerformanceEstimator {
  private systemInfo: SystemInfo;

  constructor(systemInfo: SystemInfo) {
    this.systemInfo = systemInfo;
  }

  /**
   * Estimate performance for a given model
   */
  estimatePerformance(model: LLMModel, quantization: string = 'Q4_K_M'): PerformanceEstimate {
    const paramCount = this.parseParameterCount(model.parameters);
    const hasGPU = this.systemInfo.gpus && this.systemInfo.gpus.length > 0;
    const gpuVRAM = this.getGPUVRAM();
    const canRunOnGPU = hasGPU && gpuVRAM >= model.min_vram_gb;

    // Base calculations
    const tokensPerSecond = this.calculateTokensPerSecond(paramCount, quantization, canRunOnGPU);
    const loadTimeSeconds = this.calculateLoadTime(paramCount, quantization, canRunOnGPU);
    const memoryUsage = this.calculateMemoryUsage(paramCount, quantization, canRunOnGPU);
    const gpuUtilization = canRunOnGPU ? this.calculateGPUUtilization(paramCount, gpuVRAM) : 0;
    const powerConsumption = this.calculatePowerConsumption(paramCount, canRunOnGPU);

    // Generate notes
    const notes = this.generateNotes(model, quantization, canRunOnGPU, tokensPerSecond, memoryUsage);

    return {
      tokensPerSecond: Math.round(tokensPerSecond * 10) / 10,
      speedRating: this.getSpeedRating(tokensPerSecond),
      loadTimeSeconds: Math.round(loadTimeSeconds),
      memoryUsagePercent: Math.round(memoryUsage),
      gpuUtilizationPercent: Math.round(gpuUtilization),
      powerConsumptionWatts: Math.round(powerConsumption),
      notes,
    };
  }

  private parseParameterCount(params: string): number {
    const match = params.match(/(\d+\.?\d*)/);
    if (!match) return 7; // Default

    const num = parseFloat(match[1]);
    if (params.includes('B')) return num;
    if (params.includes('M')) return num / 1000;
    return num;
  }

  private getGPUVRAM(): number {
    if (!this.systemInfo.gpus || this.systemInfo.gpus.length === 0) return 0;

    const vram = this.systemInfo.gpus[0].vramGB;
    if (typeof vram === 'string') {
      const match = vram.match(/(\d+)/);
      return match ? parseInt(match[1]) : 0;
    }
    return vram;
  }

  private calculateTokensPerSecond(paramCount: number, quantization: string, onGPU: boolean): number {
    // Base speeds (tokens/second)
    let baseSpeed: number;

    if (onGPU) {
      // GPU inference speeds based on parameter count
      const gpuVRAM = this.getGPUVRAM();
      const cpuCores = this.systemInfo.cpuCores;

      if (paramCount <= 3) {
        baseSpeed = 120; // Small models are very fast on GPU
      } else if (paramCount <= 7) {
        baseSpeed = 80;
      } else if (paramCount <= 13) {
        baseSpeed = 50;
      } else if (paramCount <= 30) {
        baseSpeed = 25;
      } else if (paramCount <= 70) {
        baseSpeed = 12;
      } else {
        baseSpeed = 5;
      }

      // Adjust for VRAM headroom
      const vramUsageRatio = (paramCount * 2) / gpuVRAM;
      if (vramUsageRatio > 0.9) baseSpeed *= 0.5; // Tight VRAM = slower
      else if (vramUsageRatio < 0.5) baseSpeed *= 1.2; // Plenty of VRAM = faster

      // Adjust for CPU cores (helps with batching)
      if (cpuCores >= 16) baseSpeed *= 1.15;
      else if (cpuCores <= 4) baseSpeed *= 0.85;

    } else {
      // CPU-only inference speeds
      const cpuCores = this.systemInfo.cpuCores;

      if (paramCount <= 3) {
        baseSpeed = 15;
      } else if (paramCount <= 7) {
        baseSpeed = 8;
      } else if (paramCount <= 13) {
        baseSpeed = 4;
      } else if (paramCount <= 30) {
        baseSpeed = 2;
      } else {
        baseSpeed = 0.5;
      }

      // Adjust for CPU cores
      if (cpuCores >= 16) baseSpeed *= 1.5;
      else if (cpuCores >= 8) baseSpeed *= 1.2;
      else if (cpuCores <= 4) baseSpeed *= 0.7;

      // Adjust for RAM
      const ramPerBillion = this.systemInfo.totalRamGB / paramCount;
      if (ramPerBillion < 2) baseSpeed *= 0.6; // Tight RAM = slower
      else if (ramPerBillion > 8) baseSpeed *= 1.1; // Plenty RAM = faster
    }

    // Quantization impact
    const quantMultiplier = this.getQuantizationMultiplier(quantization);
    baseSpeed *= quantMultiplier;

    return baseSpeed;
  }

  private getQuantizationMultiplier(quantization: string): number {
    // Higher quantization = faster inference but lower quality
    if (quantization.includes('Q2')) return 1.4;
    if (quantization.includes('Q3')) return 1.25;
    if (quantization.includes('Q4')) return 1.15;
    if (quantization.includes('Q5')) return 1.05;
    if (quantization.includes('Q8')) return 0.95;
    if (quantization.includes('FP16') || quantization.includes('F16')) return 0.85;
    return 1.0;
  }

  private calculateLoadTime(paramCount: number, quantization: string, onGPU: boolean): number {
    // Base load time in seconds
    let loadTime = paramCount * 0.8; // ~0.8 seconds per billion params

    if (onGPU) {
      loadTime *= 0.7; // GPU loading is faster
    }

    // Quantization reduces model size = faster loading
    if (quantization.includes('Q2')) loadTime *= 0.4;
    else if (quantization.includes('Q3')) loadTime *= 0.5;
    else if (quantization.includes('Q4')) loadTime *= 0.6;
    else if (quantization.includes('Q5')) loadTime *= 0.75;
    else if (quantization.includes('Q8')) loadTime *= 0.9;

    return Math.max(1, loadTime); // Minimum 1 second
  }

  private calculateMemoryUsage(paramCount: number, quantization: string, onGPU: boolean): number {
    // Calculate how much of available memory will be used
    const modelSizeGB = this.estimateModelSize(paramCount, quantization);

    if (onGPU) {
      const gpuVRAM = this.getGPUVRAM();
      return (modelSizeGB / gpuVRAM) * 100;
    } else {
      return (modelSizeGB / this.systemInfo.totalRamGB) * 100;
    }
  }

  private estimateModelSize(paramCount: number, quantization: string): number {
    // Rough estimates of model size in GB
    let sizePerBillion = 2.0; // FP16 baseline

    if (quantization.includes('Q2')) sizePerBillion = 0.5;
    else if (quantization.includes('Q3')) sizePerBillion = 0.75;
    else if (quantization.includes('Q4')) sizePerBillion = 1.0;
    else if (quantization.includes('Q5')) sizePerBillion = 1.25;
    else if (quantization.includes('Q8')) sizePerBillion = 1.5;

    return paramCount * sizePerBillion;
  }

  private calculateGPUUtilization(paramCount: number, gpuVRAM: number): number {
    // Estimate GPU utilization percentage
    const vramUsed = this.estimateModelSize(paramCount, 'Q4_K_M');
    const utilizationRatio = vramUsed / gpuVRAM;

    // Models use 70-95% of GPU when they fit
    if (utilizationRatio <= 0.5) return 75;
    if (utilizationRatio <= 0.7) return 85;
    if (utilizationRatio <= 0.9) return 90;
    return 95;
  }

  private calculatePowerConsumption(paramCount: number, onGPU: boolean): number {
    if (onGPU) {
      // GPU power consumption
      if (paramCount <= 3) return 120;
      if (paramCount <= 7) return 180;
      if (paramCount <= 13) return 250;
      if (paramCount <= 30) return 300;
      if (paramCount <= 70) return 350;
      return 400;
    } else {
      // CPU power consumption
      if (paramCount <= 3) return 30;
      if (paramCount <= 7) return 50;
      if (paramCount <= 13) return 80;
      if (paramCount <= 30) return 120;
      return 150;
    }
  }

  private getSpeedRating(tokensPerSecond: number): 'Very Fast' | 'Fast' | 'Moderate' | 'Slow' | 'Very Slow' {
    if (tokensPerSecond >= 60) return 'Very Fast';
    if (tokensPerSecond >= 30) return 'Fast';
    if (tokensPerSecond >= 10) return 'Moderate';
    if (tokensPerSecond >= 3) return 'Slow';
    return 'Very Slow';
  }

  private generateNotes(
    model: LLMModel,
    quantization: string,
    canRunOnGPU: boolean,
    tokensPerSecond: number,
    memoryUsage: number
  ): string[] {
    const notes: string[] = [];

    // Performance notes
    if (tokensPerSecond >= 50) {
      notes.push('⚡ Excellent performance for real-time interactions');
    } else if (tokensPerSecond >= 20) {
      notes.push('✅ Good performance for most use cases');
    } else if (tokensPerSecond >= 5) {
      notes.push('⚠️ Usable but may feel slow for long conversations');
    } else {
      notes.push('🐌 Very slow - consider a smaller model or better hardware');
    }

    // Memory notes
    if (memoryUsage > 90) {
      notes.push('🔴 Critical memory usage - may cause crashes');
    } else if (memoryUsage > 75) {
      notes.push('⚠️ High memory usage - close other applications');
    } else if (memoryUsage < 50) {
      notes.push('💚 Comfortable memory headroom');
    }

    // GPU notes
    if (canRunOnGPU) {
      notes.push('🎮 Running on GPU for optimal performance');
    } else if (this.systemInfo.gpus && this.systemInfo.gpus.length > 0) {
      notes.push('💡 Consider smaller quantization to enable GPU acceleration');
    } else {
      notes.push('💻 CPU-only inference - consider adding a GPU for better performance');
    }

    // Quantization notes
    if (quantization.includes('Q2') || quantization.includes('Q3')) {
      notes.push('📉 Low quantization may affect output quality');
    } else if (quantization.includes('Q8') || quantization.includes('FP16')) {
      notes.push('📈 High quality quantization for best output');
    }

    return notes;
  }
}

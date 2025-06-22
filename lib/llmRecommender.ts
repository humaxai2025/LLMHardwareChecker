// LLM Recommendation Engine
import { SystemInfo } from './systemAnalyzer';
import { LLM_DATABASE, LLMModel, CompatibilityCheck, ModelRecommendation, Recommendations } from './llmDatabase';

export class LLMRecommender {
  private systemInfo: SystemInfo;

  constructor(systemInfo: SystemInfo) {
    this.systemInfo = systemInfo;
  }

  checkCompatibility(modelName: string, modelSpecs: LLMModel): CompatibilityCheck {
    const compatibility: CompatibilityCheck = {
      can_run_cpu: false,
      can_run_gpu: false,
      performance_tier: "Not Suitable",
      notes: [],
      recommended_quant: undefined,
    };

    // Check CPU compatibility
    if (this.systemInfo.totalRamGB >= modelSpecs.min_ram_gb) {
      compatibility.can_run_cpu = true;
      if (this.systemInfo.totalRamGB >= modelSpecs.recommended_ram_gb) {
        compatibility.performance_tier = "Good (CPU)";
      } else {
        compatibility.performance_tier = "Basic (CPU)";
        compatibility.notes.push("Consider quantized version for better performance");
      }
    }

    // Check GPU compatibility
    if (this.systemInfo.gpus && this.systemInfo.gpus.length > 0) {
      for (const gpu of this.systemInfo.gpus) {
        if (typeof gpu.vramGB === 'number') {
          if (gpu.vramGB >= modelSpecs.min_vram_gb) {
            compatibility.can_run_gpu = true;
            if (gpu.vramGB >= modelSpecs.recommended_vram_gb) {
              compatibility.performance_tier = "Excellent (GPU)";
              compatibility.recommended_quant = "Q8_0 or FP16";
            } else {
              compatibility.performance_tier = "Good (GPU)";
              compatibility.recommended_quant = "Q4_K_M";
            }
            break;
          }
        }
      }
    }

    // Add specific notes
    if (!compatibility.can_run_cpu && !compatibility.can_run_gpu) {
      compatibility.notes.push("Insufficient RAM/VRAM for this model");
    } else if (compatibility.can_run_cpu && !compatibility.can_run_gpu) {
      compatibility.notes.push("GPU acceleration not available, will run on CPU");
      compatibility.recommended_quant = "Q4_K_M or Q3_K_M";
    }

    // Special handling for mobile devices
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(this.systemInfo.userAgent);
    if (isMobile && modelSpecs.parameters !== "2B" && modelSpecs.parameters !== "3B") {
      compatibility.performance_tier = "Not Suitable";
      compatibility.notes.push("Model too large for mobile device");
    }

    return compatibility;
  }

  getRecommendations(): Recommendations {
    const recommendations: Recommendations = {
      excellent: [],
      good: [],
      basic: [],
      not_suitable: []
    };

    for (const [modelName, modelSpecs] of Object.entries(LLM_DATABASE)) {
      const compatibility = this.checkCompatibility(modelName, modelSpecs);

      const modelInfo: ModelRecommendation = {
        name: modelName,
        specs: modelSpecs,
        compatibility: compatibility,
      };

      if (compatibility.performance_tier === "Excellent (GPU)") {
        recommendations.excellent.push(modelInfo);
      } else if (compatibility.performance_tier.includes("Good")) {
        recommendations.good.push(modelInfo);
      } else if (compatibility.performance_tier === "Basic (CPU)") {
        recommendations.basic.push(modelInfo);
      } else {
        recommendations.not_suitable.push(modelInfo);
      }
    }

    // Sort recommendations by parameter count (smaller first for each category)
    const sortByParameters = (a: ModelRecommendation, b: ModelRecommendation) => {
      const aParams = parseFloat(a.specs.parameters.replace('B', ''));
      const bParams = parseFloat(b.specs.parameters.replace('B', ''));
      return aParams - bParams;
    };

    recommendations.excellent.sort(sortByParameters);
    recommendations.good.sort(sortByParameters);
    recommendations.basic.sort(sortByParameters);
    recommendations.not_suitable.sort(sortByParameters);

    return recommendations;
  }

  getSystemCapabilityLevel(): 'low' | 'medium' | 'high' | 'premium' {
    const ram = this.systemInfo.totalRamGB;
    const hasGPU = this.systemInfo.gpus && this.systemInfo.gpus.length > 0;
    
    let maxVRAM = 0;
    if (hasGPU) {
      for (const gpu of this.systemInfo.gpus) {
        if (typeof gpu.vramGB === 'number' && gpu.vramGB > maxVRAM) {
          maxVRAM = gpu.vramGB;
        }
      }
    }

    if (ram >= 32 && maxVRAM >= 16) return 'premium';
    if (ram >= 16 && (maxVRAM >= 8 || ram >= 24)) return 'high';
    if (ram >= 8 && (maxVRAM >= 4 || ram >= 12)) return 'medium';
    return 'low';
  }

  getOptimizationTips(): string[] {
    const tips: string[] = [];
    const ram = this.systemInfo.totalRamGB;
    const hasGPU = this.systemInfo.gpus && this.systemInfo.gpus.length > 0;
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(this.systemInfo.userAgent);

    if (isMobile) {
      tips.push("📱 Mobile detected: Stick to 2B-3B models for best performance");
      tips.push("🔋 Consider battery impact when running models locally");
      tips.push("☁️ Cloud-based LLMs often work better on mobile devices");
      return tips;
    }

    if (ram < 8) {
      tips.push("💾 Low RAM: Use Q3_K_M or Q4_K_M quantized models");
      tips.push("❌ Close other applications before running LLMs");
      tips.push("⚡ Consider using swap file for larger models");
    } else if (ram < 16) {
      tips.push("💾 Medium RAM: Q4_K_M quantization works well");
      tips.push("✅ Can run 7B models comfortably");
    } else {
      tips.push("💾 High RAM: Can use Q5_K_M or Q8_0 for better quality");
      tips.push("🚀 Multiple models can be loaded simultaneously");
    }

    if (hasGPU) {
      const maxVRAM = Math.max(...this.systemInfo.gpus
        .map(gpu => typeof gpu.vramGB === 'number' ? gpu.vramGB : 0));
      
      if (maxVRAM > 0) {
        if (maxVRAM < 6) {
          tips.push("🎮 GPU: Use Q4_K_M quantization for GPU inference");
          tips.push("💻 Consider CPU inference for larger models");
        } else if (maxVRAM < 12) {
          tips.push("🎮 GPU: Q4_K_M or Q5_K_M work well");
          tips.push("✅ 7B models will run smoothly");
        } else {
          tips.push("🎮 GPU: Can use Q8_0 or even FP16 for best quality");
          tips.push("🚀 13B+ models are possible");
        }
      }

      // Apple Silicon detection
      const hasAppleGPU = this.systemInfo.gpus.some(gpu => gpu.type === 'Apple');
      if (hasAppleGPU) {
        tips.push("🍎 Apple Silicon: Metal acceleration works automatically");
        tips.push("🧠 Unified memory is shared between CPU and GPU");
        tips.push("⚡ Ollama has excellent Apple Silicon support");
      }
    } else {
      tips.push("💻 CPU-only: Use llama.cpp for best CPU performance");
      tips.push("🔧 Enable all CPU cores for better performance");
      tips.push("⚙️ Q4_K_M quantization balances speed and quality");
    }

    if (this.systemInfo.freeStorageGB < 50) {
      tips.push("💾 Low storage: Start with smallest models (2B-3B)");
      tips.push("🗑️ Delete unused models regularly");
    }

    // OS-specific tips
    if (this.systemInfo.os === 'Windows') {
      tips.push("🪟 Windows: Defender may scan large model files");
      tips.push("💻 Use Windows Terminal for better experience");
    } else if (this.systemInfo.os === 'macOS') {
      tips.push("🍎 macOS: Metal provides excellent performance");
      tips.push("📁 Models stored in ~/.ollama/models/");
    } else if (this.systemInfo.os === 'Linux') {
      tips.push("🐧 Linux: Best platform for LLM development");
      tips.push("🔧 Easy to build tools from source");
    }

    return tips;
  }

  getInstallationPlatforms(): Array<{
    name: string;
    description: string;
    difficulty: 'Easy' | 'Medium' | 'Advanced';
    bestFor: string;
    installation: Record<string, string>;
  }> {
    const os = this.systemInfo.os;
    
    return [
      {
        name: "Ollama",
        description: "Easiest way to run LLMs locally",
        difficulty: "Easy",
        bestFor: "Beginners and quick setup",
        installation: {
          'Windows': 'Download installer from ollama.ai/download/windows',
          'macOS': 'Download from ollama.ai/download/mac or brew install ollama',
          'Linux': 'curl -fsSL https://ollama.ai/install.sh | sh',
        }
      },
      {
        name: "LM Studio",
        description: "User-friendly GUI for running LLMs",
        difficulty: "Easy",
        bestFor: "Users who prefer graphical interfaces",
        installation: {
          'All': 'Download from lmstudio.ai - available for Windows, macOS, and Linux'
        }
      },
      {
        name: "llama.cpp",
        description: "High-performance C++ implementation",
        difficulty: "Advanced",
        bestFor: "CPU inference and custom setups",
        installation: {
          'Windows': 'Download release from GitHub or build with Visual Studio',
          'macOS': 'brew install llama.cpp or build from source',
          'Linux': 'Build from source: git clone + make'
        }
      },
      {
        name: "HuggingFace Transformers",
        description: "Python library for model deployment",
        difficulty: "Medium",
        bestFor: "Python developers and custom applications",
        installation: {
          'All': 'pip install transformers torch'
        }
      }
    ];
  }
}
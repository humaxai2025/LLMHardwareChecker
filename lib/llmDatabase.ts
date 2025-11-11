// LLM Database and Recommendation Engine
export interface InstallMethod {
  command?: string;
  note: string;
  model_id?: string;
  source?: string;
  recommended_quant?: string;
  download_url?: string;
}

export interface LLMModel {
  parameters: string;
  min_ram_gb: number;
  recommended_ram_gb: number;
  min_vram_gb: number;
  recommended_vram_gb: number;
  cpu_only: boolean;
  description: string;
  domain?: string;
  install_methods: {
    ollama?: InstallMethod;
    huggingface?: InstallMethod;
    gguf?: InstallMethod;
    lm_studio?: InstallMethod;
     llamacpp?: InstallMethod;
  };
}

export interface CompatibilityCheck {
  can_run_cpu: boolean;
  can_run_gpu: boolean;
  performance_tier: string;
  notes: string[];
  recommended_quant?: string;
}

export interface ModelRecommendation {
  name: string;
  specs: LLMModel;
  compatibility: CompatibilityCheck;
}

export interface Recommendations {
  excellent: ModelRecommendation[];
  good: ModelRecommendation[];
  basic: ModelRecommendation[];
  not_suitable: ModelRecommendation[];
}

export const LLM_DATABASE: Record<string, LLMModel> = {
  // Tiny models (0.5-1.5B parameters) - 2025 NEW
  "DeepSeek-R1 1.5B": {
    parameters: "1.5B",
    min_ram_gb: 2,
    recommended_ram_gb: 4,
    min_vram_gb: 1,
    recommended_vram_gb: 2,
    cpu_only: true,
    description: "DeepSeek's reasoning model, distilled to ultra-compact size",
    install_methods: {
      ollama: {
        command: "ollama run deepseek-r1:1.5b",
        note: "Latest reasoning model in compact size",
      },
      huggingface: {
        model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        command: "No login required",
        note: "Distilled reasoning capabilities",
      },
    },
  },
  "Qwen3 0.6B": {
    parameters: "0.6B",
    min_ram_gb: 2,
    recommended_ram_gb: 4,
    min_vram_gb: 0.5,
    recommended_vram_gb: 1,
    cpu_only: true,
    description: "Alibaba's ultra-lightweight model for edge devices",
    install_methods: {
      ollama: {
        command: "ollama run qwen3:0.6b",
        note: "Perfect for low-resource devices",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-0.6B-Instruct",
        command: "No login required",
        note: "Runs on phones and tablets",
      },
    },
  },

  // Small models (2-4B parameters)
  "Gemma 3 2B": {
    parameters: "2B",
    min_ram_gb: 3,
    recommended_ram_gb: 6,
    min_vram_gb: 1.5,
    recommended_vram_gb: 3,
    cpu_only: true,
    description: "Google's latest compact model with vision capabilities",
    install_methods: {
      ollama: {
        command: "ollama run gemma3:2b",
        note: "Latest Gemma 3 with improved performance",
      },
      huggingface: {
        model_id: "google/gemma-3-2b-it",
        command: "Requires HF login",
        note: "Need to accept license",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/gemma-3-2b-it-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Excellent for low-resource systems",
      },
    },
  },
  "Llama 3.2 3B": {
    parameters: "3B",
    min_ram_gb: 4,
    recommended_ram_gb: 8,
    min_vram_gb: 2,
    recommended_vram_gb: 4,
    cpu_only: true,
    description: "Meta's small model with good performance",
    install_methods: {
      ollama: {
        command: "ollama run llama3.2:3b",
        note: "Easiest installation method",
      },
      huggingface: {
        model_id: "meta-llama/Llama-3.2-3B-Instruct",
        command: "Requires HF login for access",
        note: "Need to accept license on HuggingFace",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Good balance of size and quality",
      },
    },
  },
  "Qwen3 4B": {
    parameters: "4B",
    min_ram_gb: 4,
    recommended_ram_gb: 8,
    min_vram_gb: 2,
    recommended_vram_gb: 4,
    cpu_only: true,
    description: "Alibaba's efficient 4B model with strong multilingual support",
    install_methods: {
      ollama: {
        command: "ollama run qwen3:4b",
        note: "Great balance of size and capability",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-4B-Instruct",
        command: "No login required",
        note: "Excellent for multilingual tasks",
      },
    },
  },

  // Medium models (7-8B parameters)
  "DeepSeek-R1 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Advanced reasoning model with O3-level performance",
    install_methods: {
      ollama: {
        command: "ollama run deepseek-r1:7b",
        note: "Latest reasoning model from DeepSeek",
      },
      huggingface: {
        model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        command: "No login required",
        note: "Distilled from 671B flagship model",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Excellent reasoning capabilities",
      },
    },
  },
  "Qwen3 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Alibaba's latest 7B with strong multilingual and coding abilities",
    install_methods: {
      ollama: {
        command: "ollama run qwen3:7b",
        note: "Latest Qwen generation",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-7B-Instruct",
        command: "No login required",
        note: "Excellent for diverse tasks",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Qwen3-7B-Instruct-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Great general-purpose model",
      },
    },
  },
  "Llama 3.1 8B": {
    parameters: "8B",
    min_ram_gb: 8,
    recommended_ram_gb: 16,
    min_vram_gb: 5,
    recommended_vram_gb: 8,
    cpu_only: true,
    description: "Excellent balance of performance and efficiency",
    install_methods: {
      ollama: {
        command: "ollama run llama3.1:8b",
        note: "Proven and reliable choice",
      },
      huggingface: {
        model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        command: "Requires HF login",
        note: "Full precision model",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        recommended_quant: "Q4_K_M for 8GB VRAM, Q8_0 for 16GB+",
        note: "Choose quantization based on your VRAM",
      },
    },
  },
  "Gemma 3 9B": {
    parameters: "9B",
    min_ram_gb: 8,
    recommended_ram_gb: 16,
    min_vram_gb: 5,
    recommended_vram_gb: 8,
    cpu_only: true,
    description: "Google's latest 9B model with vision capabilities",
    install_methods: {
      ollama: {
        command: "ollama run gemma3:9b",
        note: "Latest Gemma 3 with multimodal support",
      },
      huggingface: {
        model_id: "google/gemma-3-9b-it",
        command: "Requires HF login",
        note: "Vision-enabled model",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/gemma-3-9b-it-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Strong multilingual performance",
      },
    },
  },
  "Mistral 7B v0.3": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "High-quality model with strong performance",
    install_methods: {
      ollama: {
        command: "ollama run mistral:7b-instruct-v0.3",
        note: "Latest Mistral 7B version",
      },
      huggingface: {
        model_id: "mistralai/Mistral-7B-Instruct-v0.3",
        command: "No login required",
        note: "Open license model",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Reliable general-purpose model",
      },
    },
  },
  "Qwen3-Coder 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Specialized for code generation with agentic capabilities",
    install_methods: {
      ollama: {
        command: "ollama run qwen3-coder:7b",
        note: "Latest Qwen coding model",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-Coder-7B-Instruct",
        command: "No login required",
        note: "State-of-the-art coding model",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Qwen3-Coder-7B-Instruct-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Optimized for coding tasks",
      },
    },
  },

  // Large models (13-27B parameters)
  "Phi-4 14B": {
    parameters: "14B",
    min_ram_gb: 12,
    recommended_ram_gb: 24,
    min_vram_gb: 8,
    recommended_vram_gb: 12,
    cpu_only: false,
    description: "Microsoft's state-of-the-art 14B model with excellent reasoning",
    install_methods: {
      ollama: {
        command: "ollama run phi4:14b",
        note: "Latest Phi model from Microsoft",
      },
      huggingface: {
        model_id: "microsoft/phi-4",
        command: "No login required",
        note: "Excellent single-GPU performance",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/phi-4-GGUF",
        recommended_quant: "Q4_K_M for 8GB VRAM, Q5_K_M for 12GB+",
        note: "Great performance in compact size",
      },
    },
  },
  "DeepSeek-R1 14B": {
    parameters: "14B",
    min_ram_gb: 12,
    recommended_ram_gb: 24,
    min_vram_gb: 8,
    recommended_vram_gb: 12,
    cpu_only: false,
    description: "Reasoning-focused model with strong analytical capabilities",
    install_methods: {
      ollama: {
        command: "ollama run deepseek-r1:14b",
        note: "Advanced reasoning in mid-size model",
      },
      huggingface: {
        model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        command: "No login required",
        note: "Distilled reasoning model",
      },
    },
  },
  "Qwen3 14B": {
    parameters: "14B",
    min_ram_gb: 12,
    recommended_ram_gb: 24,
    min_vram_gb: 8,
    recommended_vram_gb: 12,
    cpu_only: false,
    description: "Alibaba's powerful 14B model with excellent multilingual support",
    install_methods: {
      ollama: {
        command: "ollama run qwen3:14b",
        note: "Strong general-purpose performance",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-14B-Instruct",
        command: "No login required",
        note: "Excellent for complex tasks",
      },
    },
  },
  "Gemma 3 27B": {
    parameters: "27B",
    min_ram_gb: 20,
    recommended_ram_gb: 32,
    min_vram_gb: 16,
    recommended_vram_gb: 24,
    cpu_only: false,
    description: "Google's flagship Gemma 3 with top performance and vision",
    install_methods: {
      ollama: {
        command: "ollama run gemma3:27b",
        note: "Most capable single-GPU Gemma model",
      },
      huggingface: {
        model_id: "google/gemma-3-27b-it",
        command: "Requires HF login",
        note: "State-of-the-art multimodal capabilities",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/gemma-3-27b-it-GGUF",
        recommended_quant: "Q3_K_M for 16GB VRAM, Q4_K_M for 24GB+",
        note: "Top-tier performance",
      },
    },
  },

  // Extra large models (30B-70B+ parameters)
  "DeepSeek-R1 32B": {
    parameters: "32B",
    min_ram_gb: 24,
    recommended_ram_gb: 48,
    min_vram_gb: 20,
    recommended_vram_gb: 32,
    cpu_only: false,
    description: "Powerful reasoning model with near-flagship performance",
    install_methods: {
      ollama: {
        command: "ollama run deepseek-r1:32b",
        note: "Advanced reasoning for high-end systems",
      },
      huggingface: {
        model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        command: "No login required",
        note: "Distilled from 671B model",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        recommended_quant: "Q3_K_M for 16GB VRAM, Q4_K_M for 24GB+",
        note: "Excellent reasoning capabilities",
      },
    },
  },
  "Qwen3-Coder 32B": {
    parameters: "32B",
    min_ram_gb: 24,
    recommended_ram_gb: 48,
    min_vram_gb: 20,
    recommended_vram_gb: 32,
    cpu_only: false,
    description: "Advanced coding model with agentic capabilities",
    install_methods: {
      ollama: {
        command: "ollama run qwen3-coder:32b",
        note: "State-of-the-art coding assistant",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-Coder-32B-Instruct",
        command: "No login required",
        note: "Professional-grade code generation",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Qwen3-Coder-32B-Instruct-GGUF",
        recommended_quant: "Q3_K_M for 16GB VRAM, Q4_K_M for 24GB+",
        note: "Best coding model for complex tasks",
      },
    },
  },
  "Llama 3.3 70B": {
    parameters: "70B",
    min_ram_gb: 48,
    recommended_ram_gb: 80,
    min_vram_gb: 40,
    recommended_vram_gb: 80,
    cpu_only: false,
    description: "Latest flagship Llama with performance matching 405B",
    install_methods: {
      ollama: {
        command: "ollama run llama3.3:70b",
        note: "State-of-the-art Llama model",
      },
      huggingface: {
        model_id: "meta-llama/Llama-3.3-70B-Instruct",
        command: "Requires HF login",
        note: "Requires license acceptance",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF",
        recommended_quant: "Q2_K for 24GB VRAM, Q4_K_M for 48GB+",
        note: "Top-tier general-purpose model",
      },
    },
  },
  "Qwen3 70B": {
    parameters: "70B",
    min_ram_gb: 48,
    recommended_ram_gb: 80,
    min_vram_gb: 40,
    recommended_vram_gb: 80,
    cpu_only: false,
    description: "Alibaba's flagship model with exceptional multilingual capabilities",
    install_methods: {
      ollama: {
        command: "ollama run qwen3:70b",
        note: "Powerful general-purpose model",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-70B-Instruct",
        command: "No login required",
        note: "Excellent for complex reasoning",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Qwen3-70B-Instruct-GGUF",
        recommended_quant: "Q2_K for 24GB VRAM, Q4_K_M for 48GB+",
        note: "Consider cloud deployment",
      },
    },
  },
  "DeepSeek-R1 671B": {
    parameters: "671B",
    min_ram_gb: 400,
    recommended_ram_gb: 500,
    min_vram_gb: 320,
    recommended_vram_gb: 400,
    cpu_only: false,
    description: "Flagship reasoning model with O3-level performance (MoE: 37B active)",
    install_methods: {
      ollama: {
        command: "ollama run deepseek-r1:671b",
        note: "Requires enterprise/cloud hardware",
      },
      huggingface: {
        model_id: "deepseek-ai/DeepSeek-R1",
        command: "No login required",
        note: "Mixture-of-experts architecture",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/DeepSeek-R1-GGUF",
        recommended_quant: "Q2_K for 80GB+ VRAM, Q4_K_M for 160GB+",
        note: "Cloud deployment recommended",
      },
    },
  },

  // SPECIALIZED MODELS
  "Qwen3-Math 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    domain: "Mathematics",
    description: "Specialized mathematical reasoning model from Qwen3 series",
    install_methods: {
      ollama: {
        command: "ollama run qwen3-math:7b",
        note: "Latest math-specialized model",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-Math-7B-Instruct",
        command: "No login required",
        note: "Excellent for mathematical problem solving",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Qwen3-Math-7B-Instruct-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Top performance on math benchmarks",
      },
    },
  },
  "Qwen3-VL 7B": {
    parameters: "7B",
    min_ram_gb: 8,
    recommended_ram_gb: 16,
    min_vram_gb: 6,
    recommended_vram_gb: 8,
    cpu_only: true,
    domain: "Vision-Language",
    description: "Multimodal model for vision and language understanding",
    install_methods: {
      ollama: {
        command: "ollama run qwen3-vl:7b",
        note: "Latest vision-language model",
      },
      huggingface: {
        model_id: "Qwen/Qwen3-VL-7B-Instruct",
        command: "No login required",
        note: "Analyze images and answer questions",
      },
    },
  },
  "DeepSeek-Coder 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    domain: "Code Generation",
    description: "DeepSeek's specialized coding model",
    install_methods: {
      ollama: {
        command: "ollama run deepseek-coder:7b",
        note: "Strong coding performance",
      },
      huggingface: {
        model_id: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        command: "No login required",
        note: "Multi-language code generation",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/deepseek-coder-7b-instruct-v1.5-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Optimized for coding tasks",
      },
    },
  },
  "Granite 4 8B": {
    parameters: "8B",
    min_ram_gb: 8,
    recommended_ram_gb: 16,
    min_vram_gb: 5,
    recommended_vram_gb: 8,
    cpu_only: true,
    domain: "Enterprise/Business",
    description: "IBM's latest enterprise-grade model with strong instruction-following",
    install_methods: {
      ollama: {
        command: "ollama run granite4:8b",
        note: "Latest Granite generation",
      },
      huggingface: {
        model_id: "ibm-granite/granite-4-8b-instruct",
        command: "No login required",
        note: "Apache 2.0 licensed for commercial use",
      },
    },
  },
  "Llama 4 Vision 11B": {
    parameters: "11B",
    min_ram_gb: 10,
    recommended_ram_gb: 20,
    min_vram_gb: 7,
    recommended_vram_gb: 12,
    cpu_only: true,
    domain: "Vision-Language",
    description: "Meta's latest multimodal model with vision and tool-use",
    install_methods: {
      ollama: {
        command: "ollama run llama4-vision:11b",
        note: "Multimodal capabilities",
      },
      huggingface: {
        model_id: "meta-llama/Llama-4-11B-Vision-Instruct",
        command: "Requires HF login",
        note: "Vision and text understanding",
      },
    },
  },

  // ADDITIONAL SMALL MODELS (1-4B)
  "TinyLlama 1.1B": {
    parameters: "1.1B",
    min_ram_gb: 2,
    recommended_ram_gb: 4,
    min_vram_gb: 1,
    recommended_vram_gb: 2,
    cpu_only: true,
    description: "Ultra-compact Llama model for edge devices",
    install_methods: {
      ollama: {
        command: "ollama run tinyllama",
        note: "Perfect for resource-constrained environments",
      },
      huggingface: {
        model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        command: "No login required",
        note: "Runs on phones and tablets",
      },
    },
  },

  "Phi-3.5 Mini 3.8B": {
    parameters: "3.8B",
    min_ram_gb: 4,
    recommended_ram_gb: 8,
    min_vram_gb: 2,
    recommended_vram_gb: 4,
    cpu_only: true,
    description: "Microsoft's updated 3.8B model with improved performance",
    install_methods: {
      ollama: {
        command: "ollama run phi3.5:mini",
        note: "Latest Phi version",
      },
      huggingface: {
        model_id: "microsoft/Phi-3.5-mini-instruct",
        command: "No login required",
        note: "Enhanced capabilities over Phi-3",
      },
    },
  },

  "Qwen2.5 0.5B": {
    parameters: "0.5B",
    min_ram_gb: 1,
    recommended_ram_gb: 2,
    min_vram_gb: 0.5,
    recommended_vram_gb: 1,
    cpu_only: true,
    description: "Tiny but capable Qwen model for minimal hardware",
    install_methods: {
      ollama: {
        command: "ollama run qwen2.5:0.5b",
        note: "Ultra-lightweight",
      },
      huggingface: {
        model_id: "Qwen/Qwen2.5-0.5B-Instruct",
        command: "No login required",
        note: "Fastest inference",
      },
    },
  },

  "Qwen2.5 1.5B": {
    parameters: "1.5B",
    min_ram_gb: 2,
    recommended_ram_gb: 4,
    min_vram_gb: 1,
    recommended_vram_gb: 2,
    cpu_only: true,
    description: "Compact Qwen with good multilingual support",
    install_methods: {
      ollama: {
        command: "ollama run qwen2.5:1.5b",
        note: "Great for low-end systems",
      },
      huggingface: {
        model_id: "Qwen/Qwen2.5-1.5B-Instruct",
        command: "No login required",
        note: "Balanced size and capability",
      },
    },
  },

  "Qwen2.5 3B": {
    parameters: "3B",
    min_ram_gb: 4,
    recommended_ram_gb: 8,
    min_vram_gb: 2,
    recommended_vram_gb: 4,
    cpu_only: true,
    description: "Qwen 3B with enhanced reasoning",
    install_methods: {
      ollama: {
        command: "ollama run qwen2.5:3b",
        note: "Popular mid-small model",
      },
      huggingface: {
        model_id: "Qwen/Qwen2.5-3B-Instruct",
        command: "No login required",
        note: "Excellent for general tasks",
      },
    },
  },

  "StableLM 2 1.6B": {
    parameters: "1.6B",
    min_ram_gb: 2,
    recommended_ram_gb: 4,
    min_vram_gb: 1,
    recommended_vram_gb: 2,
    cpu_only: true,
    description: "Stability AI's compact language model",
    install_methods: {
      ollama: {
        command: "ollama run stablelm2:1.6b",
        note: "Good for creative writing",
      },
      huggingface: {
        model_id: "stabilityai/stablelm-2-1_6b",
        command: "No login required",
        note: "Trained on diverse data",
      },
    },
  },

  "Yi 6B": {
    parameters: "6B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "01.AI's bilingual model (English/Chinese)",
    install_methods: {
      ollama: {
        command: "ollama run yi:6b",
        note: "Strong bilingual performance",
      },
      huggingface: {
        model_id: "01-ai/Yi-6B-Chat",
        command: "No login required",
        note: "Excellent for Chinese tasks",
      },
    },
  },

  // MEDIUM MODELS (7-13B) ADDITIONS
  "Qwen2.5 14B": {
    parameters: "14B",
    min_ram_gb: 12,
    recommended_ram_gb: 24,
    min_vram_gb: 8,
    recommended_vram_gb: 12,
    cpu_only: false,
    description: "Mid-size Qwen with strong capabilities",
    install_methods: {
      ollama: {
        command: "ollama run qwen2.5:14b",
        note: "Best mid-size Qwen",
      },
      huggingface: {
        model_id: "Qwen/Qwen2.5-14B-Instruct",
        command: "No login required",
        note: "Excellent reasoning model",
      },
    },
  },

  "Mixtral 8x7B": {
    parameters: "47B",
    min_ram_gb: 32,
    recommended_ram_gb: 64,
    min_vram_gb: 24,
    recommended_vram_gb: 48,
    cpu_only: false,
    description: "Mistral's MoE model (only 12B active at once)",
    install_methods: {
      ollama: {
        command: "ollama run mixtral",
        note: "Mixture of Experts architecture",
      },
      huggingface: {
        model_id: "mistralai/Mixtral-8x7B-Instruct-v0.1",
        command: "No login required",
        note: "Sparse activation MoE",
      },
    },
  },

  "Nous Hermes 2 Mistral 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Fine-tuned Mistral for instruction-following",
    install_methods: {
      ollama: {
        command: "ollama run nous-hermes2:7b-mistral",
        note: "Excellent instruction adherence",
      },
      huggingface: {
        model_id: "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        command: "No login required",
        note: "DPO-trained for safety",
      },
    },
  },

  "OpenHermes 2.5 Mistral 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "High-quality Mistral fine-tune on diverse data",
    install_methods: {
      ollama: {
        command: "ollama run openhermes",
        note: "Versatile general-purpose model",
      },
      huggingface: {
        model_id: "teknium/OpenHermes-2.5-Mistral-7B",
        command: "No login required",
        note: "Strong across many domains",
      },
    },
  },

  "Zephyr 7B Beta": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Mistral fine-tuned for helpful conversations",
    install_methods: {
      ollama: {
        command: "ollama run zephyr:7b",
        note: "Great for chatbots",
      },
      huggingface: {
        model_id: "HuggingFaceH4/zephyr-7b-beta",
        command: "No login required",
        note: "DPO-aligned model",
      },
    },
  },

  "Solar 10.7B": {
    parameters: "10.7B",
    min_ram_gb: 10,
    recommended_ram_gb: 20,
    min_vram_gb: 7,
    recommended_vram_gb: 10,
    cpu_only: true,
    description: "Upstage's depth-upscaled model",
    install_methods: {
      ollama: {
        command: "ollama run solar:10.7b",
        note: "Unique depth-scaling approach",
      },
      huggingface: {
        model_id: "upstage/SOLAR-10.7B-Instruct-v1.0",
        command: "No login required",
        note: "Strong reasoning capabilities",
      },
    },
  },

  "Starling 7B Alpha": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "RLAIF-trained model with GPT-4 level responses",
    install_methods: {
      ollama: {
        command: "ollama run starling-lm",
        note: "High-quality responses",
      },
      huggingface: {
        model_id: "berkeley-nest/Starling-LM-7B-alpha",
        command: "No login required",
        note: "RLAIF alignment",
      },
    },
  },

  "Orca 2 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Microsoft's reasoning-focused model",
    install_methods: {
      ollama: {
        command: "ollama run orca2:7b",
        note: "Strong reasoning skills",
      },
      huggingface: {
        model_id: "microsoft/Orca-2-7b",
        command: "No login required",
        note: "Trained for step-by-step reasoning",
      },
    },
  },

  "Orca 2 13B": {
    parameters: "13B",
    min_ram_gb: 12,
    recommended_ram_gb: 24,
    min_vram_gb: 8,
    recommended_vram_gb: 12,
    cpu_only: false,
    description: "Larger Orca with better reasoning",
    install_methods: {
      ollama: {
        command: "ollama run orca2:13b",
        note: "Enhanced capabilities",
      },
      huggingface: {
        model_id: "microsoft/Orca-2-13b",
        command: "No login required",
        note: "Advanced reasoning model",
      },
    },
  },

  "Neural Chat 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Intel's optimized conversational model",
    install_methods: {
      ollama: {
        command: "ollama run neural-chat",
        note: "Optimized for Intel hardware",
      },
      huggingface: {
        model_id: "Intel/neural-chat-7b-v3-1",
        command: "No login required",
        note: "Great for chat applications",
      },
    },
  },

  "Falcon 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "TII's open-source model trained on diverse data",
    install_methods: {
      ollama: {
        command: "ollama run falcon:7b",
        note: "Strong multilingual support",
      },
      huggingface: {
        model_id: "tiiuae/falcon-7b-instruct",
        command: "No login required",
        note: "Apache 2.0 license",
      },
    },
  },

  "Yi 34B": {
    parameters: "34B",
    min_ram_gb: 24,
    recommended_ram_gb: 48,
    min_vram_gb: 20,
    recommended_vram_gb: 32,
    cpu_only: false,
    description: "01.AI's flagship bilingual model",
    install_methods: {
      ollama: {
        command: "ollama run yi:34b",
        note: "Top-tier bilingual performance",
      },
      huggingface: {
        model_id: "01-ai/Yi-34B-Chat",
        command: "No login required",
        note: "Best for Chinese+English",
      },
    },
  },
};
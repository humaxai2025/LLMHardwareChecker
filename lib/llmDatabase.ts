// LLM Database and Recommendation Engine
export interface InstallMethod {
  command?: string;
  note: string;
  model_id?: string;
  source?: string;
  recommended_quant?: string;
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
  // Small models (1-3B parameters)
  "Phi-3 Mini (3.8B)": {
    parameters: "3.8B",
    min_ram_gb: 4,
    recommended_ram_gb: 8,
    min_vram_gb: 2,
    recommended_vram_gb: 4,
    cpu_only: true,
    description: "Microsoft's efficient small model, great for basic tasks",
    install_methods: {
      ollama: {
        command: "ollama run phi3:mini",
        note: "Automatically downloads and runs",
      },
      huggingface: {
        model_id: "microsoft/Phi-3-mini-4k-instruct",
        command: "from transformers import AutoModelForCausalLM, AutoTokenizer",
        note: "Use with transformers library",
      },
      gguf: {
        source: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf",
        recommended_quant: "Q4_K_M",
        note: "For llama.cpp and similar tools",
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
    description: "Meta's latest small model with good performance",
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
  "Gemma 2B": {
    parameters: "2B",
    min_ram_gb: 3,
    recommended_ram_gb: 6,
    min_vram_gb: 1.5,
    recommended_vram_gb: 3,
    cpu_only: true,
    description: "Google's compact model, very efficient",
    install_methods: {
      ollama: {
        command: "ollama run gemma2:2b",
        note: "Latest Gemma 2 version",
      },
      huggingface: {
        model_id: "google/gemma-2-2b-it",
        command: "Requires HF login",
        note: "Need to accept license",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Excellent for low-resource systems",
      },
    },
  },

  // Medium models (7-8B parameters)
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
        note: "Most popular choice",
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
  "Mistral 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "High-quality model with strong performance",
    install_methods: {
      ollama: {
        command: "ollama run mistral:7b",
        note: "Well-optimized by Ollama team",
      },
      huggingface: {
        model_id: "mistralai/Mistral-7B-Instruct-v0.3",
        command: "No login required",
        note: "Open license model",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        recommended_quant: "Q4_K_M",
        note: "TheBloke's high-quality quantizations",
      },
    },
  },
  "Code Llama 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    description: "Specialized for code generation and analysis",
    install_methods: {
      ollama: {
        command: "ollama run codellama:7b",
        note: "For general coding tasks",
      },
      huggingface: {
        model_id: "codellama/CodeLlama-7b-Instruct-hf",
        command: "Requires HF login",
        note: "Instruction-tuned version",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Optimized for coding tasks",
      },
    },
  },

  // Large models (13-15B parameters)
  "Llama 3.1 13B": {
    parameters: "13B",
    min_ram_gb: 12,
    recommended_ram_gb: 24,
    min_vram_gb: 8,
    recommended_vram_gb: 12,
    cpu_only: false,
    description: "High-quality model requiring more resources",
    install_methods: {
      ollama: {
        command: "ollama run llama3.1:13b",
        note: "Will download ~7.4GB",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Meta-Llama-3.1-13B-Instruct-GGUF",
        recommended_quant: "Q4_K_M for 8GB VRAM, Q5_K_M for 12GB+",
        note: "Requires good GPU or lots of RAM",
      },
    },
  },
  "Vicuna 13B": {
    parameters: "13B",
    min_ram_gb: 12,
    recommended_ram_gb: 24,
    min_vram_gb: 8,
    recommended_vram_gb: 12,
    cpu_only: false,
    description: "Fine-tuned for conversations",
    install_methods: {
      huggingface: {
        model_id: "lmsys/vicuna-13b-v1.5",
        command: "No login required",
        note: "Popular chat model",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Good for extended conversations",
      },
    },
  },

  // Extra large models (30B+ parameters)
  "Llama 3.1 70B": {
    parameters: "70B",
    min_ram_gb: 48,
    recommended_ram_gb: 80,
    min_vram_gb: 40,
    recommended_vram_gb: 80,
    cpu_only: false,
    description: "State-of-the-art performance, requires high-end hardware",
    install_methods: {
      ollama: {
        command: "ollama run llama3.1:70b",
        note: "Requires 40GB+ free space and powerful hardware",
      },
      gguf: {
        source: "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        recommended_quant: "Q2_K for 24GB VRAM, Q4_K_M for 48GB+",
        note: "Consider cloud deployment for this size",
      },
    },
  },
  "Code Llama 34B": {
    parameters: "34B",
    min_ram_gb: 24,
    recommended_ram_gb: 48,
    min_vram_gb: 20,
    recommended_vram_gb: 24,
    cpu_only: false,
    description: "Advanced code model for complex programming tasks",
    install_methods: {
      ollama: {
        command: "ollama run codellama:34b",
        note: "Excellent for complex coding projects",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF",
        recommended_quant: "Q3_K_M for 16GB VRAM, Q4_K_M for 24GB+",
        note: "Professional-grade coding assistant",
      },
    },
  },

  // SPECIALIZED MODELS
  "StarCoder 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    domain: "Code Generation",
    description: "Multi-language code generation, supports 80+ programming languages",
    install_methods: {
      ollama: {
        command: "ollama run starcoder:7b",
        note: "Excellent for diverse programming languages",
      },
      huggingface: {
        model_id: "bigcode/starcoder",
        command: "No login required",
        note: "Open source code generation model",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/starcoder-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Optimized for code completion and generation",
      },
    },
  },
  "BioMistral 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    domain: "Medical/Biology",
    description: "Medical knowledge model for healthcare professionals and researchers",
    install_methods: {
      huggingface: {
        model_id: "BioMistral/BioMistral-7B",
        command: "No login required",
        note: "Specialized for medical and biological text",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/BioMistral-7B-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Optimized for medical Q&A and research",
      },
    },
  },
  "MetaMath 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    domain: "Mathematics",
    description: "Mathematical reasoning and problem-solving specialist",
    install_methods: {
      huggingface: {
        model_id: "meta-math/MetaMath-7B-V1.0",
        command: "No login required",
        note: "Excellent for mathematical problem solving",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/MetaMath-7B-V1.0-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Optimized for step-by-step math solutions",
      },
    },
  },
  "WizardCoder 7B": {
    parameters: "7B",
    min_ram_gb: 6,
    recommended_ram_gb: 12,
    min_vram_gb: 4,
    recommended_vram_gb: 6,
    cpu_only: true,
    domain: "Code Generation",
    description: "Advanced coding assistant with strong reasoning capabilities",
    install_methods: {
      huggingface: {
        model_id: "WizardLM/WizardCoder-7B-V1.0",
        command: "No login required",
        note: "Enhanced code generation and debugging",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/WizardCoder-7B-V1.0-GGUF",
        recommended_quant: "Q4_K_M",
        note: "Strong performance on coding benchmarks",
      },
    },
  },
  "Nous Hermes 2 - Solar 10.7B": {
    parameters: "10.7B",
    min_ram_gb: 8,
    recommended_ram_gb: 16,
    min_vram_gb: 6,
    recommended_vram_gb: 8,
    cpu_only: true,
    domain: "Research/Analysis",
    description: "Research and analysis specialist with strong reasoning",
    install_methods: {
      ollama: {
        command: "ollama run nous-hermes2-solar:10.7b",
        note: "Excellent for research and analytical tasks",
      },
      huggingface: {
        model_id: "NousResearch/Nous-Hermes-2-SOLAR-10.7B",
        command: "No login required",
        note: "High-quality reasoning and analysis",
      },
      gguf: {
        source: "https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF",
        recommended_quant: "Q4_K_M for 8GB VRAM, Q5_K_M for 12GB+",
        note: "Great for complex analytical tasks",
      },
    },
  },
};
// types/index.ts - Unified type definitions for the entire application

export interface SystemInfo {
  os: string;
  architecture: string;
  processor: string;
  cpuCores: number;
  totalRamGB: number;
  availableRamGB: number;
  totalStorageGB: number;
  freeStorageGB: number;
  gpus?: Array<{
    name: string;
    vramGB: number | string;
    type?: string;
  }>;
  // Browser detection fields (optional)
  userAgent?: string;
  screenResolution?: string;
  colorDepth?: number;
  language?: string;
  timezone?: string;
}

export interface InstallationMethod {
  ollama?: {
    command: string;
    note: string;
  };
  huggingface?: {
    model_id: string;
    note: string;
  };
  gguf?: {
    source: string;
    recommended_quant: string;
    note: string;
  };
  lmstudio?: {
    note: string;
    download_url?: string;
  };
  llamacpp?: {
    note: string;
    compile_instructions?: string;
  };
}

export interface ModelSpecs {
  parameters: string;
  min_ram_gb: number;
  recommended_ram_gb: number;
  min_vram_gb: number;
  recommended_vram_gb: number;
  description: string;
  domain?: string;
  quantization_options?: string[];
  use_cases?: string[];
  install_methods: InstallationMethod;
}

export interface ModelCompatibility {
  performance_tier: 'excellent' | 'good' | 'basic' | 'not_suitable';
  recommended_quant?: string;
  notes: string[];
  estimated_speed?: string;
  memory_usage?: string;
}

export interface ModelRecommendation {
  name: string;
  specs: ModelSpecs;
  compatibility: ModelCompatibility;
  category?: 'general' | 'coding' | 'creative' | 'chat' | 'specialized';
}

export interface Recommendations {
  excellent: ModelRecommendation[];
  good: ModelRecommendation[];
  basic: ModelRecommendation[];
  not_suitable: ModelRecommendation[];
}

export interface InstallationPlatform {
  name: string;
  description: string;
  difficulty: 'Easy' | 'Intermediate' | 'Advanced';
  bestFor: string;
  installation: Record<string, string>;
  website?: string;
  pros?: string[];
  cons?: string[];
}

export interface LLMRecommender {
  getRecommendations(): Recommendations;
  getSystemCapabilityLevel(): 'low' | 'medium' | 'high' | 'premium';
  getInstallationPlatforms(): InstallationPlatform[];
  getOptimizationTips(): string[];
  getBestModelsForUseCase(useCase: string): ModelRecommendation[];
  getMemoryEstimates(): {
    available: number;
    recommended_usage: number;
    max_model_size: number;
  };
}

export interface ReportData {
  systemInfo: SystemInfo;
  recommendations: Recommendations;
  recommender: LLMRecommender;
  timestamp: Date;
}

// Analysis state interfaces
export interface AnalysisState {
  systemInfo: SystemInfo | null;
  recommendations: Recommendations | null;
  recommender: LLMRecommender | null;
  isLoading: boolean;
  isAnalyzing: boolean;
  error: string | null;
  analysisComplete: boolean;
  showManualInput: boolean;
  browserDetected: Partial<SystemInfo> | null;
}

// Component prop interfaces
export interface SystemSpecsCardProps {
  systemInfo: SystemInfo;
}

export interface ModelRecommendationsProps {
  recommendations: Recommendations;
  systemInfo: SystemInfo;
}

export interface InstallationGuideProps {
  recommender: LLMRecommender;
}

export interface OptimizationTipsProps {
  recommender: LLMRecommender;
}

export interface ReportDownloadProps {
  systemInfo: SystemInfo;
  recommendations: Recommendations;
  recommender: LLMRecommender;
}

export interface ManualHardwareInputProps {
  onComplete: (systemInfo: SystemInfo) => void;
  browserDetected?: Partial<SystemInfo> | null;
}

// Utility types
export type SystemCapabilityLevel = 'low' | 'medium' | 'high' | 'premium';
export type PerformanceTier = 'excellent' | 'good' | 'basic' | 'not_suitable';
export type ModelCategory = 'general' | 'coding' | 'creative' | 'chat' | 'specialized';
export type InstallationDifficulty = 'Easy' | 'Intermediate' | 'Advanced';

// Report generation types
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

export interface BrowserCompatibility {
  canGenerateHTML: boolean;
  canGeneratePDF: boolean;
  canCopyToClipboard: boolean;
  issues: string[];
}

// Error types
export class LLMAnalysisError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly details?: any
  ) {
    super(message);
    this.name = 'LLMAnalysisError';
  }
}

export class SystemDetectionError extends LLMAnalysisError {
  constructor(message: string, details?: any) {
    super(message, 'SYSTEM_DETECTION_ERROR', details);
    this.name = 'SystemDetectionError';
  }
}

export class RecommendationError extends LLMAnalysisError {
  constructor(message: string, details?: any) {
    super(message, 'RECOMMENDATION_ERROR', details);
    this.name = 'RecommendationError';
  }
}

// Constants
export const PERFORMANCE_TIERS: Record<PerformanceTier, { label: string; color: string; description: string }> = {
  excellent: {
    label: 'Excellent Performance',
    color: 'green',
    description: 'Runs smoothly with fast response times'
  },
  good: {
    label: 'Good Performance', 
    color: 'blue',
    description: 'Runs well with acceptable response times'
  },
  basic: {
    label: 'Basic Performance',
    color: 'yellow', 
    description: 'Runs but may be slower'
  },
  not_suitable: {
    label: 'Not Suitable',
    color: 'red',
    description: 'Insufficient hardware for this model'
  }
};

export const CAPABILITY_LEVELS: Record<SystemCapabilityLevel, { label: string; color: string; minRam: number }> = {
  low: {
    label: 'Entry Level',
    color: 'orange',
    minRam: 8
  },
  medium: {
    label: 'Mid Range', 
    color: 'blue',
    minRam: 16
  },
  high: {
    label: 'High End',
    color: 'green', 
    minRam: 32
  },
  premium: {
    label: 'Premium',
    color: 'purple',
    minRam: 64
  }
};

// Default values
export const DEFAULT_SYSTEM_INFO: Partial<SystemInfo> = {
  os: 'Unknown',
  architecture: 'Unknown',
  processor: 'Unknown',
  cpuCores: 4,
  totalRamGB: 8,
  availableRamGB: 6,
  totalStorageGB: 256,
  freeStorageGB: 100,
  gpus: []
};

export const DEFAULT_RECOMMENDATIONS: Recommendations = {
  excellent: [],
  good: [],
  basic: [],
  not_suitable: []
};